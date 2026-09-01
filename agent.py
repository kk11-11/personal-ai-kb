# agent.py
# 手写 ReAct 循环: 思考(Thought) -> 行动(Action/工具调用) -> 观察(Observation) -> 循环
#
# 为什么不用框架: 框架会把"循环、解析、执行"藏在黑盒里, 出问题无从排查。
# 这里全部手写, 每一步都看得见, 也是面试时能讲清细节的前提。
#
# 关键分工: 模型只负责"请求调用某个工具并给出参数", 真正执行工具的是下面这些普通 Python 函数。

import json
import re

# 多轮对话: 最近保留轮数(超过的部分先被 LLM 摘要压缩, 而非直接丢弃)
# 调大 4->6 是为了应对长对话里仍要精准定位最早那几轮内容, 避免摘要压缩后丢失关键概念
HISTORY_RECENT = 6


def summarize_history(history, llm, model="glm-4-flash", keep_recent=HISTORY_RECENT):
    """长对话上下文压缩(Context Engineering 核心手段之一)。

    对话超过 keep_recent 轮时, 把更早的轮次用 LLM 压成一段摘要, 只把
    [摘要 + 最近 keep_recent 轮] 拼进 prompt。既保留长程上下文, 又控制 token 成本。

    返回 (summary_or_None, recent_turns)
    - 未超阈值: (None, 原始 history) —— 零额外调用, 与原固定窗口行为一致
    - 超阈值且摘要成功: (摘要文本, 最近 keep_recent 轮)
    - 超阈值但摘要失败: (None, 最近 keep_recent 轮) —— 退化为原行为, 不阻断主流程
    """
    if not history or len(history) <= keep_recent:
        return None, history
    old = history[:-keep_recent]
    recent = history[-keep_recent:]
    # 摘要里按"全局轮次编号"对齐, 这样后续『第 1 轮』『第一个问题』这类相对指代能精准对回历史
    old_lines = []
    # old 在 history 里是从头开始的早期轮次, 编号从 1 起算
    for i, h in enumerate(old, start=1):
        old_lines.append(
            f"[第 {i} 轮] 用户: {h.get('q', '')}\n助手: {h.get('a', '')}"
        )
    old_text = "\n\n".join(old_lines)
    prompt = (
        "请把下面这段对话历史压缩成一段简洁摘要(中文, 300 字以内)。\n"
        "保留: 用户已确认的事实、已做出的决定、已解决的关键问题, 以及对后续对话重要的上下文"
        "(例如提到的具体文件 / 项目 / 偏好)。\n"
        "**必须按格式保留每轮的全局轮次编号**: '[第 N 轮] 用户问: ... 助手答: ...'\n"
        "丢弃: 寒暄、重复内容、可被最近对话覆盖的细节。\n"
        "不要编造新信息。\n\n"
        f"【对话历史】\n{old_text}\n\n【摘要】"
    )
    try:
        resp = llm.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        summary = (resp.choices[0].message.content or "").strip()
        return (summary or None), recent
    except Exception as e:
        print("⚠️ 历史摘要失败, 退化为固定窗口:", e)
        return None, recent


def build_tools(embedder, get_collection):
    """构造工具表。

    get_collection 传的是**函数**而不是 collection 对象:
    因为 app.py 的 reingest() 会把全局 collection 重指向新建的集合,
    如果这里捕获了对象, 重建知识库后工具就会拿着失效的旧句柄查询而报错。
    """

    def list_sources(args):
        """列出知识库中所有文档及其片段数。"""
        coll = get_collection()
        if coll.count() == 0:
            return "知识库当前是空的, 还没有任何文档。"
        try:
            metas = coll.get().get("metadatas") or []
        except Exception as e:
            return f"读取知识库失败: {e}"
        counter = {}
        for m in metas:
            counter[m["source"]] = counter.get(m["source"], 0) + 1
        if not counter:
            return "知识库当前是空的。"
        lines = [f"- {name}（{n} 个片段）" for name, n in sorted(counter.items())]
        return "知识库共有以下文档:\n" + "\n".join(lines)

    def search_by_source(args):
        """在指定文档中检索内容。filename 必须是 list_sources 返回的真实文件名。"""
        filename = (args.get("filename") or "").strip()
        keyword = (args.get("keyword") or "").strip()
        coll = get_collection()
        if not filename:
            return "错误: 必须提供 filename。可以先用 list_sources 查看有哪些文档。"
        if coll.count() == 0:
            return "知识库是空的, 无法检索。"
        try:
            if keyword:
                # 注意: 本项目的 collection 入库时是显式传入 embeddings 的,
                # 没有绑定 embedding_function, 所以不能用 query_texts, 必须自己编码。
                q_emb = embedder.encode([keyword]).tolist()
                res = coll.query(
                    query_embeddings=q_emb, n_results=3, where={"source": filename}
                )
                docs = res["documents"][0]
                metas = res["metadatas"][0]
            else:
                res = coll.get(where={"source": filename})
                docs = (res.get("documents") or [])[:5]
                metas = (res.get("metadatas") or [])[:5]
        except Exception as e:
            return f"检索失败: {e}"
        if not docs:
            return f"在 {filename} 中没有找到相关内容（可能是文件名不对或未命中）。"
        out = []
        for d, m in zip(docs, metas):
            out.append(f"[{m['source']}#{m['chunk']}] {d[:300]}")
        return "\n\n".join(out)

    tools = {
        "list_sources": {
            "fn": list_sources,
            "schema": {
                "type": "function",
                "function": {
                    "name": "list_sources",
                    "description": "列出知识库中所有文档的名称和片段数。当用户问'有哪些文档''哪几篇讲了X'时, 应该先调用它。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        },
        "search_by_source": {
            "fn": search_by_source,
            "schema": {
                "type": "function",
                "function": {
                    "name": "search_by_source",
                    "description": "在指定文档中检索内容。filename 必须是 list_sources 返回的真实文件名, keyword 是要找的关键词或问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "目标文件名, 例如 notes.md",
                            },
                            "keyword": {
                                "type": "string",
                                "description": "要检索的关键词或问题, 可留空表示取该文档前几个片段",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
        },
    }
    return tools


def _exec_tool(tools, name, args):
    """执行工具。关键: 任何异常都转成字符串返回, 让模型看到错误并自行纠错, 而不是让整条链路崩掉。"""
    tool = tools.get(name)
    if tool is None:
        return f"错误: 没有名为 '{name}' 的工具。可用工具: {', '.join(tools.keys())}"
    try:
        return str(tool["fn"](args or {}))
    except Exception as e:
        return f"工具 '{name}' 执行出错: {e}"


def _parse_text_action(text):
    """兜底: 解析文本协议 'Action: tool_name({"k": "v"})', 供不支持原生 function calling 的模型使用。"""
    m = re.search(r"Action\s*[:：]\s*([A-Za-z_]\w*)\s*(\([^\n]*\))?", text, re.S)
    if not m:
        return None
    name = m.group(1)
    raw = (m.group(2) or "()").strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    if not raw:
        return (name, {})
    for candidate in (raw, raw.replace("'", '"')):
        try:
            return (name, json.loads(candidate))
        except Exception:
            continue
    return (name, {"_raw": raw})


def run_react(question, llm, tools, model="glm-4-flash", max_steps=4, history=None):
    """手写 ReAct 循环。

    返回 (final_answer, trace)。
    trace 记录每一步调用了什么工具、传了什么参数、拿到什么结果 —— 排错和评测都靠它。
    """
    schemas = [t["schema"] for t in tools.values()]
    system = (
        "你是用户的个人知识库助手, 可以使用工具。\n"
        "按 ReAct 方式工作: 先想清楚需要什么信息(Thought), 再决定调用工具(Action)还是直接回答。\n"
        f"规则:\n"
        f"1. 不知道有哪些文档时先调用 list_sources; 需要在某篇文档里查内容时调用 search_by_source。\n"
        f"2. 一次结果不够可以继续调用其他工具, 但总共最多 {max_steps} 轮。\n"
        f"3. 信息足够时直接给最终答案, 不要再调用工具。\n"
        f"4. 答案必须基于工具返回的内容, 工具没返回的东西不许编造; 确实没有就说没找到。\n"
        f"5. 若看到以 [早期对话摘要] 开头的内容, 那是对更早对话的压缩, 结合它来理解当前问题。\n"
    )
    messages = [{"role": "system", "content": system}]
    if history:
        summary, recent = summarize_history(history, llm, model=model)
        if summary:
            messages.append(
                {"role": "user", "content": f"[早期对话摘要]\n{summary}"}
            )
        for h in recent:
            messages.append({"role": "user", "content": h.get("q", "")})
            messages.append({"role": "assistant", "content": h.get("a", "")})
    messages.append({"role": "user", "content": question})

    trace = []
    final = None

    for step in range(1, max_steps + 1):
        try:
            resp = llm.chat.completions.create(
                model=model, messages=messages, tools=schemas, tool_choice="auto"
            )
        except Exception as e:
            # 接口不支持 tools 参数 -> 降级为普通对话, 保证可用性
            try:
                resp = llm.chat.completions.create(model=model, messages=messages)
                final = (resp.choices[0].message.content or "").strip()
                trace.append({"step": step, "type": "no_tools_fallback", "note": str(e)})
            except Exception as e2:
                return (f"调用模型失败: {e2}", trace)
            break

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            # 原生 function calling 路径: 模型请求调用工具
            messages.append(msg)
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result = _exec_tool(tools, name, args)
                trace.append(
                    {"step": step, "tool": name, "args": args, "result": result[:400]}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            continue

        content = (msg.content or "").strip()
        parsed = _parse_text_action(content)
        if parsed:
            # 文本协议兜底路径
            name, args = parsed
            result = _exec_tool(tools, name, args)
            trace.append(
                {"step": step, "tool": name, "args": args, "result": result[:400]}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation（{name} 的返回）:\n{result}\n\n如果信息还不够可以继续调用工具, 否则直接给出最终答案。",
                }
            )
            continue

        # 没有请求任何工具 -> 这一轮的输出就是最终答案
        final = content
        break

    if final is None:
        # 达到轮数上限: 强制收尾, 让它基于已有观察作答, 而不是无限转圈烧 token
        messages.append(
            {
                "role": "user",
                "content": "已达到最大工具调用轮数, 请基于上面已有的信息直接给出最终答案。",
            }
        )
        try:
            resp = llm.chat.completions.create(model=model, messages=messages)
            final = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            final = f"生成最终答案失败: {e}"
        trace.append({"step": max_steps, "type": "max_steps_reached"})

    return final, trace
