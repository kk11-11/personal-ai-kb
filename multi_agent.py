# multi_agent.py
# 在已有「单 Agent RAG」(langgraph_agent.run_react_langgraph) 之上, 用 LangGraph 加一层
# 「主管 Supervisor + 专职工人」的多 Agent 编排 —— 对应视频技术栈里的 DeepAgent / 多 Agent。
#
# 架构(5 个节点):
#   supervisor       主管: 看全局进度, 决定下一步交给哪个工人, 或收尾
#   planning_worker  规划工人: 把复杂问题拆成子问题(query decomposition), 提升检索覆盖
#   retrieval_worker 检索工人: 按"原问题+子问题"多路检索, 复用现有 RAG ReAct 循环
#   writing_worker   写作工人: 把检索结果改写成流畅、结构化的答案
#   verification_worker 校验工人: 核对答案是否忠于知识库、有无遗漏; 不合格打回写作
#
# 为什么这样设计:
#   单 Agent 是「一个问题 → 直接答」。多 Agent 是「一个问题 → 主管调度 → 规划拆解 →
#   多路检索 → 专人写作 → 专人校验 → 不合格重写」。每个环节各司其职, 主管只管调度与纠错,
#   这正是"体系化项目"和单 demo 拉开差距的地方。
#
# 复用关系:
#   - retrieval_worker 直接调用 langgraph_agent.run_react_langgraph(...)  —— 你 W1 的成果原样复用
#   - build_tools / summarize_history 来自 agent.py  —— 业务逻辑零改动
#   - 本文件只新增「编排层 + 写作/校验两个轻量 LLM 节点」

import json
import os
import sys

from langchain_core.messages import convert_to_openai_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

DEFAULT_MODEL = "glm-4-flash"


# ====== State ======
# 比单 Agent 多了几个"工人之间的交接字段": retrieval / draft / verification / next / iter
def _state_schema():
    from typing import Annotated, Sequence, TypedDict

    class MultiState(TypedDict):
        messages: Annotated[Sequence, add_messages]   # 全流程决策日志(可观测)
        question: str                                  # 用户原始问题
        plan: str                                      # 规划工人的产出(JSON 子问题列表; "" = 尚未规划)
        retrieval: str                                 # 检索工人的产出
        draft: str                                     # 写作工人的产出(最终答案来源)
        verification: str                              # 校验工人的意见
        next: str                                      # 主管决定的下一步(驱动条件边)
        last_worker: str                               # 上一个干活的工人(用于幂等校验, 防重复/死循环)
        iter: int                                      # 主循环计数, 防无限打回
        rewrites: int                                  # 已打回重写次数(校验 FAIL -> writing)

    return MultiState


# 主管用来"选下一个工人"的结构化工具(强制它返回枚举决策, 而不是自由文本)
ROUTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "route",
        "description": "根据当前进度, 决定下一步交给哪个专职工人, 或收尾。",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": ["planning", "retrieval", "writing", "verification", "finish"],
                    "description": "下一个执行的工人; finish 表示可以输出最终答案",
                },
                "reason": {"type": "string", "description": "一句话决策理由"},
            },
            "required": ["next"],
        },
    },
}

# 最多打回重写几次(防止写作↔校验死循环)
MAX_REWRITE = 2
# 复杂问题最多拆成几个子问题(防止检索轮数爆炸, 直接决定单次问答成本)
MAX_SUBQ = 3


def build_multi_agent_graph(llm, tools, model=DEFAULT_MODEL, summarize_fn=None):
    """把多 Agent 协作编译成一张 LangGraph 图。

    tools 直接复用 agent.build_tools() 产物; retrieval_worker 内部调用
    langgraph_agent.run_react_langgraph。
    """
    from langgraph_agent import run_react_langgraph

    MultiState = _state_schema()

    def _rule_route(state):
        """规则兜底: LLM 决策缺失/解析失败/不合法时, 按流程进度确定性路由(保证系统永不崩)。

        这层是工程上的关键: LLM 作为"调度优化", 规则作为"正确性底线"。
        """
        last = state.get("last_worker", "")
        if state.get("plan", "") == "":        # "" 表示还没规划过
            return "planning"
        if not state.get("retrieval"):
            return "retrieval"
        if not state.get("draft"):
            return "writing"
        if last == "writing":                  # 刚写完 -> 必须去校验, 不能原地重写
            return "verification"
        if not state.get("verification"):
            return "verification"
        if "FAIL" in state.get("verification", "").upper() and state.get("rewrites", 0) < MAX_REWRITE:
            return "writing"
        return "finish"

    def _is_valid(decision, state):
        """幂等校验: LLM 说的这一步, 对当前进度是否合法。

        没有这层会出真事故 —— 实测 glm-4-flash 连续 10 次都回 "planning",
        主管照办就变成"无限规划": 检索永远不执行, 最终答案为空。
        """
        last = state.get("last_worker", "")
        if decision == "planning":
            return state.get("plan", "") == ""          # 已规划过就不再规划
        if decision == "retrieval":
            return bool(state.get("plan")) and not state.get("retrieval")   # 先规划再检索, 且不重复
        if decision == "writing":
            return bool(state.get("retrieval")) and last != "writing"       # 有料才写, 不连写
        if decision == "verification":
            return bool(state.get("draft")) and last != "verification"      # 有草稿才校验, 不连校
        return decision == "finish" and bool(state.get("draft"))            # 没草稿不许收尾

    def _parse_route(resp):
        """解析 LLM 的路由决策。三重容错: tool_calls -> 文本关键词 -> None(交给规则)。"""
        msg = resp.choices[0].message
        # 第一优先: 标准 tool_calls(坑: 智谱 glm-4-flash 在强制 tool_choice 下
        # 仍可能返回 tool_calls=None 的纯文本, 直接 tool_calls[0] 会 TypeError)
        calls = getattr(msg, "tool_calls", None)
        if calls:
            try:
                nxt = json.loads(calls[0].function.arguments).get("next")
                if nxt in ("planning", "retrieval", "writing", "verification", "finish"):
                    return nxt, "tool_calls"
            except Exception:
                pass
        # 第二优先: 从纯文本回复里找关键词
        content = msg.content or ""
        for kw in ("planning", "retrieval", "writing", "verification", "finish"):
            if kw in content.lower():
                return kw, "text"
        return None, "none"

    def supervisor(state):
        """主管节点: 看全局进度决定下一步。带 iter 护栏, 防止无限打回。"""
        if state.get("iter", 0) >= MAX_REWRITE * 2 + 6:
            return {"next": "finish", "messages": [{"role": "assistant",
                    "content": f"[主管] 已达最大轮数, 强制收尾。"}]}

        sys_prompt = (
            "你是多 Agent 系统的主管。系统包含四个专职工人: "
            "planning(把复杂问题拆成子问题)、retrieval(检索知识库)、"
            "writing(把检索结果写成答案)、verification(校验答案质量)。\n"
            "标准流程: planning -> retrieval -> writing -> verification。\n"
            "verification 若判定答案不合格(含 FAIL), 应回到 writing 重写(最多打回 "
            f"{MAX_REWRITE} 次); 若合格(含 PASS)或已打回足够多次, 则 finish。\n"
            "请基于下面的进度调用 route 工具, 只需返回 next 字段。"
        )
        msgs = [{"role": "system", "content": sys_prompt}]
        if state.get("question"):
            msgs.append({"role": "user", "content": f"用户原始问题: {state['question']}"})
        if state.get("retrieval"):
            msgs.append({"role": "assistant", "content": f"[检索结果] {state['retrieval'][:1500]}"})
        if state.get("draft"):
            msgs.append({"role": "assistant", "content": f"[当前草稿] {state['draft'][:1500]}"})
        if state.get("verification"):
            msgs.append({"role": "assistant", "content": f"[校验意见] {state['verification'][:800]}"})

        decision, source, reason = None, "rule", ""
        try:
            # tool_choice 用 "auto" 而非强制指定函数: 部分国产模型不兼容强制格式,
            # 会直接回纯文本(tool_calls=None) —— 与 parse 的容错配合使用。
            resp = llm.chat.completions.create(
                model=model, messages=msgs, tools=[ROUTE_SCHEMA], tool_choice="auto",
            )
            decision, source = _parse_route(resp)
        except Exception as e:
            reason = f"LLM 决策异常({e}), "

        if decision is None:
            decision = _rule_route(state)
            reason += "LLM 未给出有效决策, 按流程规则兜底"
        elif not _is_valid(decision, state):
            decision = _rule_route(state)
            reason = "LLM 决策与当前进度冲突(重复/已完成), 已纠正为规则路由"
        else:
            reason = {"tool_calls": "route 工具决策", "text": "从文本提取决策"}.get(source, "")

        # 打回重写计数: 校验已存在却再次决策 writing, 记一次 rewrite
        rewrites = state.get("rewrites", 0)
        if decision == "writing" and state.get("verification"):
            rewrites += 1

        return {
            "next": decision,
            "iter": state.get("iter", 0) + 1,
            "rewrites": rewrites,
            "messages": [{"role": "assistant",
                          "content": f"[主管] 决策 -> {decision} ({reason})"}],
        }

    def planning_worker(state):
        """规划工人: 把复杂问题拆成若干子问题, 供检索工人逐条检索。

        这是 query decomposition(查询分解): 单个问题往往覆盖不全知识库里的多个角度,
        拆开检索能显著提升召回覆盖。简单问题返回空列表, 不额外增加开销。
        """
        sys_prompt = (
            "你是规划工人。判断用户问题是否需要拆解以便检索。\n"
            "若问题简单、单一明确 -> 只输出: {\"sub_questions\": []}\n"
            "若问题复杂(涉及多个方面/需要对比/需要多个知识点) -> 输出最多 3 个子问题: "
            "{\"sub_questions\": [\"子问题1\", \"子问题2\"]}\n"
            "只输出 JSON, 不要其他文字。"
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"用户问题: {state['question']}"},
        ]
        subq, note = [], "无需拆解"
        try:
            resp = llm.chat.completions.create(model=model, messages=msgs)
            content = resp.choices[0].message.content or ""
            # 容错: 从回复里抠出第一个 JSON 对象(模型常爱加解释文字)
            start, end = content.find("{"), content.rfind("}")
            data = json.loads(content[start:end + 1]) if start >= 0 and end > start else {}
            raw = data.get("sub_questions") or []
            if isinstance(raw, list):
                subq = [str(x).strip() for x in raw if str(x).strip()][:MAX_SUBQ]
        except Exception as e:
            note = f"规划失败({e}), 按原问题检索"
        if subq:
            note = f"拆解为 {len(subq)} 个子问题"
        return {
            "plan": json.dumps(subq, ensure_ascii=False),   # "" -> "[]"/"[...]" 表示已规划
            "last_worker": "planning",
            "messages": [{"role": "assistant", "content": f"[规划工人] {note}: {subq}"}],
        }

    def retrieval_worker(state):
        """检索工人: 复用 W1 的 RAG ReAct 循环, 按"原问题 + 各子问题"多路检索后汇总。"""
        subq = []
        try:
            subq = json.loads(state.get("plan") or "[]")
            if not isinstance(subq, list):
                subq = []
        except Exception:
            subq = []

        # 原问题先查一次(保证不丢主干), 再逐条查子问题
        queries = [state["question"]] + [q for q in subq if q != state["question"]][:MAX_SUBQ]
        parts = []
        for q in queries:
            answer, _trace = run_react_langgraph(
                q, llm, tools, model=model, summarize_fn=summarize_fn
            )
            if answer:
                label = "原问题" if q == state["question"] else f"子问题: {q}"
                parts.append(f"【{label}】\n{answer}" if subq else str(answer))
        merged = "\n\n".join(parts)
        return {
            "retrieval": merged,
            "last_worker": "retrieval",
            "messages": [{"role": "assistant",
                          "content": f"[检索工人] {len(queries)} 路检索完成: {merged[:200]}"}],
        }

    def writing_worker(state):
        """写作工人: 基于检索结果, 生成流畅、结构化的答案。"""
        sys_prompt = (
            "你是写作工人。请基于『检索结果』, 针对『用户问题』撰写一段清晰、结构化、"
            "可直接交付的答案。只使用检索结果中的事实, 不要编造。若检索结果不足以回答, "
            "就如实说明缺失部分。"
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"用户问题: {state['question']}"},
            {"role": "user", "content": f"检索结果: {state.get('retrieval', '')}"},
        ]
        resp = llm.chat.completions.create(model=model, messages=msgs)
        draft = resp.choices[0].message.content or ""
        return {"draft": draft, "last_worker": "writing",
                "messages": [{"role": "assistant", "content": f"[写作工人] 草稿: {draft}"}]}

    def verification_worker(state):
        """校验工人: 核对草稿是否忠于检索结果、是否完整。输出 PASS 或 FAIL+原因。"""
        sys_prompt = (
            "你是校验工人。对比『检索结果』与『当前草稿』, 判断草稿是否: "
            "(1) 忠于检索结果(无编造)、(2) 完整回应了用户问题。\n"
            "若通过, 第一行写 PASS; 若不通过, 第一行写 FAIL: 并说明缺了什么/哪里编造了。"
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"用户问题: {state['question']}"},
            {"role": "user", "content": f"检索结果: {state.get('retrieval', '')}"},
            {"role": "user", "content": f"当前草稿: {state.get('draft', '')}"},
        ]
        resp = llm.chat.completions.create(model=model, messages=msgs)
        verdict = resp.choices[0].message.content or ""
        return {"verification": verdict, "last_worker": "verification",
                "messages": [{"role": "assistant", "content": f"[校验工人] {verdict}"}]}

    graph = StateGraph(MultiState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("planning", planning_worker)
    graph.add_node("retrieval", retrieval_worker)
    graph.add_node("writing", writing_worker)
    graph.add_node("verification", verification_worker)

    graph.set_entry_point("supervisor")
    # 主管决定下一步(条件边)
    graph.add_conditional_edges(
        "supervisor", lambda s: s["next"],
        {"planning": "planning", "retrieval": "retrieval", "writing": "writing",
         "verification": "verification", "finish": END},
    )
    # 每个工人干完都回到主管, 由主管再决策(形成协作+打回闭环)
    graph.add_edge("planning", "supervisor")
    graph.add_edge("retrieval", "supervisor")
    graph.add_edge("writing", "supervisor")
    graph.add_edge("verification", "supervisor")

    return graph.compile(checkpointer=MemorySaver())


_GRAPH_CACHE = {}


def get_multi_agent_graph(llm, tools, model=DEFAULT_MODEL, summarize_fn=None):
    # key 带 id(llm): 与单 Agent 同样的坑(闭包捕获旧实例)
    key = (id(llm), model, tuple(sorted(tools.keys())))
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = build_multi_agent_graph(
            llm, tools, model=model, summarize_fn=summarize_fn
        )
    return _GRAPH_CACHE[key]


def run_multi_agent(question, llm, tools, model=DEFAULT_MODEL, summarize_fn=None,
                    thread_id="default", max_steps=12):
    """多 Agent 入口, 返回 (final_answer, trace)。

    trace 每一项标注了哪个工人干了什么, 方便你观察"主管调度 + 工人协作"的全过程。
    """
    graph = get_multi_agent_graph(llm, tools, model=model, summarize_fn=summarize_fn)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps * 4 + 10}

    init_state = {
        "messages": [{"role": "user", "content": f"用户问题: {question}"}],
        "question": question,
        "plan": "",
        "retrieval": "",
        "draft": "",
        "verification": "",
        "next": "",
        "last_worker": "",
        "iter": 0,
        "rewrites": 0,
    }

    trace = []
    final = None
    for event in graph.stream(init_state, config, stream_mode="updates"):
        for node_name, payload in event.items():
            new_msgs = payload.get("messages") or []
            for m in new_msgs:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if role == "assistant" and content:
                    trace.append({"worker": node_name, "log": content[:200]})

    # 取最终答案: 优先 draft, 兜底从 checkpoint 取
    snapshot = graph.get_state(config)
    for m in reversed(snapshot.values.get("messages", [])):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role == "assistant" and content and ("[写作工人]" in content or "[主管]" in content):
            # 优先取写作工人的草稿作为答案
            if "[写作工人]" in content:
                final = content.split("[写作工人]", 1)[-1].strip()
                break
    if not final:
        # 兜底1: 直接用 state 里的 draft
        final = snapshot.values.get("draft") or ""
    if not final:
        # 兜底2: 写作工人没跑完(如被护栏强制收尾), 至少把检索到的原始内容返回,
        # 别让调用方拿到一个空答案 —— 空答案比"不完整但有依据"更难排查。
        final = snapshot.values.get("retrieval") or ""
        if final:
            final = "[未经过写作工人整理, 以下是检索原始内容]\n" + final
    if not final:
        final = "未能生成答案: 检索与写作均未产出内容, 请检查知识库是否为空或模型调用是否失败。"
    return final, trace


# ====== 命令行自检: python multi_agent.py "你的问题" ======
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agent import build_tools, summarize_history
    from sentence_transformers import SentenceTransformer
    import chromadb
    from openai import OpenAI

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    embedder = SentenceTransformer(os.path.join(APP_DIR, "models/bge-small-zh-v1.5"))
    client = chromadb.PersistentClient(path=os.path.join(APP_DIR, "chroma_db"))
    coll = client.get_or_create_collection("personal_kb")

    api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not api_key or not api_key.isascii():
        # 常见坑: .env 里存的是"你的key"这类中文占位符 -> HTTP 请求头构建时抛
        # UnicodeEncodeError('ascii' codec...), 极难排查。这里提前用人话拦下。
        raise SystemExit(
            "❌ ZHIPU_API_KEY 无效: 缺失, 或 .env 里存的还是占位符(含中文)。\n"
            "   请打开 D:\\ASUS\\Documents\\workbuddy\\personal-ai-kb\\.env,\n"
            "   把 ZHIPU_API_KEY=... 改成真实 key (纯英文数字)。\n"
            "   或在 PowerShell 临时导出: $env:ZHIPU_API_KEY='真实key'"
        )
    llm = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")

    tools = build_tools(embedder, lambda: coll)
    question = sys.argv[1] if len(sys.argv) > 1 else "这个项目是怎么做上下文压缩的?"
    answer, tr = run_multi_agent(question, llm, tools, summarize_fn=summarize_history)
    print("问题:", question)
    print("—— 多 Agent 协作轨迹 ——")
    for i, t in enumerate(tr, 1):
        print(f"  {i}. [{t['worker']}] {t['log']}")
    print("—— 最终答案 ——")
    print(answer)
