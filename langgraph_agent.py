# langgraph_agent.py
# 用 LangGraph 重写 agent.py 里的手写 ReAct 循环 —— 接口保持完全一致,
# 方便两边对照跑: run_react(手写) vs run_react_langgraph(框架)。
#
# 设计取舍: 工具层(业务代码)一个字没改, 只把"编排层"(循环/分支/终止)交给 LangGraph。
# 这样能看清框架到底替你做了什么 —— 它替换的是控制流, 不是业务逻辑。
#
# 与手写版 run_react() 的对照关系:
#   手写 for step in range(max_steps)   ->  图的循环边 tools -> model
#   手写 if tool_calls: ... else: break ->  条件边 should_continue
#   手写 messages.append(...)           ->  State + add_messages reducer(自动合并)
#   手写 max_steps 计数                 ->  state["step"] + recursion_limit 双保险
#   手写 trace 列表                     ->  stream(stream_mode="updates") 逐步收集
#   手写"达到上限强制收尾"              ->  force_answer 节点

import json

from langchain_core.messages import convert_to_openai_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

HISTORY_RECENT = 6
DEFAULT_MODEL = "glm-4-flash"


# ====== State 定义 ======
# add_messages 是这个框架最核心的抽象: 它规定了"新消息如何并入旧消息"。
# 手写版里你要自己 messages.append(), 还要注意别重复追加; 这里交给 reducer。
# 好处是后续接 checkpoint 时, 框架能自动做增量持久化和消息去重。
def _state_schema():
    from typing import Annotated, Sequence, TypedDict

    class AgentState(TypedDict):
        messages: Annotated[Sequence, add_messages]
        step: int

    return AgentState


def _exec_tool(tools, name, args):
    """与手写版一致: 工具异常转成字符串喂回模型, 让它自己纠错, 而不是整条链路崩掉。"""
    tool = tools.get(name)
    if tool is None:
        return f"错误: 没有名为 '{name}' 的工具。可用工具: {', '.join(tools.keys())}"
    try:
        return str(tool["fn"](args or {}))
    except Exception as e:
        return f"工具 '{name}' 执行出错: {e}"


def build_react_graph(llm, tools, model=DEFAULT_MODEL, max_steps=4):
    """把 ReAct 循环编译成一张 LangGraph 图。

    tools 直接复用 agent.py 的 build_tools() 产物, 工具函数不用改。
    """
    schemas = [t["schema"] for t in tools.values()]
    AgentState = _state_schema()

    def call_model(state):
        """节点一: 调模型。对应手写版循环体里 llm.chat.completions.create 那一段。"""
        openai_msgs = convert_to_openai_messages(state["messages"])
        resp = llm.chat.completions.create(
            model=model, messages=openai_msgs, tools=schemas, tool_choice="auto"
        )
        msg = resp.choices[0].message
        out = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return {"messages": [out], "step": state.get("step", 0) + 1}

    def call_tools(state):
        """节点二: 执行工具。对应手写版 for tc in tool_calls 那段。"""
        last = state["messages"][-1]
        outs = []
        for tc in getattr(last, "tool_calls", None) or []:
            try:
                args = json.loads(tc["args"]) if isinstance(tc.get("args"), str) else (tc.get("args") or {})
            except Exception:
                args = {}
            result = _exec_tool(tools, tc["name"], args)
            outs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        return {"messages": outs}

    def force_answer(state):
        """节点三: 达到轮数上限时强制收尾, 防止无限转圈烧 token(手写版同款兜底)。"""
        openai_msgs = convert_to_openai_messages(state["messages"])
        openai_msgs.append(
            {"role": "user", "content": "已达到最大工具调用轮数, 请基于上面已有的信息直接给出最终答案。"}
        )
        resp = llm.chat.completions.create(model=model, messages=openai_msgs)
        return {"messages": [{"role": "assistant", "content": resp.choices[0].message.content or ""}]}

    def should_continue(state):
        """条件边: 手写版里那句 if tool_calls: continue else: break, 在这里被显式画成了图的一条边。"""
        last = state["messages"][-1]
        has_tools = bool(getattr(last, "tool_calls", None))
        if not has_tools:
            return END
        if state.get("step", 0) >= max_steps:
            return "force"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("force", force_answer)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", "force": "force", END: END})
    graph.add_edge("tools", "model")   # 这一行就是手写版的 continue
    graph.add_edge("force", END)

    # checkpointer 是手写版完全没有的能力: 对话状态可持久化、可回溯到任意历史步骤、
    # 可中断让人介入(human-in-the-loop)。先用内存版, 后续换 SqliteSaver 即可落盘。
    return graph.compile(checkpointer=MemorySaver())


# ====== 图缓存 ======
# 坑点: 编译图有开销, 更重要的是 —— 如果每次问答都新建 MemorySaver(),
# checkpoint 每次都是空的, "跨请求记忆"就成了摆设。必须让同一个
# (model, max_steps, 工具集) 复用同一张编译好的图, checkpoint 才有意义。
_GRAPH_CACHE = {}


def get_react_graph(llm, tools, model=DEFAULT_MODEL, max_steps=4):
    # key 必须带上 id(llm): 图的节点是闭包, 会捕获传入的 llm 实例。
    # 若漏掉这一项, 换了一个 llm(换 key / 换模型 / 测试里换 mock)仍会命中旧图, 拿到旧 client。
    key = (id(llm), model, max_steps, tuple(sorted(tools.keys())))
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = build_react_graph(llm, tools, model=model, max_steps=max_steps)
    return _GRAPH_CACHE[key]


def run_react_langgraph(question, llm, tools, model=DEFAULT_MODEL, max_steps=4,
                        history=None, thread_id="default", summarize_fn=None):
    """与 agent.run_react() 同签名, 返回 (final_answer, trace)。

    summarize_fn: 传入 agent.summarize_history 即可复用你已有的上下文压缩逻辑。
    thread_id: 同一 thread_id 的对话会共享 checkpoint, 实现跨请求记忆。
    """
    graph = get_react_graph(llm, tools, model=model, max_steps=max_steps)

    system_msg = {
        "role": "system",
        "content": (
            "你是用户的个人知识库助手, 可以使用工具。\n"
            "按 ReAct 方式工作: 先想清楚需要什么信息(Thought), 再决定调用工具(Action)还是直接回答。\n"
            f"规则:\n"
            f"1. 不知道有哪些文档时先调用 list_sources; 需要在某篇文档里查内容时调用 search_by_source。\n"
            f"2. 一次结果不够可以继续调用其他工具, 但总共最多 {max_steps} 轮。\n"
            f"3. 信息足够时直接给最终答案, 不要再调用工具。\n"
            f"4. 答案必须基于工具返回的内容, 工具没返回的东西不许编造; 确实没有就说没找到。\n"
            f"5. 若看到以 [早期对话摘要] 开头的内容, 那是对更早对话的压缩, 结合它来理解当前问题。\n"
        ),
    }
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps * 3 + 5}

    # 关键设计(踩过的坑): 对话历史只能由一方管理, 不能"自己传 history"和
    # "交给 checkpoint 累积"同时做 —— 两边都管会让每轮消息重复翻倍。
    # 策略:
    #   - 该 thread 首次调用(checkpoint 为空): 用传入的 history 初始化, 兼容 app.py 现有机制
    #   - 后续调用: checkpoint 里已有完整历史, 只传增量(新问题), 由框架负责累积
    try:
        snap = graph.get_state(config)
        has_state = bool(snap and snap.values.get("messages"))
    except Exception:
        has_state = False

    if has_state:
        messages = [{"role": "user", "content": question}]
    else:
        messages = [system_msg]
        if history:
            # 上下文压缩逻辑与手写版共用, 不重复实现
            if summarize_fn:
                summary, recent = summarize_fn(history, llm, model=model)
            else:
                summary, recent = None, history[-HISTORY_RECENT:]
            if summary:
                messages.append({"role": "user", "content": f"[早期对话摘要]\n{summary}"})
            for h in recent:
                messages.append({"role": "user", "content": h.get("q", "")})
                messages.append({"role": "assistant", "content": h.get("a", "")})
        messages.append({"role": "user", "content": question})
    trace = []
    final = None

    for event in graph.stream({"messages": messages, "step": 0}, config, stream_mode="updates"):
        for node_name, payload in event.items():
            new_msgs = payload.get("messages") or []
            for m in new_msgs:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role == "assistant":
                    calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
                    if calls:
                        for c in calls:
                            trace.append({"node": node_name, "tool": c["function"]["name"],
                                          "args": c["function"].get("arguments")})
                    else:
                        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                        if content:
                            final = content
                            trace.append({"node": node_name, "type": "final"})
                elif role == "tool":
                    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    trace.append({"node": node_name, "type": "observation", "result": str(content)[:400]})

    if final is None:
        # 兜底: 从 checkpoint 里取最后一条 assistant 消息
        snapshot = graph.get_state(config)
        for m in reversed(snapshot.values.get("messages", [])):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "assistant":
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if content:
                    final = content
                    break

    return final, trace


# ====== 命令行自检: python langgraph_agent.py "你的问题" ======
if __name__ == "__main__":
    import os
    import sys

    # 兜底: 从 .env 读取 ZHIPU_API_KEY(若已在环境变量里设过则不受影响)。
    # 和 app.py 行为保持一致, 但更友好 —— 不用每次手动 export。
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agent import build_tools, summarize_history

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    from sentence_transformers import SentenceTransformer
    import chromadb
    from openai import OpenAI

    embedder = SentenceTransformer(os.path.join(APP_DIR, "models/bge-small-zh-v1.5"))
    client = chromadb.PersistentClient(path=os.path.join(APP_DIR, "chroma_db"))
    coll = client.get_or_create_collection("personal_kb")

    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise SystemExit("请设置 ZHIPU_API_KEY")
    llm = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")

    tools = build_tools(embedder, lambda: coll)
    question = sys.argv[1] if len(sys.argv) > 1 else "知识库里有哪些文档?"
    answer, tr = run_react_langgraph(question, llm, tools, summarize_fn=summarize_history)
    print("问题:", question)
    print("答案:", answer)
    print("轨迹:")
    for t in tr:
        print("  ", t)
