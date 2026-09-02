# 第 1 周：用 LangGraph 重写手写 ReAct 循环

> 目标（对应视频"先把框架都了解"）：不是会用框架就行，而是**知道框架在你已有的手写代码上替换了哪一层**。
> 只有这样，面试里才能讲清"手写 vs 框架"的取舍——这才是差异化点，而不是"我也会调 LangChain"。

## 一句话结论

LangGraph 替换的**只有编排层**（循环、分支、终止、状态累积），**一行业务逻辑都没动**：
`agent.build_tools()` 原样复用，`summarize_history()` 原样复用，工具函数一个字没改。

手写版 130+ 行的 `for step in range(max_steps)` 控制流，被压缩成一张 3 节点的图。

## 手写 ReAct ↔ LangGraph 概念映射

| 手写 `agent.py` | LangGraph 对应 | 说明 |
|---|---|---|
| `for step in range(max_steps):` | 图的**循环边** `tools → model` | 框架负责"转回去" |
| `if tool_calls: 执行工具 else: break` | **条件边** `should_continue` | 显式画成一条边，可观测 |
| `messages.append(...)` | `State` + `add_messages` reducer | 框架自动合并、去重 |
| `max_steps` 计数器 | `state["step"]` + `recursion_limit` | 双保险防无限循环 |
| `trace.append(...)` | `graph.stream(stream_mode="updates")` | 逐步收集，天然可观测 |
| 达到上限"强制收尾" | `force_answer` 节点 | 同款兜底，防烧 token |
| 无 | `checkpointer=MemorySaver()` | **手写版完全没有**：状态可持久化、可回溯、可 human-in-the-loop |

## 三个最值得记住的 LangGraph 概念

1. **State + reducer（`add_messages`）**
   手写版你要自己管 `messages.append`，还得小心别重复追加。LangGraph 用 `TypedDict` 定义状态，
   用 `add_messages` 规定"新消息如何并入旧消息"——后续接 checkpoint 时，框架自动做增量持久化与去重。

2. **节点（node）就是函数，边（edge）就是控制流**
   `model → (tools | force | END)` 把"什么时候继续调工具、什么时候收尾"画成了图，而不是藏在一个 `for` 里。
   代价：调试时你要理解"图在跑"，而不是"我的函数在跑"。

3. **Checkpointer = 跨请求记忆**
   手写版每个请求都是无状态的，历史靠 `app.py` 自己传 `history`。框架用 `thread_id` 把对话状态存进
   checkpointer，同一 `thread_id` 自动记住上一轮——这是手写版没有的能力。

## 踩过的 3 个真坑（都在代码注释里）

- **坑 1：每次问答都重建图 + 新建 MemorySaver → checkpoint 永远空。**
  `get_react_graph()` 用缓存复用编译好的图，checkpointer 才有意义。
- **坑 2：history 和 checkpoint 同时管历史 → 每轮消息重复翻倍。**
  定死策略：该 thread 首次调用用 `history` 初始化；后续只传增量新问题，由框架累积。
- **坑 3：缓存 key 漏了 `id(llm)` → 换 mock/client 仍命中旧图。**
  key 必须带 `id(llm)`，因为图的节点是闭包，会捕获传入的 llm 实例。

## 框架给了你、但手写版没有的东西（面试可聊）

- **持久化 & 时间旅行**：`graph.get_state(config)` 能回到任意历史步骤重跑。
- **Human-in-the-loop**：中断执行让人介入（审批、改参数）再继续——生产 Agent 的刚需。
- **可观测**：`stream_mode="updates"` 天然输出每一步轨迹，不用自己造 trace。

> 下一步想体验这些，把 `MemorySaver` 换成 `SqliteSaver` 就能落盘；加一个 `interrupt_before=["tools"]` 就能做人工审批。

## 怎么验证

```bash
# 逻辑测试(5 个场景, 无需 API Key): 正常调用 / 坏工具容错 / 死循环强制收尾 / checkpoint 记忆 / 跨调用累积
python ../../worrkbuddy01/.workbuddy/tmp/test_langgraph_mock.py

# 真实跑(需 ZHIPU_API_KEY):
python langgraph_agent.py "知识库里有哪些文档?"
```

## 交付物清单（Week 1）

- [x] `langgraph_agent.py` —— 接口与 `agent.run_react()` 完全一致的框架版
- [x] `test_langgraph_mock.py` —— 5 场景验证（已全过）
- [x] `requirements.txt` 增加 `langgraph` / `langchain-core`
- [x] 本文件 —— 手写 vs 框架的对照与面试话术
