# tests/fakes.py
# 离线假 LLM 构件 —— 不依赖任何 API key, 用于把"LangGraph 单 Agent / 多 Agent 编排"的
# 控制流逻辑固化成可重复跑的回归测试(也是 W3 自进化闭环的基础设施)。
#
# 设计原则:
#   1. 完全模仿 OpenAI SDK 的 `client.chat.completions.create(...)` 接口签名与返回结构
#      (choices[0].message.content / .tool_calls), 这样被测代码无需改动即可接入。
#   2. 用"确定性响应"而非随机, 保证测试可断言、可回归。
import json
import uuid
from types import SimpleNamespace


class FakeToolCall:
    """模仿 OpenAI SDK 返回的 tool_call 对象(用属性访问, 与真实 SDK 一致)。"""

    def __init__(self, name, args="{}", cid=None):
        self.id = cid or ("call_" + uuid.uuid4().hex[:8])
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=args)


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or None


class FakeCompletion:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


class _Completions:
    def __init__(self, llm):
        self._llm = llm

    def create(self, model=None, messages=None, tools=None, tool_choice=None, **kw):
        return self._llm._respond(messages, tools)


class _ChatNamespace:
    """模仿 OpenAI SDK 的 client.chat 命名空间(含 .completions.create 两层)。"""

    def __init__(self, llm):
        self.completions = _Completions(llm)


# ---------------------------------------------------------------------------
# 单 Agent 假 LLM: 按"调用顺序"弹出预设响应。
# 每个响应是 FakeMessage; 带 tool_calls 表示要调工具, 否则直接给最终答案。
# ---------------------------------------------------------------------------
class SingleAgentFakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.chat = _ChatNamespace(self)
        self.calls = 0

    def _respond(self, messages, tools):
        self.calls += 1
        if self._responses:
            return FakeCompletion(self._responses.pop(0))
        return FakeCompletion(FakeMessage(content="(兜底)最终答案"))


# ---------------------------------------------------------------------------
# 多 Agent 假 LLM: 按"调用上下文"分派响应。
#   - 若 tools 含 route schema(主管决策) -> 弹出 route_plan 里的下一个决策
#   - 若带 tools 但非 route(检索子调用) -> 直接给 rag_answer(让 run_react_langgraph 一步完成)
#   - 规划工人(system 含"规划工人") -> 返回 sub_questions 的 JSON
#   - 写作 / 校验工人 -> 返回 draft / verdict
# ---------------------------------------------------------------------------
class MultiAgentFakeLLM:
    def __init__(self, route_plan=None, sub_questions=None,
                 rag_answer="[检索到的知识库内容]",
                 draft="基于检索结果整理的答案。",
                 verdict="PASS: 草稿忠于检索结果。"):
        self._route_plan = list(route_plan or [])
        self._sub = list(sub_questions or [])
        self._rag = rag_answer
        self._draft = draft
        self._verdict = verdict
        self.chat = _ChatNamespace(self)
        self.calls = 0

    @staticmethod
    def _is_route(tools):
        if not tools:
            return False
        return "route" in json.dumps(tools, ensure_ascii=False)

    def _respond(self, messages, tools):
        self.calls += 1
        sys_text = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                sys_text = m.get("content", "")
            elif hasattr(m, "role") and m.role == "system":
                sys_text = getattr(m, "content", "")
        if self._is_route(tools):
            dec = self._route_plan.pop(0) if self._route_plan else "finish"
            if dec is None:
                return FakeCompletion(FakeMessage(content="我无法决定下一步"))
            return FakeCompletion(FakeMessage(
                content="",
                tool_calls=[FakeToolCall("route", json.dumps({"next": dec, "reason": "auto"}))]))
        if tools:  # 检索子调用: 直接给答案, 让单 Agent 循环一步到位
            return FakeCompletion(FakeMessage(content=self._rag))
        if "规划工人" in sys_text:
            return FakeCompletion(FakeMessage(
                content=json.dumps({"sub_questions": self._sub}, ensure_ascii=False)))
        if "写作工人" in sys_text:
            return FakeCompletion(FakeMessage(content=self._draft))
        if "校验工人" in sys_text:
            return FakeCompletion(FakeMessage(content=self._verdict))
        return FakeCompletion(FakeMessage(content="(默认)回复"))


def make_tools():
    """构造最小 tools dict, 兼容 agent.build_tools 的产物格式
    (每个 value 含 "schema"(OpenAI 工具描述) 与 "fn"(可执行函数))。"""
    return {
        "list_sources": {
            "schema": {
                "type": "function",
                "function": {
                    "name": "list_sources", "description": "列出所有知识库文档名",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            "fn": lambda args: "文档: 01_intro.md, 02_rag.md",
        },
        "search_by_source": {
            "schema": {
                "type": "function",
                "function": {
                    "name": "search_by_source", "description": "在某文档内检索",
                    "parameters": {"type": "object",
                                   "properties": {"source": {"type": "string"}}},
                },
            },
            "fn": lambda args: "相关片段内容。",
        },
    }
