# tests/test_langgraph_agent.py
# LangGraph 单 Agent 编排的回归测试: 验证"循环 / 工具调用 / 终止 / 超步数强制收尾"
# 这些原本靠手写循环保证, 现在由 StateGraph 的条件边与 force 节点保证。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_agent import run_react_langgraph
from fakes import SingleAgentFakeLLM, FakeMessage, FakeToolCall, make_tools
import unittest


class TestLangGraphAgent(unittest.TestCase):
    def setUp(self):
        self.tools = make_tools()

    def test_direct_answer(self):
        """模型不调工具, 直接给答案 -> 图应一步终止。"""
        llm = SingleAgentFakeLLM([FakeMessage(content="RAG 是检索增强生成。")])
        ans, trace = run_react_langgraph("RAG 是什么?", llm, self.tools, max_steps=4)
        self.assertIn("RAG 是检索增强生成", ans)
        self.assertTrue(any("model" in t["node"] for t in trace))

    def test_one_tool_then_answer(self):
        """先调一次工具, 再基于工具结果作答 -> 验证 tools 边与 call_tools 执行。"""
        llm = SingleAgentFakeLLM([
            FakeMessage(content="", tool_calls=[FakeToolCall("list_sources", "{}")]),
            FakeMessage(content="根据资料, RAG 是检索增强生成。"),
        ])
        ans, trace = run_react_langgraph("有哪些文档?", llm, self.tools, max_steps=4)
        self.assertIn("检索增强生成", ans)
        nodes = [t["node"] for t in trace]
        self.assertIn("tools", nodes)

    def test_force_answer_on_max_steps(self):
        """模型永远调工具 -> 达到 max_steps 后 force_answer 必须强制作出非空答案, 不死循环。"""
        llm = SingleAgentFakeLLM(
            [FakeMessage(content="", tool_calls=[FakeToolCall("list_sources", "{}")])] * 2
            + [FakeMessage(content="强制作答结果")]
        )
        ans, trace = run_react_langgraph("?", llm, self.tools, max_steps=2)
        self.assertTrue(ans, "超步数后必须给出兜底答案, 不能为空")
        self.assertIn("强制作答结果", ans)


if __name__ == "__main__":
    unittest.main()
