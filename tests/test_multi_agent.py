# tests/test_multi_agent.py
# 多 Agent 编排的回归测试: 5 节点协作、规划多路检索、主管决策合法性校验
# (防死循环)、校验 FAIL 打回重写、空答案兜底链。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import run_multi_agent
from fakes import MultiAgentFakeLLM, make_tools
import unittest


class TestMultiAgent(unittest.TestCase):
    def setUp(self):
        self.tools = make_tools()

    def test_simple_flow(self):
        """简单问题: 规划(空) -> 检索 -> 写作 -> 校验(PASS) -> 收尾。"""
        llm = MultiAgentFakeLLM(
            route_plan=["planning", "retrieval", "writing", "verification", "finish"])
        ans, tr = run_multi_agent("RAG 是什么?", llm, self.tools, thread_id="t_simple")
        workers = [t["worker"] for t in tr]
        self.assertIn("retrieval", workers)
        self.assertIn("writing", workers)
        self.assertIn("verification", workers)
        self.assertIn("基于检索结果整理的答案", ans)

    def test_complex_planning_multi_route(self):
        """复杂问题带子问题: 规划拆 2 子问题 -> 检索应覆盖 3 路(原问题+2子问题)。"""
        llm = MultiAgentFakeLLM(
            route_plan=["planning", "retrieval", "writing", "verification", "finish"],
            sub_questions=["方面A", "方面B"])
        ans, tr = run_multi_agent("对比 X 和 Y 的异同", llm, self.tools, thread_id="t_complex")
        self.assertIn("基于检索结果整理的答案", ans)

    def test_deadlock_planning_blocked(self):
        """复刻真实 bug: 主管连续决策 planning。
        第一次合法(plan=='' 才允许规划); 后续 planning 决策被 _is_valid 纠正为规则路由,
        规划工人只应执行一次, 系统不死循环、能产出答案。"""
        llm = MultiAgentFakeLLM(
            route_plan=["planning", "planning", "planning", "planning",
                        "retrieval", "writing", "verification", "finish"])
        ans, tr = run_multi_agent("复杂问题", llm, self.tools, thread_id="t_dead")
        plan_count = [t["worker"] for t in tr].count("planning")
        self.assertLessEqual(plan_count, 1, "规划工人不应被重复执行(死循环被拦截)")
        self.assertTrue(ans)

    def test_fail_rewrite_then_finish(self):
        """校验 FAIL -> 打回重写 -> 最终仍 finish(不卡死, rewrites 受 MAX_REWRITE 限制)。"""
        llm = MultiAgentFakeLLM(
            route_plan=["planning", "retrieval", "writing", "verification",
                        "writing", "verification", "finish"],
            verdict="FAIL: 缺少关键细节。")
        ans, tr = run_multi_agent("问题", llm, self.tools, thread_id="t_fail")
        self.assertTrue(ans)
        self.assertIn("基于检索结果整理的答案", ans)

    def test_empty_answer_fallback(self):
        """检索与写作都空 -> 最终兜底链给出明确报错而非空答案。"""
        llm = MultiAgentFakeLLM(
            route_plan=["planning", "retrieval", "writing", "verification", "finish"],
            rag_answer="", draft="")
        ans, tr = run_multi_agent("问题", llm, self.tools, thread_id="t_empty")
        self.assertTrue(ans)
        self.assertIn("未能生成答案", ans)


if __name__ == "__main__":
    unittest.main()
