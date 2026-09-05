# tests/run_all.py
# 一键跑全部回归测试(无需 API key, 纯离线 mock)。
# 用法: 在项目根目录执行  venv\Scripts\python.exe tests\run_all.py
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_langgraph_agent import TestLangGraphAgent
from test_multi_agent import TestMultiAgent

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLangGraphAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiAgent))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
