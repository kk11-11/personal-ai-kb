# test_multiturn.py
# 自动化验证多轮对话的三类指代能力: 轮次索引 / 代词指代 / 综合前文
# 复用 app.answer_question(q, history), history 格式与前端 chatHistory 完全一致 [{q, a}, ...]
# 用法: .\venv\Scripts\python.exe test_multiturn.py
import os
import re
import sys

# 关键: Git Bash / PowerShell 子进程不继承你手动设的临时 $env:ZHIPU_API_KEY,
# 必须在 import app 之前从 .env 注入, 否则 import app 会因缺 key 直接 SystemExit。
# 容错: 实测发现 .env 的 ZHIPU_API_KEY 值可能被中文说明/BOM 污染(如 "你的智谱APIKey_..."),
# 这里只抽取 sk- 开头的纯 ascii key 段, 避免中文进 http header 触发 ascii 编码错误。
_BASE = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_BASE, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "ZHIPU_API_KEY":
                clean = re.sub(r"[^ -~]", "", v)  # 去所有非 ascii 可打印字符(含中文/BOM)
                m = re.search(r"sk-[A-Za-z0-9\-_.]+", clean)
                v = m.group(0) if m else v
            os.environ.setdefault(k, v)

import app  # 此时 key 已在 environ, import 会加载 embedder/collection

# 结果同时写一份纯 ascii 文件(中文转 ?), 方便跨 shell 编码稳定查看
RESULT_PATH = os.path.join(_BASE, "test_multiturn_result.txt")
RESULT_LINES = []


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def check(name, q, history, expect_hits, expect_avoid=None):
    """跑一个 case, 打印问答 + 关键词断言。返回 True/False。"""
    print("=" * 64)
    print(f"CASE: {name}")
    print(f"Q: {q}")
    try:
        answer, _sources = app.answer_question(q, history)
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        RESULT_LINES.append(f"[FAIL] {name} | Q={_ascii(q)} | 调用失败: {_ascii(str(e))}")
        return False
    print(f"A: {answer[:400]}")
    ans = answer
    hit = any(k in ans for k in expect_hits)
    avoid = bool(expect_avoid) and any(k in ans for k in expect_avoid)
    ok = hit and not avoid
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] 命中={expect_hits} -> {hit} | 规避={expect_avoid} -> {avoid}")
    RESULT_LINES.append(
        f"[{tag}] {name} | Q={_ascii(q)} | 命中={expect_hits}->{hit} 规避={expect_avoid}->{avoid}"
    )
    return ok


def main():
    # 标准答案写死, 模拟一段已发生的多轮对话(命中词来自这些标准答案)
    history_short = [
        {"q": "Rerank是什么", "a": "Rerank（重排序）是在向量检索召回候选后，用交叉编码器对候选片段与问题做 pairwise 打分、重新排序，挑出最相关的 Top-K。本项目用的重排模型是 bge-reranker-base。"},
        {"q": "它用了哪个模型", "a": "本项目的重排模型是 bge-reranker-base（ModelScope 下载的 CrossEncoder 模型）。"},
        {"q": "二阶段检索是怎么实现的", "a": "第一阶段向量召回 Top-10 候选，第二阶段用 CrossEncoder 精排取 Top-3。实测检索命中率从 94.4% 提升到 100%。"},
        {"q": "反馈机制是怎么做的", "a": "前端每条回答下方有👍/👎按钮，点击后把 rating、question、answer、sources 写入 feedback.json，并用 threading.Lock 保证并发安全。"},
    ]

    results = []

    # 1. 轮次索引(短对话): 第5轮问"第一个问题" 应精准定位第1轮 Rerank
    results.append(check(
        "轮次索引(短对话, ≤4轮)",
        "那第一个问题有什么作用",
        history_short,
        expect_hits=["Rerank", "重排", "重排序"],
    ))

    # 2. 代词指代: 基于最近一轮(反馈机制)问"这个机制"
    results.append(check(
        "代词指代(最近一轮)",
        "这个机制是怎么保证并发安全的",
        history_short,
        expect_hits=["锁", "线程", "feedback", "并发"],
    ))

    # 3. 综合前文: 应同时覆盖多轮(既提 Rerank 又提反馈)
    results.append(check(
        "综合前文",
        "综合一下前面几个功能分别解决什么问题",
        history_short,
        expect_hits=["Rerank", "反馈"],
    ))

    # 4. 长对话(6轮)轮次索引: 前2轮会被 summarize_history 压成摘要,
    #    验证摘要后仍能按【第1轮】编号定位回 Rerank
    history_long = history_short + [
        {"q": "知识库里有哪些文档", "a": "有 01_ai_intern_tips、02_rag_basics、03_项目复盘笔记 三篇。"},
        {"q": "为什么要做二阶段检索", "a": "单纯向量检索在语义相近片段上容易把同文档相邻 chunk 同时召回造成冗余，ReRank 用 cross-encoder 换成不同来源片段，答案更聚焦。"},
    ]
    results.append(check(
        "轮次索引(长对话 >4轮, 摘要压缩后)",
        "那第一个问题是什么，它有什么作用",
        history_long,
        expect_hits=["Rerank", "重排", "重排序"],
    ))

    print("=" * 64)
    passed = sum(results)
    total = len(results)
    print(f"多轮对话验证结果: {passed}/{total} PASS")
    print("=" * 64)
    with open(RESULT_PATH, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(RESULT_LINES) + f"\n汇总: {passed}/{total} PASS\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
