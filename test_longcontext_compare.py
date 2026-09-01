# test_longcontext_compare.py
# 长对话记忆方案对比: 摘要压缩(summary) vs 向量化历史召回(vector)。
#
# 核心指标: 两种方案是否把"用户问的目标轮次"送进 prompt(信息送入率)。
#   - 这比直接比 LLM 回答更根本: 信息没送进去, 模型再强也答不出。
#   - 策略 B(向量召回)用本地 embedder 真跑, 不需联网 key; 策略 A 按 keep_recent=6 窗口做确定性分析。
#   - 加 --live 参数(需有效 ZHIPU_API_KEY)可额外真跑两种模式的 LLM 回答, 看最终提取效果。
#
# 两类问题设计(故意制造差异):
#   位置指代: "第一个问题/第3轮/最前面那个" -> 摘要方案靠【第N轮】编号应更准
#   语义回溯: "之前讨论过的重排序模型叫什么" -> 向量召回靠语义应更准(召回相关原文)

import os
import re
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_BASE, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import app  # 加载 embedder / llm / answer_question
from history_retrieval import prepare_history_vector

# ---------- 构造 12 轮超长对话(每轮一个明确主题, 互相区分) ----------
HISTORY = [
    {"q": "Rerank 是什么", "a": "Rerank 是重排序, 在 RAG 里做二阶段检索: 先向量召回候选, 再用交叉编码器精排取 Top-3, 提升命中率。"},
    {"q": "ReRank 用了哪个模型", "a": "用的重排序模型是 bge-reranker-base, 从 ModelScope 下载到本地 models 目录, 缺失时自动降级为纯向量召回。"},
    {"q": "反馈机制怎么做的", "a": "页面每个回答下有 👍/👎 按钮, 点击后把 rating/question/answer/sources 写入 feedback.json, 用 threading.Lock 保证并发安全。"},
    {"q": "手写 ReAct 怎么实现", "a": "ReAct 循环每一步是 Thought->Action->Observation: 模型先想(Thought), 再决定调工具(Action), 工具返回结果(Observation), 最多 max_steps 轮。"},
    {"q": "trace 面板展示什么", "a": "Agent 模式的 trace 面板可折叠, 展示每一步调用的工具名、传入参数、以及返回结果前 400 字, 用来排查链路。"},
    {"q": "Chroma 怎么存向量", "a": "文档切片后由 bge-small-zh 编码, 存入 Chroma 的 collection(名为 personal_kb), 每个片段带 source 和 chunk 元数据。"},
    {"q": "embedding 用哪个模型", "a": "向量化用的是 bge-small-zh-v1.5, 本地运行不联网, 切片大小约 300 字带 80 字重叠。"},
    {"q": "为什么用 RAG 而不是直接问大模型", "a": "因为直接问大模型会幻觉且看不到你的私有资料, RAG 强制模型只基于你上传的文档回答并标注出处。"},
    {"q": "Agent 模式怎么开", "a": "前端有个 Agent 模式开关(checkbox), 勾选后请求走 /ask_agent 端点, 触发工具调用和 trace 展示。"},
    {"q": "长对话怎么压缩", "a": "对话超过 keep_recent=6 轮时, 早期轮次用 LLM 压成一段摘要(summarize_history), 只保留最近 6 轮原文, 控制 token。"},
    {"q": "多轮对话怎么定位轮次", "a": "给历史每条加显式【第 N 轮】编号, 模型按编号精准定位『第一个问题』『第 3 轮』, 避免凭感觉挑一条。"},
    {"q": "部署要注意什么", "a": "Flask 默认 5000 端口, 用 PORT 环境变量可改; 启动前要确认端口没被旧进程占用, 否则需 taskkill 释放。"},
]

# target = 目标轮次(1-based); type 用于结论解读
CASES = [
    ("第一个问题讲了什么", 1, "重排序/二阶段", "位置指代"),
    ("第 3 轮说的反馈机制用什么文件存数据", 3, "feedback.json", "位置指代"),
    ("最前面那个问题提到的重排序模型叫什么", 2, "bge-reranker-base", "位置指代"),
    ("我们之前详细讨论过的重排序模型具体叫什么名字", 2, "bge-reranker-base", "语义回溯"),
    ("反馈功能是怎么收集用户评价的", 3, "feedback.json", "语义回溯"),
    ("手写 Agent 那个循环里每一步的专业名称是什么", 4, "Thought/Action/Observation", "语义回溯"),
]

KEEP_RECENT = 6  # 与 agent.HISTORY_RECENT 对齐


def analyze(q, target, embedder, top_k=6):
    # 策略 B: 向量召回(真跑)
    text_b = prepare_history_vector(HISTORY, q, embedder, top_k=top_k)
    keep = [int(m) for m in re.findall(r"【第 (\d+) 轮】", text_b)]
    hit_b = target in keep
    # 策略 A: 摘要窗口分析(keep_recent=6, 12轮 -> 前6轮压摘要, 后6轮原文)
    n = len(HISTORY)
    recent_start = n - KEEP_RECENT + 1
    if target >= recent_start:
        a_loc = f"recent原文(完整, 第{recent_start}-{n}轮)"
    else:
        a_loc = f"old摘要段(细节可能丢, 第1-{recent_start-1}轮)"
    return hit_b, keep, a_loc


def main():
    live = "--live" in sys.argv
    lines = []
    lines.append("=" * 78)
    lines.append("长对话记忆方案对比: 摘要压缩(summary) vs 向量化历史召回(vector)")
    lines.append(f"对话轮数={len(HISTORY)}  keep_recent={KEEP_RECENT}  top_k={6}")
    lines.append("=" * 78)
    lines.append(f"{'CASE':<6}{'TYPE':<10}{'TARGET':<7}{'A(摘要方案)':<28}{'B(向量召回)':<12}{'B召回轮次'}")
    lines.append("-" * 78)

    results = []
    for idx, (q, target, kw, typ) in enumerate(CASES, 1):
        hit_b, keep, a_loc = analyze(q, target, app.embedder)
        a_short = "原文完整" if "完整" in a_loc else "摘要(丢细节风险)"
        lines.append(
            f"#{idx:<5}{typ:<10}{'第'+str(target)+'轮':<7}{a_loc:<28}{('命中' if hit_b else '未命中'):<12}{str(keep)}"
        )
        results.append((q, target, kw, typ, hit_b, keep, a_loc))

    lines.append("-" * 78)
    # 统计
    hit_b_count = sum(1 for r in results if r[4])
    lines.append(f"向量召回(B) 信息送入命中: {hit_b_count}/{len(results)}")
    lines.append(f"摘要方案(A): 目标轮次落入 recent 原文的 = "
                 f"{sum(1 for r in results if 'recent' in r[6])}/{len(results)}; 落入 old 摘要段的 = "
                 f"{sum(1 for r in results if 'old' in r[6])}/{len(results)}")
    lines.append("=" * 78)

    if live:
        lines.append("")
        lines.append("[LIVE] 真跑两种模式的 LLM 回答(需有效 key):")
        lines.append("=" * 78)
        for idx, (q, target, kw, typ, hit_b, keep, a_loc) in enumerate(results, 1):
            try:
                ans_a, _ = app.answer_question(q, HISTORY, "summary")
            except Exception as e:
                ans_a = f"<summary 调用失败: {e}>"
            try:
                ans_b, _ = app.answer_question(q, HISTORY, "vector")
            except Exception as e:
                ans_b = f"<vector 调用失败: {e}>"
            lines.append(f"#{idx} [{typ}] Q: {q}")
            lines.append(f"  A摘要方案: {ans_a[:200]}")
            lines.append(f"  B向量召回: {ans_b[:200]}")
            lines.append("")

    # 结论
    lines.append("=" * 78)
    lines.append("结论解读:")
    lines.append("  - 位置指代类('第一个/第3轮/最前面'): 摘要方案靠【第N轮】编号可定位,")
    lines.append("    但目标若在 old 段(前6轮), 细节已被压成摘要, 答得出来但可能丢具体概念;")
    lines.append("    向量召回靠语义, 若目标轮内容恰好最相关才命中, 否则可能召不中(位置≠语义)。")
    lines.append("  - 语义回溯类('之前讨论过的X叫什么'): 向量召回按语义把相关原文拉回, 命中率高;")
    lines.append("    摘要方案目标若在 old 段, 同样面临细节丢失。")
    lines.append("  => 两者互补: 摘要擅长按编号定位早期轮次, 向量召回擅长按语义找回相关历史细节。")
    lines.append("  => 工程取舍: 真要无限长对话不丢实体, 可'摘要(控token) + 向量召回(补细节)'双路并进。")
    lines.append("=" * 78)

    out = "\n".join(lines)
    print(out)
    res_path = os.path.join(_BASE, "test_longcontext_result.txt")
    with open(res_path, "w", encoding="ascii", errors="replace") as f:
        f.write(out + "\n")
    # 同时保留原始中文版(utf-8)供用户阅读
    with open(os.path.join(_BASE, "test_longcontext_result.zh.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\n[结果已写入 test_longcontext_result.txt(ASCII) 与 test_longcontext_result.zh.txt(中文)]")


if __name__ == "__main__":
    main()
