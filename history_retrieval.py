# history_retrieval.py
# 长对话记忆的第二种方案: 向量化历史召回(Vectorized History Retrieval)。
#
# 与 summarize_history(摘要压缩) 的对比:
#   - 摘要方案: 把早期轮次压成一段摘要, 靠【第 N 轮】编号让模型按位置精准定位,
#               但摘要会丢失具体概念/细节(已踩坑: keep_recent=4 时压丢第 1 轮 Rerank 概念)。
#   - 向量召回方案: 对每轮历史做 embedding, 按当前问题的语义召回最相关的 top-k 轮原文,
#               保留原文细节, 但召回是"语义相似"而非"位置定位"——
#               问"第一个问题"时, 模型问的是位置, 向量召回可能召不中目标轮次(除非该轮内容恰好最相关)。
#   两者互补: 摘要擅长按编号定位早期轮次, 向量召回擅长按语义找回相关历史内容。
#
# 注意: embedder 是本地模型(bge-small-zh), 此模块不依赖联网 API, 可离线跑。

import numpy as np


def _cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def prepare_history_vector(history, q, embedder, top_k=6, _cache=None):
    """对每轮 (q+a) 编码, 按当前问题 q 的语义召回 top_k 轮, 带全局【第 N 轮】编号返回文本。

    返回拼好的 history 文本块(空串表示无历史)。
    _cache: 可选 {turn_text: vec} 缓存, 避免同一轮重复编码(多问题批量评测时省时间)。
    """
    if not history:
        return ""
    if _cache is None:
        _cache = {}
    turn_vecs = []
    for h in history:
        text = f"用户: {h.get('q', '')}\n助手: {h.get('a', '')}"
        if text in _cache:
            v = _cache[text]
        else:
            try:
                v = embedder.encode([text]).tolist()[0]
                _cache[text] = v
            except Exception:
                v = None
        turn_vecs.append(v)
    try:
        q_vec = embedder.encode([q]).tolist()[0]
    except Exception:
        q_vec = None
    if q_vec is None:
        # 编码失败则降级为最近 top_k 轮
        keep = list(range(max(1, len(history) - top_k + 1), len(history) + 1))
    else:
        scored = []
        for i, v in enumerate(turn_vecs, start=1):
            if v is None:
                continue
            scored.append((_cosine(q_vec, v), i))
        scored.sort(reverse=True)
        keep = [i for _, i in scored[:top_k]]
        keep.sort()  # 按原对话顺序呈现, 便于模型理解
    lines = [
        f"【第 {i} 轮】\n用户: {history[i - 1].get('q', '')}\n助手: {history[i - 1].get('a', '')}"
        for i in keep
    ]
    return "\n\n".join(lines)
