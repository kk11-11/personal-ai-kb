# eval.py
# 个人 AI 知识库 —— 检索效果与性能自评估
# 设计目标：无需 LLM API key 也能运行（只评估向量检索部分），
#           如需端到端延迟则设置 ZHIPU_API_KEY 后会自动追加测试。
import os
import glob
import re
import time
from sentence_transformers import SentenceTransformer
import chromadb

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models/bge-small-zh-v1.5")
DB_DIR = os.path.join(APP_DIR, "chroma_db")
DOCS_DIR = os.path.join(APP_DIR, "docs")
COLLECTION = "personal_kb"
TOP_K = 3


def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)
    if path.lower().endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs if p.text.strip())
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text, size=500, overlap=80):
    text = text.strip().replace("\r\n", "\n")
    if not text:
        return []
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        out.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return out


def main():
    print("🔧 加载本地向量模型 ...")
    embedder = SentenceTransformer(MODEL_DIR)
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_or_create_collection(COLLECTION)
    n_chunks = coll.count()
    print(f"📦 知识库片段总数: {n_chunks}")
    if n_chunks == 0:
        print("⚠️ 知识库为空，请先 `python ingest.py` 建库后再评估。")
        return

    # 文档清单
    doc_files = []
    for ext in ("*.md", "*.txt", "*.pdf", "*.docx"):
        doc_files.extend(sorted(glob.glob(os.path.join(DOCS_DIR, ext))))
    print(f"📚 文档数: {len(doc_files)}")

    # 构造自测查询：从每篇文档抽取代表句作为 query，预期 source = 该文档。
    # 这是“检索召回率”的代理指标：衡量给定一个与资料相关的问题，
    # 检索器能否把正确文档的相关片段召回进 Top-K。
    queries = []
    for p in doc_files:
        text = read_text(p)
        sents = re.split(r"[。！？!?\n]", text)
        sents = [s.strip() for s in sents if 12 <= len(s.strip()) <= 70]
        for s in sents[:6]:
            queries.append((s, os.path.basename(p)))
    print(f"❓ 自测问题数: {len(queries)}")

    hit1 = hit3 = 0
    latencies = []
    examples = []
    for q, src in queries:
        t0 = time.time()
        q_emb = embedder.encode([q]).tolist()
        res = coll.query(query_embeddings=q_emb, n_results=TOP_K)
        latencies.append((time.time() - t0) * 1000)
        sources = [m["source"] for m in res["metadatas"][0]]
        if src in sources[:1]:
            hit1 += 1
        if src in sources[:3]:
            hit3 += 1
        if len(examples) < 3:
            examples.append((q, sources))

    n = len(queries)
    avg_lat = sum(latencies) / len(latencies)
    print(f"✅ Top-1 命中率: {hit1 / n * 100:.1f}%  ({hit1}/{n})")
    print(f"✅ Top-3 命中率: {hit3 / n * 100:.1f}%  ({hit3}/{n})")
    print(f"⏱️  平均检索延迟: {avg_lat:.1f} ms")
    print("🔍 示例:")
    for q, s in examples:
        print(f"   Q: {q[:28]}  →  召回来源: {s}")

    # 可选：端到端延迟（需智谱 API key）
    key = os.environ.get("ZHIPU_API_KEY")
    if key:
        from openai import OpenAI
        llm = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4/")
        q, _ = queries[0]
        t0 = time.time()
        q_emb = embedder.encode([q]).tolist()
        res = coll.query(query_embeddings=q_emb, n_results=TOP_K)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        context = "\n\n".join(
            f"[{m['source']}#{m['chunk']}]\n{d}" for d, m in zip(docs, metas)
        )
        prompt = (
            "你是用户的个人知识库助手。请严格根据下面提供的【资料】回答问题,"
            "如果资料里没有答案,请直说'资料里没找到相关信息',不要编造。\n\n"
            f"【资料】\n{context}\n\n【问题】{q}\n\n【回答】"
        )
        resp = llm.chat.completions.create(
            model="glm-4-flash", messages=[{"role": "user", "content": prompt}]
        )
        e2e = (time.time() - t0) * 1000
        print(f"⏱️  端到端延迟(含 LLM 生成): {e2e:.1f} ms")
        print(f"💬 示例回答: {resp.choices[0].message.content[:140]}")
    else:
        print("(未设置 ZHIPU_API_KEY，跳过端到端延迟测试；检索评估已完成)")


if __name__ == "__main__":
    main()
