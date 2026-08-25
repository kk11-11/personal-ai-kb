# ask.py
# 提问:从 Chroma 检索相关片段,喂给智谱 GLM-4-flash 回答
import os
import sys
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

MODEL_DIR = "./models/bge-small-zh-v1.5"
DB_DIR = "./chroma_db"
COLLECTION = "personal_kb"
TOP_K = 3

API_KEY = os.environ.get("ZHIPU_API_KEY")
if not API_KEY:
    raise SystemExit(
        "❌ 没找到 ZHIPU_API_KEY。PowerShell: $env:ZHIPU_API_KEY='你的key'\n"
        "cmd:        set ZHIPU_API_KEY=你的key"
    )

client_ai = OpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")


def main():
    if len(sys.argv) < 2:
        q = input("请输入你的问题: ").strip()
    else:
        q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("问题不能为空。")
        return

    print("加载本地向量模型...")
    model = SentenceTransformer(MODEL_DIR)
    client_db = chromadb.PersistentClient(path=DB_DIR)
    coll = client_db.get_or_create_collection(COLLECTION)
    if coll.count() == 0:
        print("⚠️ 知识库是空的。先跑 python ingest.py 把文档入库。")
        return

    q_emb = model.encode([q]).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context = "\n\n".join(
        f"[{m['source']}#{m['chunk']}]\n{d}" for d, m in zip(docs, metas)
    )

    prompt = (
        "你是用户的个人知识库助手。请严格根据下面提供的【资料】回答问题,"
        "如果资料里没有答案,请直说'资料里没找到相关信息',不要编造。\n\n"
        f"【资料】\n{context}\n\n"
        f"【问题】{q}\n\n"
        f"【回答】"
    )

    resp = client_ai.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content

    print("\n=== 回答 ===")
    print(answer)
    print("\n=== 参考资料 ===")
    for d, m in zip(docs, metas):
        print(f"- {m['source']}#{m['chunk']}: {d[:80]}...")


if __name__ == "__main__":
    main()
