# ingest.py
# 读取 ./docs/ 下的文档（.md / .txt / .pdf），切分，向量化，存入 Chroma
import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb
import pdfplumber
import docx

CHUNK_SIZE = 500        # 每段目标长度（字符）
CHUNK_OVERLAP = 80      # 段与段之间的重叠，避免上下文断裂
DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
COLLECTION = "personal_kb"
MODEL_DIR = "./models/bge-small-zh-v1.5"


def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                parts.append(t)
        return "\n".join(parts)
    if path.lower().endswith(".docx"):
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())
    # txt / md
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.strip().replace("\r\n", "\n")
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    paths = []
    for ext in ("*.md", "*.txt", "*.pdf", "*.docx"):
        paths.extend(glob.glob(os.path.join(DOCS_DIR, ext)))
    paths = sorted(set(paths))

    if not paths:
        print(f"⚠️  {DOCS_DIR}/ 下没有任何 .md / .txt / .pdf / .docx 文件。先放点资料进去再跑。")
        return

    print(f"📚 找到 {len(paths)} 个文档，开始处理...")

    print("加载本地向量模型...")
    model = SentenceTransformer(MODEL_DIR)

    client = chromadb.PersistentClient(path=DB_DIR)
    # 每次重建,避免重复入库
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.get_or_create_collection(COLLECTION)

    all_chunks, all_metas, all_ids = [], [], []
    for p in paths:
        text = read_text(p)
        chunks = split_into_chunks(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_metas.append({"source": os.path.basename(p), "chunk": i})
            all_ids.append(f"{os.path.basename(p)}::{i}")

    print(f"🔪 共切出 {len(all_chunks)} 个片段,开始向量化（首次约 1–2 分钟）...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    coll.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metas,
        ids=all_ids,
    )
    print(f"✅ 已存入 {DB_DIR},共 {len(all_chunks)} 条。")


if __name__ == "__main__":
    main()
