# app.py
# 个人 AI 知识库 - Flask Web 版
import os
import glob
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import chromadb
import pdfplumber
import docx
from openai import OpenAI

# ====== 路径配置(以本文件所在目录为根)======
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models/bge-small-zh-v1.5")
DB_DIR = os.path.join(APP_DIR, "chroma_db")
DOCS_DIR = os.path.join(APP_DIR, "docs")
COLLECTION = "personal_kb"
TOP_K = 3

# ====== 启动时:加载模型 + 初始化 Chroma(只加载一次)======
print("🔧 启动中: 加载本地向量模型 ...")
embedder = SentenceTransformer(MODEL_DIR)
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION)

API_KEY = os.environ.get("ZHIPU_API_KEY")
if not API_KEY:
    raise SystemExit(
        "❌ 没找到 ZHIPU_API_KEY。PowerShell: $env:ZHIPU_API_KEY='你的key'"
    )
llm = OpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")


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
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, size: int = 500, overlap: int = 80):
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


def reingest() -> int:
    """扫描 docs/, 切分, 向量化, 重建知识库。返回入库片段数。"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    paths = sorted(
        glob.glob(os.path.join(DOCS_DIR, "*.md"))
        + glob.glob(os.path.join(DOCS_DIR, "*.txt"))
        + glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
        + glob.glob(os.path.join(DOCS_DIR, "*.docx"))
    )
    if not paths:
        return 0
    try:
        chroma_client.delete_collection(name=COLLECTION)
    except Exception:
        pass
    coll = chroma_client.get_or_create_collection(name=COLLECTION)

    all_chunks, all_metas, all_ids = [], [], []
    for p in paths:
        text = read_text(p)
        for i, c in enumerate(split_into_chunks(text)):
            all_chunks.append(c)
            all_metas.append({"source": os.path.basename(p), "chunk": i})
            all_ids.append(f"{os.path.basename(p)}::{i}")
    if not all_chunks:
        return 0
    embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()
    coll.add(documents=all_chunks, embeddings=embeddings, metadatas=all_metas, ids=all_ids)
    # 关键:重建后,把全局句柄也指向新集合(否则 /ask 用的是被删的旧句柄)
    global collection
    collection = coll
    return len(all_chunks)


def answer_question(q: str):
    """检索 + 智谱回答。返回 (answer, sources)。"""
    if collection.count() == 0:
        return ("⚠️ 知识库是空的。请先在页面上传一些文档。", [])
    q_emb = embedder.encode([q]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=TOP_K)
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
    resp = llm.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content
    sources = [
        {"source": m["source"], "chunk": m["chunk"], "preview": d[:120]}
        for d, m in zip(docs, metas)
    ]
    return answer, sources


# ====== Flask 路由 ======
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/docs", methods=["GET"])
def list_docs():
    files = []
    for ext in ("*.md", "*.txt", "*.pdf", "*.docx"):
        files.extend(sorted(glob.glob(os.path.join(DOCS_DIR, ext))))
    files = [os.path.basename(f) for f in files]
    return jsonify({"ok": True, "files": files, "chunks": collection.count()})


@app.route("/upload", methods=["POST"])
def upload():
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = request.files.getlist("files")
    saved = []
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".md", ".txt", ".pdf", ".docx"):
            continue
        dest = os.path.join(DOCS_DIR, f.filename)
        f.save(dest)
        saved.append(f.filename)
    n = reingest()
    return jsonify({"ok": True, "saved": saved, "chunks": n})


@app.route("/delete", methods=["POST"])
def delete_doc():
    data = request.get_json(silent=True) or {}
    fname = (data.get("filename") or "").strip()
    if not fname:
        return jsonify({"ok": False, "error": "文件名不能为空"}), 400
    # 防目录穿越:只用纯文件名,丢弃任何路径成分
    safe = os.path.basename(fname)
    path = os.path.join(DOCS_DIR, safe)
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "文件不存在"}), 404
    try:
        os.remove(path)
        # 只删该来源对应的片段,不必全量重建(比 reingest 更轻量)
        collection.delete(where={"source": safe})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "deleted": safe, "chunks": collection.count()})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400
    try:
        answer, sources = answer_question(q)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "answer": answer, "sources": sources})


if __name__ == "__main__":
    print("🚀 个人 AI 知识库 已启动: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
