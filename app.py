# app.py
# 个人 AI 知识库 - Flask Web 版
import os
import glob
import json
from flask import Flask, request, jsonify, render_template, Response
from sentence_transformers import SentenceTransformer
import chromadb
import pdfplumber
import docx
from openai import OpenAI
import sys
import time
import threading
from agent import build_tools, run_react, summarize_history
from history_retrieval import prepare_history_vector
# Windows 下强制 stdout/stderr 为 utf-8, 避免 print 与默认编码报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ====== 路径配置(以本文件所在目录为根)======
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models/bge-small-zh-v1.5")
DB_DIR = os.path.join(APP_DIR, "chroma_db")
DOCS_DIR = os.path.join(APP_DIR, "docs")
COLLECTION = "personal_kb"
TOP_K = 3
RERANK_DIR = os.path.join(APP_DIR, "models/bge-reranker-base")  # 二阶段重排模型(可选,缺失则自动降级)
RERANK_TOP_K = 10  # 第一阶段向量召回候选数, 再交给 ReRank 精排取 TOP_K

# ====== 模型自动保障:本地缺失则从 ModelScope 下载(国内可访问)======
def ensure_model():
    if os.path.isdir(MODEL_DIR):
        return
    print("📥 未检测到本地向量模型,正在从 ModelScope 下载 BAAI/bge-small-zh-v1.5 ...")
    try:
        from modelscope import snapshot_download
        snapshot_download("BAAI/bge-small-zh-v1.5", local_dir=MODEL_DIR)
    except Exception as e:
        raise SystemExit(f"❌ 模型下载失败: {e}\n请手动把模型放到 {MODEL_DIR}")


def ensure_reranker(async_download: bool = True):
    """可选:云端首次启动自动从 ModelScope 下载 ReRank 模型,保持 100% 检索命中率。
    仅在环境变量 AUTO_DOWNLOAD_RERANKER=1 时尝试;否则缺失即降级为纯向量检索(94.4%)。

    默认后台线程下载(async_download=True):**不阻塞服务启动**。
    ReRank 模型约 1GB,若同步下载会拖住 gunicorn 起不来,导致 PaaS 判定启动超时 /
    健康检查失败(表现为日志"卡住不动"、部署一直转圈)。改为后台下载后:服务先用
    纯向量检索对外提供(94.4%),下载完成即自动启用二阶段重排(100%),无需重启。
    """
    if os.environ.get("AUTO_DOWNLOAD_RERANKER") != "1":
        return
    if os.path.isdir(RERANK_DIR):
        return

    def _download():
        print("📥 后台下载 ReRank 模型 BAAI/bge-reranker-base ...")
        try:
            from modelscope import snapshot_download
            snapshot_download("BAAI/bge-reranker-base", local_dir=RERANK_DIR)
            print("✅ ReRank 模型就绪,后续检索自动启用二阶段重排")
        except Exception as e:
            print(f"⚠️ ReRank 模型下载失败,将降级为纯向量检索: {e}")

    if async_download:
        threading.Thread(target=_download, daemon=True).start()
        print("ℹ️  ReRank 模型转入后台下载(期间用纯向量检索,完成后自动启用)")
    else:
        _download()


# ====== 启动时:加载模型 + 初始化 Chroma(只加载一次)======
print("🔧 启动中: 加载本地向量模型 ...")
ensure_model()
ensure_reranker()  # 可选:云端按需下载 ReRank 模型(默认不下载,自动降级)
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


# 知识库为空且有资料时,自动建库(提升首次体验;本地 / Docker 通用)
# 注意:必须放在 reingest 定义之后。原来这段写在文件前部,执行时 reingest 尚未定义会抛 NameError,
# 又被 except 静默吞掉 —— 导致"自动建库"从未真正生效,新 clone 的项目知识库一直是空的。
if collection.count() == 0:
    try:
        _n = reingest()
        if _n:
            print(f"✅ 首次启动自动建库完成, 共 {_n} 个片段")
    except Exception as e:
        print("⚠️ 自动建库跳过:", e)


# ====== Agent 工具表(ReAct 用)======
# 传 lambda 而不是 collection 对象: reingest() 会把全局 collection 重指向新集合,
# 工具每次取最新的句柄才不会查到失效集合。
AGENT_TOOLS = build_tools(embedder, lambda: collection)


def answer_question(q: str, history=None, history_mode="summary"):
    """检索 + 智谱回答。history: 最近几轮 [(q,a),...] 用于指代消歧。返回 (answer, sources)。"""
    if collection.count() == 0:
        return ("⚠️ 知识库是空的。请先在页面上传一些文档。", [])
    q_emb = embedder.encode([q]).tolist()
    # 第一阶段:向量召回更多候选(给 ReRank 留空间; 不超过知识库实际片段数)
    n_candidates = min(RERANK_TOP_K, collection.count())
    res = collection.query(query_embeddings=q_emb, n_results=n_candidates)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    # 第二阶段:ReRank 交叉编码器精排(模型缺失/失败自动降级,不影响可用性)
    docs, metas = rerank(q, docs, metas, top_n=TOP_K)
    context = "\n\n".join(
        f"[{m['source']}#{m['chunk']}]\n{d}" for d, m in zip(docs, metas)
    )
    # 拼接对话历史: 超过阈值时先摘要压缩早期轮次
    # 每条加显式"第 N 轮"编号, 让模型能精准定位"第一个问题""第 3 轮""刚才"这类相对指代
    # (检索仍锚定当前问题避免漂移; 长对话不会无限膨胀 token)
    hist_text = ""
    if history:
        if history_mode == "vector":
            # 方案B: 向量化历史召回 - 按当前问题语义召回 top_k 轮原文(保留细节, 但召回是语义的)
            # 用于和方案A(摘要压缩)做长对话记忆对比; 默认仍走 summary 分支, 生产行为不变
            hist_text = prepare_history_vector(history, q, embedder, top_k=6)
        else:
            # 方案A(默认): 摘要压缩 - 早期轮次压摘要, 最近 keep_recent 轮原文
            summary, recent = summarize_history(history, llm)
            if summary:
                hist_text = f"[早期对话摘要]\n{summary}\n\n"
            # recent 是 history 末尾 K 条; 全局轮次编号 = len(history) - len(recent) + 1 + i
            K = len(recent)
            start_turn = len(history) - K + 1
            recent_lines = []
            for i, h in enumerate(recent):
                turn_no = start_turn + i
                recent_lines.append(
                    f"【第 {turn_no} 轮】\n用户: {h.get('q', '')}\n助手: {h.get('a', '')}"
                )
            hist_text += "\n\n".join(recent_lines)
    prompt = (
        "你是用户的个人知识库助手。\n"
        "【资料】是你回答的事实依据,优先基于【资料】作答;若【资料】里没有相关信息,不要编造。\n"
        "【对话历史】(含可能的[早期对话摘要])里每条都带『第 N 轮』编号,可结合它:\n"
        "  1) 理解用户问题中代词(它/这个/上面/那个/刚才)的指代;\n"
        "  2) 当用户说『第一个问题』『第二个问题』『第 X 轮』『最前面那个』『前面聊到的』时, 按编号精确定位对应轮次, 不要凭感觉挑一条;\n"
        "  3) 当用户明显在要求『总结 / 回顾 / 综合前面的回答』时, "
        "     允许直接引用【对话历史】里已记录的助手回答来作答——那些回答本身是基于资料生成的, 可以安全复用。\n"
        "注意:不要凭空编造【资料】和【对话历史】里都没有的新事实。\n\n"
        f"【资料】\n{context}\n\n"
    )
    if hist_text:
        prompt += f"【对话历史】\n{hist_text}\n\n"
    prompt += (
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
# 让 jsonify 也走 ensure_ascii=False(防御性, 主路径已绕开)
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass


# ====== 全局异常兜底:任何未捕获异常都返回安全 JSON,绝不抛 ascii codec 错误 ======
@app.errorhandler(Exception)
def handle_all(e):
    import traceback
    print("❌ 未捕获异常:\n", traceback.format_exc())
    return json_response({"ok": False, "error": _safe_text(e)}, status=500)


def _safe_text(s) -> str:
    """把任意对象转成纯 str; 含奇怪的 unicode 时做一次 utf-8 兜底, 避免 json.dumps 阶段再次踩 ASCII 坑。"""
    try:
        return str(s)
    except Exception:
        try:
            return repr(s).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        except Exception:
            return "<unprintable>"


def json_response(data, status: int = 200):
    """绕开 Flask jsonify, 直接 json.dumps(ensure_ascii=False) 并显式声明 charset=utf-8。彻底规避 ASCII codec 错误。"""
    body = json.dumps(data, ensure_ascii=False, default=str)
    return Response(body, status=status, content_type="application/json; charset=utf-8")


# ====== 用户反馈存储(轻量 JSON + 锁, 不影响主问答性能)======
FEEDBACK_PATH = os.path.join(APP_DIR, "feedback.json")
_fb_lock = threading.Lock()

def save_feedback(entry: dict) -> dict:
    """追加一条反馈到 feedback.json, 返回最新统计(供复盘)。"""
    entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with _fb_lock:
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append(entry)
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return feedback_stats()

def feedback_stats() -> dict:
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
    pos = sum(1 for x in data if x.get("rating") == "up")
    neg = sum(1 for x in data if x.get("rating") == "down")
    return {"ok": True, "total": len(data), "positive": pos, "negative": neg}


# ====== 二阶段检索:向量召回 + ReRank 交叉编码器重排(模型缺失/失败则降级)======
_reranker = None
_reranker_lock = threading.Lock()

def get_reranker():
    """懒加载 ReRank 交叉编码器; 模型目录不存在或加载失败都返回 None(走降级)。"""
    global _reranker
    if _reranker is not None:
        return _reranker
    if not os.path.isdir(RERANK_DIR):
        return None
    with _reranker_lock:
        if _reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANK_DIR)
                print("✅ ReRank 模型已加载,启用二阶段重排")
            except Exception as e:
                print("⚠️ ReRank 模型加载失败,降级为纯向量检索:", _safe_text(e))
                return None
    return _reranker

def rerank(q: str, docs, metas, top_n: int = TOP_K):
    """用 ReRank 模型对候选片段精排,取 top_n。任意异常都降级为原始顺序。"""
    model = get_reranker()
    if model is None or not docs:
        return docs[:top_n], metas[:top_n]
    try:
        pairs = [[q, d] for d in docs]
        scores = model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)[:top_n]
        return [docs[i] for i in order], [metas[i] for i in order]
    except Exception as e:
        print("⚠️ ReRank 推理失败,降级为纯向量检索:", _safe_text(e))
        return docs[:top_n], metas[:top_n]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/docs", methods=["GET"])
def list_docs():
    files = []
    for ext in ("*.md", "*.txt", "*.pdf", "*.docx"):
        files.extend(sorted(glob.glob(os.path.join(DOCS_DIR, ext))))
    files = [os.path.basename(f) for f in files]
    return json_response({"ok": True, "files": files, "chunks": collection.count()})


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
    return json_response({"ok": True, "saved": saved, "chunks": n})


@app.route("/delete", methods=["POST"])
def delete_doc():
    data = request.get_json(silent=True) or {}
    fname = (data.get("filename") or "").strip()
    if not fname:
        return json_response({"ok": False, "error": "文件名不能为空"}, status=400)
    # 防目录穿越:只用纯文件名,丢弃任何路径成分
    safe = os.path.basename(fname)
    path = os.path.join(DOCS_DIR, safe)
    if not os.path.exists(path):
        return json_response({"ok": False, "error": "文件不存在"}, status=404)
    try:
        os.remove(path)
        # 只删该来源对应的片段,不必全量重建(比 reingest 更轻量)
        collection.delete(where={"source": safe})
    except Exception as e:
        # 防止错误信息本身含特殊字符在序列化时再次编码失败
        return json_response({"ok": False, "error": _safe_text(e)}, status=500)
    return json_response({"ok": True, "deleted": safe, "chunks": collection.count()})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    history = data.get("history") or []
    if not q:
        return json_response({"ok": False, "error": "问题不能为空"}, status=400)
    try:
        answer, sources = answer_question(q, history)
    except Exception as e:
        # 防止错误信息本身含特殊字符在序列化时再次编码失败
        return json_response({"ok": False, "error": _safe_text(e)}, status=500)
    return json_response({"ok": True, "answer": answer, "sources": sources})


@app.route("/ask_agent", methods=["POST"])
def ask_agent():
    """Agent 模式: 手写 ReAct 循环 + 工具调用。与 /ask 完全独立, 不影响原有问答链路。"""
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    history = data.get("history") or []
    if not q:
        return json_response({"ok": False, "error": "问题不能为空"}, status=400)
    try:
        max_steps = int(data.get("max_steps") or 4)
    except Exception:
        max_steps = 4
    try:
        answer, trace = run_react(
            q,
            llm,
            AGENT_TOOLS,
            model="glm-4-flash",
            max_steps=max_steps,
            history=history,
        )
    except Exception as e:
        return json_response({"ok": False, "error": _safe_text(e)}, status=500)
    return json_response({"ok": True, "answer": answer, "trace": trace})


@app.route("/feedback", methods=["POST"])
def feedback():
    """记录用户对某条回答的评分(👍/👎), 只存该轮内容, 不影响问答链路。"""
    data = request.get_json(silent=True) or {}
    rating = data.get("rating")
    if rating not in ("up", "down"):
        return json_response({"ok": False, "error": "rating 必须是 up 或 down"}, status=400)
    # 只记该轮的问题/答案/来源, 来源截断避免文件过大
    entry = {
        "rating": rating,
        "question": (data.get("question") or "")[:500],
        "answer": (data.get("answer") or "")[:2000],
        "sources": [
            {"source": s.get("source"), "chunk": s.get("chunk")}
            for s in (data.get("sources") or [])[:3]
        ],
    }
    try:
        stats = save_feedback(entry)
    except Exception as e:
        return json_response({"ok": False, "error": _safe_text(e)}, status=500)
    return json_response(stats)


@app.route("/feedback/stats", methods=["GET"])
def feedback_stats_route():
    return json_response(feedback_stats())


@app.route("/health", methods=["GET"])
def health_route():
    """健康检查:供部署平台探活(PaaS 据此判断实例存活、决定是否路由流量)。"""
    return json_response({"ok": True, "status": "running"})


if __name__ == "__main__":
    print("🚀 个人 AI 知识库 已启动 (json_response utf-8 安全模式): http://0.0.0.0:5000")
    print("   如果下面没见到这行,说明你跑的是旧代码 —— 请先彻底关闭旧进程再启动!")
    host = os.environ.get("HOST", "0.0.0.0")
    # 端口可用环境变量覆盖, 方便调试时另开实例而不影响已在跑的 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=False)
