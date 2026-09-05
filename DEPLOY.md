# 部署指南（personal-ai-kb）

本项目是 **Flask 后端 + 前端** 应用，需要 Python 运行时，**不能用纯静态托管**（如只支持 HTML 的静态部署工具）。
下面给出四种可真正跑起来、拿到公网 demo 链接的方案（Hugging Face Spaces 免费首选）。任选其一。

---

## 0. 部署前必读（三个硬约束）

1. **必须配置环境变量 `ZHIPU_API_KEY`**
   云端没有你的本地 `.env`（已被 `.gitignore` 忽略）。在部署平台的环境变量设置里填入：
   ```
   ZHIPU_API_KEY=sk-你的真实key
   ```
   若缺失，容器启动会直接 `SystemExit` 退出。

2. **首次启动会自动下载模型 + 建库（约 1~3 分钟）**
   - `models/bge-small-zh-v1.5`（~130MB）从 ModelScope 自动下载；
   - `docs/` 三个 md 已进 git，启动时空库会自动 `reingest()` 建库；
   - 因此**无需手动准备知识库**，部署后等首次启动完成即可访问。

3. **想保持 100% 检索命中率（简历写的数字），需让云端也有 ReRank 模型**
   默认云端没有 `bge-reranker-base`，会自动降级为纯向量检索（命中率 94.4%）。
   设置环境变量开启自动下载：
   ```
   AUTO_DOWNLOAD_RERANKER=1
   ```
   首次启动会多下载 ~400MB，之后持久化。

---

## 1. Hugging Face Spaces（免费首选，README 已配 front matter）

最省事、免费、海外访问稳定，且本项目 README 已写好 `sdk: docker` 的 front matter，仓库关联后 HF 即识别为 Docker Space 自动构建。

1. 登录 huggingface.co → 右上角 **New Space** → 命名（如 `personal-ai-kb`）→ **Space 类型选 Docker**（不是 Gradio/Streamlit）→ 可见性选 Public（demo 需公网访问）或 Private。
2. 创建时选「Import from GitHub」关联 `kk11-11/personal-ai-kb`，或在 Space 设置里关联。HF 按仓库根目录 `Dockerfile` 自动 `docker build` 并部署，分配 `*.hf.space` 公网地址。
3. Space → Settings → **Secrets** 加环境变量：
   ```
   ZHIPU_API_KEY=sk-你的真实key
   AUTO_DOWNLOAD_RERANKER=1   # 可选,要 100% 检索命中率才加(首次多下载~1GB)
   ```
4. 首次部署自动下载模型（bge-small-zh ~92MB）+ 用 `docs/` 三篇 md 建库（约 1~3 分钟），期间显示「Building」。完成后 `/health` 返回 `{"ok":true}` 即上线。
5. 分配的 `*.hf.space` 即可演示链接，面试直接发。

> 若 HF 未自动识别 `sdk: docker`，在 Space 设置确认 Docker 运行环境即可；本项目 `Dockerfile` + `requirements.txt` 已就绪，无需改代码。

---

## 2. 腾讯云 CloudBase（容器版 / 云托管）

适合国内访问、与腾讯生态契合。

1. 在 CloudBase 控制台创建「容器服务」或「云托管」，源码方式选「代码仓库」（关联 GitHub 仓库 `kk11-11/personal-ai-kb`）。
2. 构建配置：
   - 运行环境：Docker（自动识别仓库根目录 `Dockerfile`）
   - 监听端口：填 `5000`（或平台给的 `$PORT`，代码已支持）
3. 环境变量：
   ```
   ZHIPU_API_KEY=sk-xxx
   AUTO_DOWNLOAD_RERANKER=1   # 可选,要 100% 命中率才加
   ```
4. 实例规格建议：内存 ≥ 1GB（sentence-transformers + chroma 占内存较大），单实例。
5. 部署完成后平台会分配一个公网域名（如 `xxx.apigw.tencentcs.com`），即 demo 链接。

---

## 3. Railway（最省事，国外）

1. 登录 railway.app，New Project → Deploy from GitHub repo → 选 `kk11-11/personal-ai-kb`。
2. Railway 自动识别 `Dockerfile` 构建。
3. Variables 里加 `ZHIPU_API_KEY`、`AUTO_DOWNLOAD_RERANKER=1`（可选）。
4. 默认会注入 `$PORT`，代码已绑定。
5. 部署后自动分配 `*.up.railway.app` 公网地址。

> 注意：Railway 免费额度有限，且国内访问可能慢。演示用足够。

---

## 4. Render（免费档，国外）

1. 登录 render.com，New → Web Service → 关联 GitHub 仓库。
2. Runtime 选 Docker，端口填 `5000`。
3. Environment → Add Environment Variable：`ZHIPU_API_KEY`、`AUTO_DOWNLOAD_RERANKER=1`（可选）。
4. 免费实例休眠后首次访问会冷启动（模型下载 + 建库），需等 1~3 分钟。

---

## 5. 本地 docker 自测（验证镜像没问题再上云）

```bash
cd personal-ai-kb
docker build -t personal-ai-kb .
docker run -d -p 5000:5000 \
  -e ZHIPU_API_KEY=sk-xxx \
  -e AUTO_DOWNLOAD_RERANKER=1 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/models:/app/models \
  --name kb personal-ai-kb
# 打开 http://localhost:5000 验证
curl http://localhost:5000/health
```

挂载 `chroma_db` / `models` 两个 volume，避免容器重启丢失向量库与模型。

---

## 6. 健康检查

应用提供 `GET /health` 返回 `{"ok": true}`（200），供平台探活。
