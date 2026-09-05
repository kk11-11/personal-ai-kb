# 部署指南（personal-ai-kb）

本项目是 **Flask 后端 + 前端** 应用，需要 Python 运行时，**不能用纯静态托管**（如只支持 HTML 的静态部署工具）。
下面给出几种可真正跑起来、拿到公网 demo 链接的方案。**国内访问优先选 CloudBase / ModelScope（国内可直连、免费或低成本）；Hugging Face Spaces 现需付费且国内常打不开，仅作海外备选。** 任选其一。

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
   - `docs/` 资料（已进 git）启动时空库会自动 `reingest()` 建库；
   - 因此**无需手动准备知识库**，部署后等首次启动完成即可访问。

3. **想保持 100% 检索命中率（简历写的数字），需让云端也有 ReRank 模型**
   默认云端没有 `bge-reranker-base`，会自动降级为纯向量检索（命中率 94.4%）。
   设置环境变量开启自动下载：
   ```
   AUTO_DOWNLOAD_RERANKER=1
   ```
   首次启动会多下载 ~400MB，之后持久化（取决于平台磁盘是否持久）。

---

## 1. 腾讯云 CloudBase（国内首选，推荐）

适合国内访问、与腾讯生态契合，有免费额度。

1. 在 CloudBase 控制台创建「容器服务」或「云托管」，源码方式选「代码仓库」（关联 GitHub 仓库 `kk11-11/personal-ai-kb`）。
2. 构建配置：
   - 运行环境：Docker（自动识别仓库根目录 `Dockerfile`）
   - 监听端口：填 `7860`（与容器默认监听一致；若平台注入 `$PORT` 则自动读取）
3. 环境变量：
   ```
   ZHIPU_API_KEY=sk-xxx
   AUTO_DOWNLOAD_RERANKER=1   # 可选,要 100% 命中率才加
   ```
4. 实例规格建议：内存 ≥ 1GB（sentence-transformers + chroma 占内存较大），单实例。
5. 部署完成后平台会分配一个公网域名（如 `xxx.apigw.tencentcs.com`），即 demo 链接。

---

## 2. ModelScope 魔搭创空间（国内备选，Docker，免费档）

国内 CDN 快、免费（2 核 CPU + 16G 内存档），支持自定义 Docker，适合"已有东西想展示出去"。

> 前置：需完成实名认证（绑定阿里云账号 + 云账号实名）。Docker 创空间目前为 Beta。

1. 在魔搭「创空间」列表 → 创建创空间 → 选 **Docker** 类型（或"编程式创空间 / 快速部署并创建"），上传项目文件夹或 Git 推送。
2. **端口必须为 7860**：本项目默认即监听 `7860`（无需再设 `PORT`，保持平台路由端口一致即可）。
3. 项目根目录已放好 `ms_deploy.json`（已随代码提交），内容：
   ```json
   {
     "sdk_type": "docker",
     "port": 7860,
     "resource_configuration": "platform/2v-cpu-16g-mem",
     "environment_variables": [
       { "key": "AUTO_DOWNLOAD_RERANKER", "value": "1" }
     ]
   }
   ```
   > `ZHIPU_API_KEY` **不要写进 `ms_deploy.json`**（会进 git 仓库泄露）。请在 Studio 创建后于「环境变量 / 密钥」设置里单独添加。
4. Secrets / 环境变量配：
   ```
   ZHIPU_API_KEY=sk-xxx
   PORT=7860
   AUTO_DOWNLOAD_RERANKER=1   # 可选
   ```
   （运行时环境变量会注入容器，代码用 `os.environ.get(...)` 读取即可；注意 Docker 创空间构建时不带环境变量，只有运行时注入。）
5. 部署完成后平台分配 `https://<空间名>.modelscope.cn`（或类似）公网地址。

> 注意：创空间磁盘默认**非持久**——容器重启会丢 `chroma_db` / `models`，但本项目启动会自动下载模型 + `reingest()` 建库，只是重启会慢 1~3 分钟，不影响可用性。

---

## 3. Hugging Face Spaces（海外备选，⚠️ 需付费 + 国内常打不开）

> ⚠️ **2026 年起政策变化**：创建一个会跑代码的 Space（Gradio 或 Docker）现在**需要付费套餐**（个人 PRO **$9/月**）。唯一仍免费的是静态 Space 和最多 2 个跑在 ZeroGPU 上的 Gradio Space。**且 `*.hf.space` 在国内经常被墙 / 超时，面试官也可能打不开。** 因此仅作海外备选，不作为国内用户主推。

若坚持使用（海外访问稳定、且本项目 README 已配 `sdk: docker` front matter）：

1. 登录 huggingface.co → 右上角 **New Space** → 命名 → **Space 类型选 Docker** → 可见性 Public。
2. 创建时选「Import from GitHub」关联 `kk11-11/personal-ai-kb`，HF 按根目录 `Dockerfile` 自动 `docker build` 并分配 `*.hf.space`。
3. Settings → **Secrets** 加：
   ```
   ZHIPU_API_KEY=sk-你的真实key
   AUTO_DOWNLOAD_RERANKER=1   # 可选,要 100% 命中率才加
   ```
4. 首次部署自动下载模型 + 用 `docs/` 资料建库（约 1~3 分钟）。完成后 `/health` 返回 `{"ok":true}` 即上线。

---

## 4. Railway（国外，最省事）

1. 登录 railway.app，New Project → Deploy from GitHub repo → 选 `kk11-11/personal-ai-kb`。
2. Railway 自动识别 `Dockerfile` 构建。
3. Variables 里加 `ZHIPU_API_KEY`、`AUTO_DOWNLOAD_RERANKER=1`（可选）。
4. 默认会注入 `$PORT`，代码已绑定。
5. 部署后自动分配 `*.up.railway.app` 公网地址。

> 注意：Railway 免费额度有限，且国内访问可能慢。

---

## 5. Render（国外，免费档）

1. 登录 render.com，New → Web Service → 关联 GitHub 仓库。
2. Runtime 选 Docker，端口填 `7860`。
3. Environment → Add Environment Variable：`ZHIPU_API_KEY`、`AUTO_DOWNLOAD_RERANKER=1`（可选）。
4. 免费实例休眠后首次访问会冷启动（模型下载 + 建库），需等 1~3 分钟。

---

## 6. 本地 docker 自测（验证镜像没问题再上云）

```bash
cd personal-ai-kb
docker build -t personal-ai-kb .
docker run -d -p 7860:7860 \
  -e ZHIPU_API_KEY=sk-xxx \
  -e AUTO_DOWNLOAD_RERANKER=1 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/models:/app/models \
  --name kb personal-ai-kb
# 打开 http://localhost:7860 验证
curl http://localhost:7860/health
```

挂载 `chroma_db` / `models` 两个 volume，避免容器重启丢失向量库与模型。

---

## 7. 健康检查

应用提供 `GET /health` 返回 `{"ok": true}`（200），供平台探活。
