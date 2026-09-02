# 个人 AI 知识库 - Docker 镜像
# 基础镜像:Python 3.12 (slim 版体积小)
# 注: numpy>=2.5 和 torch>=2.11 都需要 Python>=3.12, 用 3.11-slim 装不上
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装编译依赖(sentence-transformers / chromadb 的底层依赖需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖(利用 Docker 层缓存,改代码后不必重装)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir modelscope

# 拷贝应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 启动:用 gunicorn 跑生产(而非 Flask dev server);绑定 $PORT(PaaS 注入,默认 5000)
# 单 worker(-w 1):sentence-transformers 模型占内存大,多 worker 会翻倍显存/内存
# --timeout 120:首次启动要下载模型+建库,可能超过默认 30s
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} -w 1 --timeout 120 --graceful-timeout 30 app:app"]
