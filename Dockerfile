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

# 先单独装 CPU 版 torch —— 必须在 requirements.txt 之前,顺序很关键。
# PyPI 上的默认 torch 是 CUDA 版(约 2~3GB),本项目为纯 CPU 推理、完全用不上;
# 官方 CPU 源的同版本(torch==2.13.0+cpu)仅约 200MB,镜像体积因此降一个数量级,
# 构建与启动都大幅提速,也显著降低 PaaS 上超时/OOM 类失败的概率。
# 注意:若先装 sentence-transformers,它会把 CUDA 版 torch 依赖拉回来覆盖掉 CPU 版。
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0+cpu

# 再装其余依赖(requirements.txt 中刻意不含 torch,以免覆盖上面的 CPU 版)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir modelscope

# 拷贝应用代码
COPY . .

# 暴露端口(魔搭创空间 Docker 类型强制要求 7860; CloudBase 等注入 PORT 时自动读取)
EXPOSE 7860

# 启动:用 gunicorn 跑生产(而非 Flask dev server);绑定 $PORT(PaaS 注入,默认 7860)
# 单 worker(-w 1):sentence-transformers 模型占内存大,多 worker 会翻倍显存/内存
# --timeout 120:首次启动要下载模型+建库,可能超过默认 30s
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} -w 1 --timeout 120 --graceful-timeout 30 app:app"]
