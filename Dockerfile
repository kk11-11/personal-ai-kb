# 个人 AI 知识库 - Docker 镜像
# 基础镜像:Python 3.11 (slim 版体积小)
FROM python:3.11-slim

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

# 启动(模型缺失时由 app.py 自动从 ModelScope 下载)
CMD ["python", "app.py"]
