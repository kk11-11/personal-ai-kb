# verify_embedding.py
# 本地模型版（不依赖外网）：模型已从 ModelScope 下载到 ./models/bge-small-zh-v1.5
import os

# 强制离线，彻底禁止任何联网下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

MODEL_DIR = "./models/bge-small-zh-v1.5"  # 本地模型目录


def main():
    print(f"从本地加载模型: {MODEL_DIR}")
    model = SentenceTransformer(MODEL_DIR)

    sentences = [
        "你好，这是一个测试句子。",
        "我今天想搭一个个人 AI 知识库网页。",
    ]
    embeddings = model.encode(sentences)

    print("✅ 模型加载成功！")
    print("向量 shape:", embeddings.shape)  # 期望 (2, 512)
    print("第一条向量前5维:", [round(float(x), 4) for x in embeddings[0][:5]])


if __name__ == "__main__":
    main()
