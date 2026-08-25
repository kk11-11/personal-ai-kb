# test_llm.py
# 目的：测试智谱 GLM-4-flash API 连通性
# 用法（cmd 终端）：
#   set ZHIPU_API_KEY=你的key
#   python test_llm.py
import os
from openai import OpenAI

API_KEY = os.environ.get("ZHIPU_API_KEY")
if not API_KEY:
    raise SystemExit(
        "❌ 没找到 ZHIPU_API_KEY 环境变量。\n"
        "请先在终端执行：set ZHIPU_API_KEY=你的key  （PowerShell 用 $env:ZHIPU_API_KEY='你的key'）"
    )

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"  # 免费额度模型

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "用一句话介绍你自己。"}],
)

print("✅ 智谱 GLM-4-flash 调用成功！")
print("模型回答:", resp.choices[0].message.content)
