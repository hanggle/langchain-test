from langchain_community.chat_models import ChatTongyi
from langchain.schema import HumanMessage

# 设置你的 API Key
import os

# 初始化 ChatTongyi 模型
chat_model = ChatTongyi(model="Qwen2.5-Max")

# 定义图片 URL 和提示问题
image_url = "https://images.pexels.com/photos/2178371/pexels-photo-2178371.jpeg"
prompt = "这张图片中有什么内容？请详细描述。"

# 构建消息对象
messages = [
    HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_url}
        ]
    )
]

# 调用模型并获取结果
response = chat_model(messages)

# 输出结果
print("Qwen-Max 回答：")
print(response.content)