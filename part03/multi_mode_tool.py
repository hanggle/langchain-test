import base64
import os
from typing import Literal

import httpx
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms.tongyi import Tongyi
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model


@tool
def weather_tool(weather:Literal["晴朗的","多云的", "下雪的"])->None:
    """test"""
    pass


image_url = "https://images.pexels.com/photos/2178371/pexels-photo-2178371.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200"
image_url2 = "https://cdn.pixabay.com/photo/2016/11/29/05/29/buildings-1867550_640.jpg"

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "用中文描述图中的天气"
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url2},
        }
    ]
)

# 此处有问题：模型调用和调用工具有冲突，不能同时兼容
model = ChatTongyi(
                model="qwen-max",
                # top_p="...",
                # api_key="...",
                # other params...
            )

tool_model = model.bind_tools([weather_tool])
resp = tool_model.invoke([message])
print(resp)
print(resp.tool_calls)
