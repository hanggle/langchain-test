import base64
import os

import httpx
from langchain.smith.evaluation.runner_utils import ChatModelInput
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.tongyi import Tongyi
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate


image_url = "https://images.pexels.com/photos/2178371/pexels-photo-2178371.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200"
image_url2 = "https://images.pexels.com/photos/13146110/pexels-photo-13146110.jpeg?auto=compress&cs=tinysrgb&w=1200&lazy=load"

image_base64 = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
model = Tongyi()
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

resp = model.invoke([message])
print(resp)
