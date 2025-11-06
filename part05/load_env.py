import os
from dotenv import load_dotenv


load_dotenv()
print(os.getenv("Z_API_KEY"))

from langchain_community.chat_models import ChatZhipuAI, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 创建 LLM 实例
llm = ChatZhipuAI(
    temperature=0.6,
    model="glm-4.5",
    api_key=os.getenv("Z_API_KEY"),
)


prompt = ChatPromptTemplate.from_template("介绍下你的{topic}")

chain = prompt | llm | StrOutputParser()

res = chain.invoke({"topic": "模型"})
print(res)

