from http.client import responses

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Tongyi

import os

# 消息裁剪

temp_chat_history = ChatMessageHistory()
temp_chat_history.add_user_message("hello my name is tom")
temp_chat_history.add_ai_message("hello")
temp_chat_history.add_user_message("i am fine")
temp_chat_history.add_ai_message("你今天心情怎么样")
temp_chat_history.add_user_message("我下午在打篮球")
temp_chat_history.add_ai_message("你下午在做什么")
temp_chat_history.messages


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an assistant who's good at {ability}. Respond in 20 words for fewer"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)
model = Tongyi()
chain = prompt | model

# 存储最后两条数据
def trim_message(chain_input):
    stored_message = temp_chat_history.messages
    if(len(stored_message) <=2):
        return False
    temp_chat_history.clear()
    for message in stored_message[-2:]:
        temp_chat_history.add_message(message)
    return True

with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

chain_with_triming =(
        RunnablePassthrough.assign(messages_trimmed=trim_message) | with_message_history
)

# 测试后两条数据
responses = chain_with_triming.invoke(
    {"ability":"math", "input":"我下午在干什么"},
    config={"configurable":{"session_id":"abc123"}}
)
print(responses)

# 测试前两条数据
responses = chain_with_triming.invoke(
    {"ability":"math", "input":"我的名字是"},
    config={"configurable":{"session_id":"abc123"}}
)
print(responses)

