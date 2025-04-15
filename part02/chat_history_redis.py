from email.policy import default

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Tongyi
from langchain_community.chat_message_histories import RedisChatMessageHistory

import os

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an assistant who's good at {ability}. Respond in 20 words for fewer"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)
model = Tongyi()
runable = prompt | model

REDIS_URL = "redis://localhost:6379/0"


def get_session_history(user_id:str, conversation_id)->RedisChatMessageHistory:
    return RedisChatMessageHistory(user_id+conversation_id, url=REDIS_URL)

with_message_history = RunnableWithMessageHistory(
    runable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="user ID",
            description="用户标识",
            default = "",
            is_shared=True
        ),ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="conversation ID",
            description="会话标识",
            default = "",
            is_shared=True
        ),
    ]
)

responses = with_message_history.invoke(
    {"ability":"math", "input":"余弦是什么意思"},
    config={"configurable":{"user_id":"abc123", "conversation_id":"123"}}
)
print(responses)

responses = with_message_history.invoke(
    {"ability":"math", "input":"什么"},
    config={"configurable":{"user_id":"abc123", "conversation_id":"123"}}
)
print(responses)

responses = with_message_history.invoke(
    {"ability":"math", "input":"什么"},
    config={"configurable":{"user_id":"abc123", "conversation_id":"456"}}
)
print(responses)
