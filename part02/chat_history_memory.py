from http.client import responses

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Tongyi

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an assistant who's good at {ability}. Respond in 20 words for fewer"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)
model = Tongyi()
runable = prompt | model

# 内存 存储会话
store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history")

responses = with_message_history.invoke(
    {"ability":"math", "input":"余弦是什么意思"},
    config={"configurable":{"session_id":"abc123"}}
)
print(responses)


responses = with_message_history.invoke(
    {"ability":"math", "input":"什么"},
    config={"configurable":{"session_id":"abc123"}}
)
print(responses)

responses = with_message_history.invoke(
    {"ability":"math", "input":"什么"},
    config={"configurable":{"session_id":"abc456"}}
)
print(responses)
