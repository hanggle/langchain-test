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
        ("system", "you are an assistant ."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ]
)
model = Tongyi()
chain = prompt | model


with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


def summary_message(chain_input):
    stored_message = temp_chat_history.messages
    if len(stored_message) ==0:
        return False
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            (
                "user",
                "将上述消息汇总成一句话",
            ),
        ]
    )
    summary_chain = summary_prompt | model
    sum_msg =summary_chain.invoke({"chat_history":stored_message})
    temp_chat_history.clear()
    temp_chat_history.add_message(sum_msg)
    return True


chain_with_summary =(
        RunnablePassthrough.assign(messages_summarized=summary_message)|with_message_history
)


responses = chain_with_summary.invoke(
    {"input":"名字，下午在干什么"},
    config={"configurable":{"session_id":"abc123"}}
)
print(responses)

