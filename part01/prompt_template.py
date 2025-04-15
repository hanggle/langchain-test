from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# 字符串提示词模板
prompt_template = PromptTemplate.from_template(
    "告诉我你的{p1}和{p2}"
)

result = prompt_template.format(p1="身高", p2="体重")
print("PromptTemplate:\n" + result)

invoke = prompt_template.invoke({"p1": "身高", "p2": "体重"})
print("PromptTemplate:")
print(invoke)

# 聊天提示词模板
from langchain_core.prompts import ChatPromptTemplate

messages = ChatPromptTemplate.from_messages(
    [("system", "you are a helpful person"),
     ("user", "tell me a joke about{topic}")])

template_invoke = messages.invoke({"topic": "dog"})
print("ChatPromptTemplate:\n", template_invoke)

# 消息占位符
from langchain_core.prompts import MessagesPlaceholder

prompt_template  = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful person"),
    MessagesPlaceholder("msgs")
])
prompt_template_invoke = prompt_template.invoke({"msgs": [HumanMessage(content="hhhh"), HumanMessage(content="ggg")]})
print("MessagesPlaceholder:\n", prompt_template_invoke)
