from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi

import os


llm = Tongyi()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

result = chain.invoke({"question": question})
print(result)

