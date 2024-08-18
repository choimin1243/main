import os

from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response=chain.invoke({"text": "Explain the importance of low latency LLMs."})


print(response)