

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.5-9b:latest",
    temperature=0,
)

response = llm.invoke("Explain what an AI agent is in one sentence.")

print(response.content)
