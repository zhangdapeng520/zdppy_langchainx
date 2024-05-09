from langchain_community.llms import Ollama

llm = Ollama(model="qwen:4b-chat")
response = llm.invoke("how can langsmith help with testing?")
print(response)