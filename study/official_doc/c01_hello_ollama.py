import os
from langchain_community.llms import Ollama

# 假设你有一个运行在特定IP地址上的模型服务
MODEL_SERVICE_IP = "192.168.77.129"
MODEL_SERVICE_PORT = "11434"

# 设置环境变量，指定代理服务器的IP地址和端口
os.environ['http_proxy'] = f"http://{MODEL_SERVICE_IP}:{MODEL_SERVICE_PORT}"
os.environ['https_proxy'] = f"http://{MODEL_SERVICE_IP}:{MODEL_SERVICE_PORT}"

llm = Ollama(model="qwen:4b-chat")
response = llm.invoke("how can langsmith help with testing?")
print(response)