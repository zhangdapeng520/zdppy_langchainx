from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_community.llms import Ollama

"""
Github: https://github.com/zhangdapeng520/zdppy_langchainx
file: study/official_doc/c02_prompt_and_chain.py
"""

# 假设你有一个运行在特定IP地址上的模型服务
MODEL_SERVICE_IP = "192.168.77.129"
MODEL_SERVICE_PORT = "11434"

# 设置环境变量，指定代理服务器的IP地址和端口
os.environ['http_proxy'] = f"http://{MODEL_SERVICE_IP}:{MODEL_SERVICE_PORT}"
os.environ['https_proxy'] = f"http://{MODEL_SERVICE_IP}:{MODEL_SERVICE_PORT}"

# 创建模型对象
llm = Ollama(model="qwen:4b-chat")

# 构建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个精通Python语言的高级Python工程师，你将帮助我解决Python开发中遇到的各种问题。"),
    ("user", "{input}")
])

# 构建链条，这个是langchain的核心之一，之前的基础教程中讲到过
chain = prompt | llm 

# 接着，使用链条和大模型聊天
response = chain.invoke({"input": "你认为langchain是一个优秀的框架吗？"})
print(response)

# 得到的回答如下
"""
Langchain确实是一款非常优秀的框架。它的设计理念是"简单易用、高效灵活、持续优化"。

 Langchain 框架的优势在于：它具有非常强大的功能，可以帮助开发者快速构建出高质量的软件系统；另外， Langchain 框架还具有非常好的灵活性和可扩展性，可以有效地满足不同用户的需求和要求。
"""