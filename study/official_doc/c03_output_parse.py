from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import os


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

# 将结果解析为字符串
output_parser = StrOutputParser()


# 构建链条，这个是langchain的核心之一，之前的基础教程中讲到过
chain = prompt | llm | output_parser

# 接着，使用链条和大模型聊天
response = chain.invoke({"input": "你认为langchain是一个优秀的框架吗？"})
print(response)

# 得到的回答如下
"""
LangChain是一个由开发者社区维护的开源项目。

从技术角度来看，LangChain提供了一系列强大的功能，如智能路由、语义解析等。这些功能对于实现高效的分布式系统非常关键。

然而，LangChain也存在一些问题和挑战，例如如何有效处理语义不清晰和错误的语料等问题。这些问题都需要通过持续的研发工作来逐步解决和完善。
"""