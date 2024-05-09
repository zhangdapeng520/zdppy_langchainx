from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

"""
Github: https://github.com/zhangdapeng520/zdppy_langchainx
file: study/official_doc/c05_conversation_retravel_chain.py
"""

# 创建模型对象
llm = Ollama(model="qwen:0.5b")

# 构建提示词
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])


# 将结果解析为字符串
output_parser = StrOutputParser()

# 构建web加载器对象
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# 创建嵌入对象
embeddings = OllamaEmbeddings()

# 建立索引
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# 设置检索器
retriever = vector.as_retriever()
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# 调用
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response["answer"])