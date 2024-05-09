from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

"""
Github: https://github.com/zhangdapeng520/zdppy_langchainx
file: study/official_doc/c04_retraivel_chain.py
"""

# 创建模型对象
llm = Ollama(model="llama2")

# 构建提示词
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

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

# 构建链条
document_chain = create_stuff_documents_chain(llm, prompt)

# 接着，使用链条和大模型聊天
response = document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
print(response)