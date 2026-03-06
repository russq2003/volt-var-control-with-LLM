# RAG操作程序
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from Qwen import call_qwen
from langchain_community.chains import RetrievalQA

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_text("这是一个测试文本，用于演示RAG操作程序的文本分割功能。我们将这个文本分割成多个片段，每个片段的长度不超过1000个字符，并且相邻片段之间有200个字符的重叠。这样可以确保在处理长文本时不会丢失重要的信息，同时也能提高模型的理解能力。")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

dimension = 384
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings, dtype=np.float32))

query_embedding = model.encode(["测试文本的查询"])
distances, indices = index.search(np.array(query_embedding), k=5)

def qwen_llm(prompt, max_tokens=800):
    # 收集流式输出
    result = ""
    for chunk in call_qwen(prompt, max_tokens=max_tokens):
        result += chunk
    return result

# 用qwen_llm作为llm接口
chain = RetrievalQA.from_chain_type(llm=qwen_llm,
                                    retriever=index.as_retriever(),
                                    return_source_documents=True)

results = chain.run("测试文本的查询")
print(results)
