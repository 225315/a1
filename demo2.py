import os
from openai import OpenAI
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer('local_models/all-MiniLM-L6-v2')

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
knowledge=[    {"id": 1, "content": "RAG（检索增强生成）是一种将外部知识检索与大模型生成相结合的技术。"},
    {"id": 2, "content": "RAG的工作流程：先检索相关知识片段，再将其与问题一起输入给大模型。"},
    {"id": 3, "content": "FAISS是一个高效的向量检索库，常用于RAG系统中的知识检索。"},
    {"id": 4, "content": "RAG可以显著提升大模型回答的准确性和时效性。"},
    {"id": 5, "content": "使用RAG时，需要注意知识库的质量和更新频率。"}
        ]

#构建知识库
def build_faiss_index():
    embeddings=np.array([embedding_model.encode(doc["content"])for doc in knowledge]).astype("float32")
    dimension=embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
index=build_faiss_index()

#检索知识
def retrieve_relevant_docs(query,k=3):
    query_vector=embedding_model.encode(query).astype("float32").reshape(1,-1)
    distances,indices=index.search(query_vector,k)
    relevant_docs = [knowledge[idx]["content"] for idx in indices[0]]
    return relevant_docs
import gradio as gr
history=[]
def model(text):
    #检索相关知识
    relevant_docs=retrieve_relevant_docs(text)
    prompt=(f"""知识库参考内容：{"|".join(relevant_docs)}"
            用户问题:{text}
            请基于以上参考内容，提供准确、连贯的回答.""")
    global history
    history.append({'role':'user','content':prompt})
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=history
    )
    a=completion.choices[0].message.content
    history.append({'role':'assistant','content':a})

    refine_promot=(f"上述回答是否完整准确，如果不足请补充完整：{a}")
    history.append({'role':"user",'content':refine_promot})
    refine_completion=client.chat.completions.create(
        model="qwen-plus",
        messages=history
    )
    refine_answer=refine_completion.choices[0].message.content
    history.append({'role':"assistant",'content':refine_answer})
    return refine_answer

demo = gr.Interface(fn=model, inputs="text", outputs="text",title="聊天机器人")
demo.launch(share=True)

