import os
from openai import OpenAI
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr


embedding_model = SentenceTransformer("local_models/all-MiniLM-L6-v2")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
knowledge = [
    {
        "id": 1,
        "content": "RAG（检索增强生成）是一种将外部知识检索与大模型生成相结合的技术。",
    },
    {
        "id": 2,
        "content": "RAG的工作流程：先检索相关知识片段，再将其与问题一起输入给大模型。",
    },
    {"id": 3, "content": "FAISS是一个高效的向量检索库，常用于RAG系统中的知识检索。"},
    {"id": 4, "content": "RAG可以显著提升大模型回答的准确性和时效性。"},
    {"id": 5, "content": "使用RAG时，需要注意知识库的质量和更新频率。"},
]


# 构建知识库
def build_faiss_index():
    embeddings = np.array(
        [embedding_model.encode(doc["content"]) for doc in knowledge]
    ).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


index = build_faiss_index()


# 检索知识
def retrieve_relevant_docs(query, k=3):
    query_vector = embedding_model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    relevant_docs = [knowledge[idx]["content"] for idx in indices[0]]
    return relevant_docs


history = []


def generate_feedback(initial_answer, query, konwledge_docs):
    prompt = f"""
            知识库参考内容：{"|".join(konwledge_docs)}"
            用户问题:{query}
            模型初始回答：{initial_answer}
            
            请从以下方面提供反馈：
            1、回答是否基于知识库内容？
            2、信息是否准确完整？
            3、是否回答了用户的核心问题？
            4、有无逻辑或事实错误？
            5、建议如何改进？
    
            请提供具体的改进建议，而不只是评价。
            """
    response = client.chat.completions.create(
        model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def refine_answerd(initial_answer, feedback, query, knowledge_docs):
    """根据反馈优化初始回答"""
    prompt = f"""
        知识库参考内容：{"|".join(knowledge_docs)}
        用户问题：{query}
        初始回答：{initial_answer}
        反馈意见：{feedback}

        请基于反馈，优化回答，确保：
        1. 所有信息都基于知识库
        2. 回答完整准确
        3. 逻辑清晰，易于理解

        优化后的回答：
        """

    response = client.chat.completions.create(
        model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def should_refine(initial_answer, query, knowledge_docs):
    prompt = f"""
    知识库参考内容：{"|".join(knowledge_docs)}
    用户问题：{query}
    模型初始回答：{initial_answer}
    
    请判断这个回答是否需要优化：
    - 如果回答完整、准确且基于知识库，请回答：不需要优化
    - 如果回答存在以下问题，请回答：需要优化
      1. 信息不完整
      2. 存在事实错误
      3. 逻辑不连贯
      4. 未充分利用知识库
      5. 其他明显问题
      
    仅需回答"需要优化"或"不需要优化"。"""
    response = client.chat.completions.create(
        model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
    )
    decision = response.choices[0].message.content
    return decision == "需要优化"


def model(text):
    # 检索相关知识
    relevant_docs = retrieve_relevant_docs(text)
    prompt = f"""知识库参考内容：{"|".join(relevant_docs)}"
            用户问题:{text}
            请基于以上参考内容，提供准确、连贯的回答."""
    global history
    history.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="deepseek-r1", messages=history)
    initial_answer = response.choices[0].message.content
    need_refine = should_refine(initial_answer, text, relevant_docs)

    if need_refine:
        feedback = generate_feedback(initial_answer, text, relevant_docs)
        refine_answer = refine_answerd(initial_answer, feedback, text, relevant_docs)
        history.append({"role": "assistant", "content": initial_answer})
        history.append({"role": "user", "content": f"请优化这个回答：{initial_answer}"})
        history.append({"role": "assistant", "content": refine_answer})
        return refine_answer
    else:
        history.append({"role": "assistant", "content": initial_answer})
        return initial_answer


demo = gr.Interface(fn=model, inputs="text", outputs="text", title="聊天机器人")
demo.launch(share=True)
