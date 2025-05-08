import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
import gradio as gr
history=[]
promot="你是一个历史学家"
def model(text):
    global history
    if not history:
        history.append({'role':'system','content':promot})
    history.append({'role':'user','content':text})
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=history
    )
    a=completion.choices[0].message.reasoning_content+completion.choices[0].message.content
    history.append({'role':'assistant','content':a})
    return a

demo = gr.Interface(fn=model, inputs="text", outputs="text",title="聊天机器人")
demo.launch(share=True)

