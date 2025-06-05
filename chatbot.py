import gradio as gr
import requests
import json
import os
from http import HTTPStatus
from dashscope import Application
# 自定义 CSS，用于美化聊天界面

css = """ 
    body {
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }
    .gradio-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gradio-container footer {
        display: none !important;
    }
    #chatbot {
        max-height: calc(100vh - 225px) !important;
        height: calc(100vh - 225px) !important;
        width: 100% !important;
        overflow-y: hidden;
        border: none;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .chat-message.user {
        background-color: #e8f4ff;
        text-align: right;
        color: #0056b3;
        margin-left: auto;
    }
    .chat-message.assistant {
        background-color: #f1f1f1;
        text-align: left;
        color: #333;
        margin-right: auto;
    }
    .multimodal-textbox {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
.fillable.svelte-a3xscf.app {
    padding: 5px !important;
    width:100%;
    margin: 5px ;
}
.gradio-container .gradio-container-5-23-1 .svelte-a3xscf{
    width:100%;
}
    .fillable.svelte-a3xscf.svelte-a3xscf:not(.fill_width){
        max-width: 726px;
    }
"""

def read_content_file():
    try:
        with open('content.txt', 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""

# 获取content.txt的内容
content = read_content_file()
print(content)

# 处理用户输入
def deal_fn(history, message):
    if message['text'] is not None:
        history.append((message['text'], None))  # 添加用户输入到聊天历史
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def filter_think_tags(text):
    """过滤掉<think>标签中的内容"""
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# 处理模型响应
def bot(history, model_type):
    history_new = []
    for user, assistant in history[:-1]:
        history_new.append({'role': "user", 'content': user})
        history_new.append({'role': "assistant", 'content': assistant})
    history_new.append({'role': "user", 'content': history[-1][0]})
    history[-1][1] = ''  # 初始化响应

    if model_type == "DeepSeek":
        # 调用 Ollama 获取完整响应
        try:
            responses = ollama(history_new)
            for res in responses:
                if 'response' in res:
                    # 过滤掉<think>标签中的内容
                    filtered_response = filter_think_tags(res['response'])
                    # 只更新最新的响应
                    history[-1][1] = filtered_response
                    # 只返回最新的对话
                    yield [history[-1]]
        except Exception as e:
            print(f"Error in bot function: {str(e)}")
            history[-1][1] = "抱歉，处理请求时出现错误。"
            yield [history[-1]]
    elif model_type == "通义千问":
        # 调用通义千问API
        response_text = call_dashscope(history[-1][0])
        history[-1][1] = response_text
        yield [history[-1]]

# 与 Ollama 模型交互的函数
def ollama(history):
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    
    # 构建包含文件内容的prompt
    system_prompt = f"请基于以下信息回答问题：\n{content}\n上面的内容是用户提供的文档内容，若用户询问的问题文档内容相关，请回答；若无关，请回复未在知识库中检索到相关内容\n"
    user_prompt = history[-1]['content']  # 只使用最新的用户输入

    # 构建完整的对话历史
    conversation_history = ""
    for entry in history[:-1]:
        if entry['role'] == 'user':
            conversation_history += f"用户: {entry['content']}\n"
        elif entry['role'] == 'assistant':
            conversation_history += f"助手: {entry['content']}\n"

    full_prompt = f"{system_prompt}{conversation_history}用户询问: {user_prompt}\n助手: "
    print("full_prompt:", full_prompt)

    data = {
        'model': 'deepseek-r1:1.5b',
        'prompt': full_prompt,
        'parameters': {
            'max_token': 10000,
            'temperature': 0.7,
        },
        "stream": False  # 关闭流式输出
    }

    try:
        # 发送请求并获取完整响应
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if 'response' in result:
            return [{'response': result['response']}]
        else:
            print("No response in result:", result)
            return [{'response': "抱歉，未能获取到有效响应。"}]
    except Exception as e:
        print(f"Error in ollama function: {str(e)}")
        return [{'response': "抱歉，处理请求时出现错误。"}]

# 处理通义千问的API调用
def call_dashscope(prompt):
    response = Application.call(
        api_key='sk-85bfee9facbb4237b398887f74245350',
        app_id='7163f5de4683483bb1a3dacc72891dc6',  # 应用ID替换YOUR_APP_ID
        prompt=prompt,
        rag_options={
            "pipeline_ids": ["xfxa8mc76j"],  # 替换为实际的知识库ID,逗号隔开多个
        }
    )
    if response.status_code != HTTPStatus.OK:
        print(f'request_id={response.request_id}')
        print(f'code={response.status_code}')
        print(f'message={response.message}')
        print(f'请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code')
    else:
        return response.output.text

custom_color = gr.themes.Color(
c50="#E8F5E9",  # 最浅
    c100="#E8F5E9",  # 最浅色
    c200="#C8E6C9",
    c300="#A5D6A7",
    c400="#81C784",
    c500="#66BB6A",  # 中等色
    c600="#4CAF50",
    c700="#43A047",
    c800="#388E3C",
    c900="#2E7D32",  # 最深色
    c950="#1B5E20"
)
# 构建 Gradio 界面
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=custom_color)) as demo:
    with gr.Row():
        model_type = gr.Dropdown(choices=["DeepSeek", "通义千问"], label="选择模型类型", value="DeepSeek")
    with gr.Row():
        chatbot = gr.Chatbot([], elem_id='chatbot', bubble_full_width=False, height=600)
    with gr.Row():
        chat_input = gr.MultimodalTextbox(
            file_types=[],
            interactive=True,
            placeholder='请在此输入',
            show_label=False
        )
    chat_msg = chat_input.submit(
        deal_fn,
        [chatbot, chat_input],
        [chatbot, chat_input],
    )
    bot_msg = chat_msg.then(
        bot,
        [chatbot, model_type],
        chatbot,
    )
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


demo.launch()