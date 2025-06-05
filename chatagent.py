from fastapi import FastAPI, File, Form, UploadFile
from qwen_agent.agents import Assistant
import os
from fastapi.responses import FileResponse
import cv2
import numpy as np
import gradio as gr
import tempfile
import shutil
import json
import subprocess

app = FastAPI()

# 设置环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-5d8c49c360074c78ac5f04c99285575b"

# 初始化 Qwen-Agent 的 Assistant 实例
assistant = Assistant(
    llm={'model': 'qwen-max'},
    function_list=['code_interpreter'],
    system_message="你是一个图像处理专家。用户会提供一张图片和处理要求，你需要编写Python代码来处理图片。处理后的图片必须保存为'output.png'。请使用OpenCV (cv2) 库进行图像处理。"
)

def format_path(path):
    """格式化路径，确保Windows路径正确显示"""
    return path.replace('\\', '/')

def open_image(image_path):
    """使用系统默认图片查看器打开图片"""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(image_path)
        elif os.name == 'posix':  # macOS 和 Linux
            if os.system('which xdg-open') == 0:  # Linux
                subprocess.run(['xdg-open', image_path])
            else:  # macOS
                subprocess.run(['open', image_path])
    except Exception as e:
        print(f"打开图片失败: {str(e)}")

def process_image(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    try:
        print("开始处理图片...")
        print(f"提示信息: {prompt}")

        # 获取响应
        responses = []
        for response in assistant.run(messages):
            print(f"收到响应: {response}")
            responses.append(response)

            # 处理每个响应
            if isinstance(response, list):
                for resp in response:
                    if isinstance(resp, dict):
                        if 'function_call' in resp:
                            # 执行代码解释器
                            try:
                                code_data = json.loads(resp['function_call']['arguments'])
                                if 'code' in code_data:
                                    code = code_data['code']
                                    print(f'执行代码：{code}')
                                    # 在安全的上下文中执行代码
                                    local_vars = {}
                                    exec(code, {'cv2': cv2, 'np': np, 'os': os}, local_vars)
                                    return "图片处理完成"
                            except json.JSONDecodeError:
                                print("JSON解析错误")
                            except Exception as e:
                                print(f"代码执行错误: {str(e)}")
                        elif 'content' in resp and resp['content']:
                            return resp['content']

        if not responses:
            return "未收到任何响应"

        # 如果没有找到function_call或content，返回最后一个响应
        return str(responses[-1])

    except Exception as e:
        print(f"代码执行失败: {str(e)}")
        return f"代码执行失败: {str(e)}"

def handle_image_upload(image, prompt):
    try:
        if image is None:
            return None
            
        # 保存上传的图片到根目录
        input_path = "input.png"
        cv2.imwrite(input_path, image)

        # 设置输出路径（根目录）
        output_path = "output.png"

        # 获取绝对路径并格式化
        abs_input_path = format_path(os.path.abspath(input_path))
        abs_output_path = format_path(os.path.abspath(output_path))

        # 构建完整的提示
        full_prompt = f"""
        请处理以下图片：
        1. 输入图片路径：{abs_input_path}
        2. 输出图片路径：{abs_output_path}
        3. 处理要求：{prompt}
        
        请编写Python代码来处理图片，确保处理后的图片保存为output.png。
        示例代码格式：
        import cv2
        import numpy as np
        
        # 读取图片
        img = cv2.imread('{abs_input_path}')
        # 在这里添加处理逻辑
        # 保存图片
        cv2.imwrite('{abs_output_path}', img)
        """

        # 处理图片
        result = process_image(full_prompt)
        print(f"处理结果: {result}")

        # 检查输出文件是否存在
        if os.path.exists(output_path):
            # 读取处理后的图片并返回
            return cv2.imread(output_path)
        else:
            return None

    except Exception as e:
        print(f"文件处理失败: {str(e)}")
        return None


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
    
.fillable.svelte-a3xscf.app {
    padding: 5px !important;
    width:100%;
    margin: 5px ;
}
.gradio-container .gradio-container-5-23-1 .svelte-a3xscf{
    width:100%;
}
.fillable.svelte-a3xscf.svelte-a3xscf:not(.fill_width){
        max-width: 99%;
    }
"""

# Gradio 界面
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=custom_color)) as iface:
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图片")
            prompt = gr.Textbox(label="处理提示")
            submit_btn = gr.Button("处理图片", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="处理后的图片")
    
    submit_btn.click(
        fn=handle_image_upload,
        inputs=[input_image, prompt],
        outputs=output_image
    )

if __name__ == "__main__":
    iface.launch(server_port=7865, share=True)
