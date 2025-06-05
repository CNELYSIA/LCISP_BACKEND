import os
import json
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
import uuid
from TaskChainBuilder import TaskChainBuilder, TaskChain, TaskNode

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建任务链构建器
builder = TaskChainBuilder()

# 存储当前任务链
current_chain = None

# 更新函数描述，说明saveImg的新行为
builder.function_descriptions["saveImg"] = "保存图像到指定路径（可选，默认保存到桌面的result文件夹），自动选择处理流程中的最新数据源"

def build_chain(requirement):
    """根据需求构建任务链"""
    global current_chain
    try:
        current_chain = builder.build_chain_from_requirement(requirement)
        return f"任务链构建成功！\n\n任务链名称: {current_chain.name}\n任务链描述: {current_chain.description}\n\n节点数量: {len(current_chain.nodes)}"
    except Exception as e:
        return f"构建任务链失败: {str(e)}"

def save_chain(file_path):
    """保存任务链到文件"""
    global current_chain
    if current_chain is None:
        return "没有可保存的任务链"
    
    try:
        # 确保文件路径有.json后缀
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        # 如果没有指定路径，保存到task_chains目录
        if not os.path.dirname(file_path):
            os.makedirs("task_chains", exist_ok=True)
            file_path = os.path.join("task_chains", file_path)
        
        current_chain.save_to_file(file_path)
        return f"任务链已保存到 {file_path}"
    except Exception as e:
        return f"保存任务链失败: {str(e)}"

def load_chain(file_path):
    """从文件加载任务链"""
    global current_chain
    try:
        # 确保文件路径有.json后缀
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        # 如果没有指定路径，从task_chains目录加载
        if not os.path.dirname(file_path):
            file_path = os.path.join("task_chains", file_path)
        
        # 确保文件存在
        if not os.path.exists(file_path):
            return f"文件 {file_path} 不存在"
        
        current_chain = TaskChain.load_from_file(file_path)
        return f"任务链加载成功！\n\n任务链名称: {current_chain.name}\n任务链描述: {current_chain.description}\n\n节点数量: {len(current_chain.nodes)}"
    except Exception as e:
        return f"加载任务链失败: {str(e)}"

def upload_data(data_files):
    """上传数据文件"""
    global current_chain
    if current_chain is None:
        return "没有可上传数据的任务链，请先构建或加载任务链"
    
    try:
        file_paths = []
        
        # 处理上传的文件
        for file in data_files:
            if file is not None:
                file_path = builder.upload_data(file.name, current_chain)
                file_paths.append(file_path)
        
        if not file_paths:
            return "未上传任何文件"
        
        return f"数据上传成功，已上传到以下路径：\n{', '.join(file_paths)}\n\n数据路径已更新到任务链中"
    except Exception as e:
        return f"数据上传失败: {str(e)}"

def execute_chain():
    """执行当前任务链"""
    global current_chain
    if current_chain is None:
        return "没有可执行的任务链"
    
    try:
        # 检查是否已上传数据
        if not current_chain.data_path:
            # 检查读取节点是否有有效路径
            has_valid_path = False
            for node in current_chain.nodes.values():
                if node.function_name == "readImg":
                    if "strFilePath" in node.parameters:
                        path = node.parameters["strFilePath"]
                        if os.path.exists(path):
                            has_valid_path = True
                            break
            
            if not has_valid_path:
                return "请先上传数据文件"
        
        results = builder.execute_chain(current_chain)
        return f"任务链执行完成！\n\n执行日志:\n- 开始执行任务链 {current_chain.name}\n- 共执行节点数: {len(current_chain.nodes)}\n- 执行完成"
    except Exception as e:
        return f"执行任务链失败: {str(e)}"

def get_chain_info():
    """获取当前任务链信息"""
    global current_chain
    if current_chain is None:
        return "没有任务链", [], [], None
    
    # 构建节点信息
    nodes_info = []
    for node_id, node in current_chain.nodes.items():
        nodes_info.append({
            "id": node_id,
            "name": node.task_name,
            "function": node.function_name,
            "description": node.description,
            "parameters": json.dumps(node.parameters, ensure_ascii=False, indent=2)
        })
    
    # 构建边信息
    edges_info = []
    for node_id, node in current_chain.nodes.items():
        for next_node_id in node.next_nodes:
            edges_info.append({
                "from": node_id,
                "to": next_node_id
            })
    
    # 构建任务链JSON
    chain_json = {
        "chain_id": current_chain.chain_id,
        "name": current_chain.name,
        "description": current_chain.description,
        "data_path": current_chain.data_path,
        "start_nodes": current_chain.start_nodes,
        "nodes": {}
    }
    
    # 添加节点信息
    for node_id, node in current_chain.nodes.items():
        chain_json["nodes"][node_id] = {
            "task_name": node.task_name,
            "function_name": node.function_name,
            "parameters": node.parameters,
            "description": node.description,
            "next_nodes": node.next_nodes
        }
    
    # 将JSON转换为格式化的字符串
    chain_info_text = json.dumps(chain_json, ensure_ascii=False, indent=2)
    
    # 生成图形化展示
    graph_image = generate_chain_graph()
    
    return chain_info_text, nodes_info, edges_info, graph_image

def update_chain_from_json(chain_json):
    """从JSON更新任务链"""
    global current_chain
    try:
        # 解析JSON
        chain_data = json.loads(chain_json)
        
        # 创建新的任务链
        new_chain = TaskChain(
            chain_id=chain_data.get("chain_id", str(uuid.uuid4())),
            name=chain_data.get("name", "未命名任务链"),
            description=chain_data.get("description", "")
        )
        
        # 添加节点
        for node_id, node_data in chain_data.get("nodes", {}).items():
            node = TaskNode(
                node_id=node_id,
                task_name=node_data.get("task_name", ""),
                function_name=node_data.get("function_name", ""),
                parameters=node_data.get("parameters", {}),
                description=node_data.get("description", "")
            )
            node.next_nodes = node_data.get("next_nodes", [])
            new_chain.add_node(node)
        
        # 设置起始节点
        new_chain.start_nodes = chain_data.get("start_nodes", [])
        
        # 设置数据路径
        if "data_path" in chain_data:
            new_chain.data_path = chain_data["data_path"]
        
        # 更新当前任务链
        current_chain = new_chain
        
        return "任务链更新成功！"
    except Exception as e:
        return f"更新任务链失败: {str(e)}"

def generate_chain_graph():
    """生成任务链的图形化展示"""
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for node_id, node in current_chain.nodes.items():
        # 获取函数描述
        function_desc = builder.function_descriptions.get(node.function_name, "")
        # 节点标签只包含ID
        G.add_node(node_id, label=f"{node_id}")
    
    # 添加边
    for node_id, node in current_chain.nodes.items():
        for next_id in node.next_nodes:
            G.add_edge(node_id, next_id)
    
    # 设置布局
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    # 添加节点ID标签（增大字体）
    nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold')
    
    # 添加方法名和描述标签
    method_labels = {}
    for node_id, node in current_chain.nodes.items():
        # 获取节点位置
        x, y = pos[node_id]
        # 在节点下方添加方法名和描述
        method_labels[node_id] = (x, y - 0.15)  # 向下偏移更多，为描述留出空间
    
    # 绘制方法名和描述标签
    for node_id, (x, y) in method_labels.items():
        node = current_chain.nodes[node_id]
        # 获取函数描述
        function_desc = builder.function_descriptions.get(node.function_name, "")
        
        # 绘制方法名
        plt.text(x, y, node.function_name, 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=18, 
                fontweight='bold',
                color='darkblue',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # 在方法名下方绘制描述（最多显示30个字符，超出部分用...替代）
        desc = function_desc[:30] + "..." if len(function_desc) > 30 else function_desc
        plt.text(x, y - 0.1, desc,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12,
                color='gray',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # 移除坐标轴
    plt.axis('off')
    
    # 保存图形到临时文件
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 清理旧的图形文件
    for file in os.listdir(temp_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(temp_dir, file))
    
    # 保存新的图形文件
    file_path = os.path.join(temp_dir, "chain_graph.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return file_path

def get_available_functions():
    """获取可用函数列表"""
    return list(builder.function_descriptions.keys())

def get_function_description(function_name):
    """获取函数描述"""
    if function_name in builder.function_descriptions:
        return builder.function_descriptions[function_name]
    return "无描述"
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
# 创建Gradio界面
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=custom_color)) as iface:
    
    with gr.Row():
        with gr.Column(scale=1):
            chain_info = gr.Textbox(label="任务链JSON", lines=15)
            update_btn = gr.Button("更新任务链", variant="primary")
        with gr.Column(scale=1):
            chain_graph = gr.Image(label="任务链图形化展示", type="filepath", height=400)
    
    # 绑定更新按钮
    update_btn.click(update_chain_from_json, inputs=chain_info, outputs=gr.Textbox(label="更新结果")).then(
        get_chain_info,
        outputs=[chain_info, gr.State([]), gr.State([]), chain_graph]
    )
    
    with gr.Tab("构建任务链"):
        with gr.Row():
            with gr.Column():
                requirement_input = gr.Textbox(label="处理需求", placeholder="请输入处理需求，例如：读取一个TIF图像，进行裁剪，然后拼接回原图，最后保存为JPG格式")
                build_btn = gr.Button("构建任务链", variant="primary")
            with gr.Column():
                build_output = gr.Textbox(label="构建结果", lines=5)
        
        build_btn.click(build_chain, inputs=requirement_input, outputs=build_output).then(
            get_chain_info,
            outputs=[chain_info, gr.State([]), gr.State([]), chain_graph]
        )
    
    with gr.Tab("数据管理"):
        with gr.Row():
            with gr.Column():
                data_files = gr.File(label="上传数据文件", file_count="multiple")
                upload_btn = gr.Button("上传到任务链", variant="primary")
            with gr.Column():
                upload_output = gr.Textbox(label="上传结果", lines=5)
        
        upload_btn.click(upload_data, inputs=data_files, outputs=upload_output)
    
    with gr.Tab("保存/加载任务链"):
        with gr.Row():
            with gr.Column():
                file_path_input = gr.Textbox(label="文件名", placeholder="请输入文件名，例如：task_chain.json")
                save_btn = gr.Button("保存任务链", variant="primary")
                load_btn = gr.Button("加载任务链")
            with gr.Column():
                file_output = gr.Textbox(label="操作结果", lines=5)
        
        save_btn.click(save_chain, inputs=file_path_input, outputs=file_output)
        load_btn.click(load_chain, inputs=file_path_input, outputs=file_output).then(
            get_chain_info,
            outputs=[chain_info, gr.State([]), gr.State([]), chain_graph]
        )
    
    with gr.Tab("执行任务链"):
        with gr.Row():
            execute_btn = gr.Button("执行任务链", variant="primary")
            execute_output = gr.Textbox(label="执行结果", lines=10)
        
        execute_btn.click(execute_chain, outputs=execute_output)

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("task_chains", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # 清理临时文件
    for file in os.listdir("temp"):
        if file.endswith(".png"):
            os.remove(os.path.join("temp", file))
    
    # 启动界面
    iface.launch(server_port=7870, share=True) 