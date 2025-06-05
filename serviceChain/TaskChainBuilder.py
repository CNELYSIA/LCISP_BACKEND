import os
import json
import uuid
import shutil
from typing import List, Dict, Any, Optional, Callable
from qwen_agent.agents import Assistant
from geoUtiles import ImageManager

class TaskNode:
    """任务节点类，表示任务链中的一个节点"""
    def __init__(self, node_id: str, task_name: str, function_name: str, 
                 parameters: Dict[str, Any], description: str):
        self.node_id = node_id
        self.task_name = task_name
        self.function_name = function_name
        self.parameters = parameters
        self.description = description
        self.next_nodes = []  # 存储下一个节点的ID列表
        
    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典格式"""
        return {
            "node_id": self.node_id,
            "task_name": self.task_name,
            "function_name": self.function_name,
            "parameters": self.parameters,
            "description": self.description,
            "next_nodes": self.next_nodes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskNode':
        """从字典创建节点"""
        node = cls(
            node_id=data["node_id"],
            task_name=data["task_name"],
            function_name=data["function_name"],
            parameters=data["parameters"],
            description=data["description"]
        )
        node.next_nodes = data["next_nodes"]
        return node
    
    def add_next_node(self, node_id: str) -> None:
        """添加下一个节点"""
        if node_id not in self.next_nodes:
            self.next_nodes.append(node_id)
    
    def remove_next_node(self, node_id: str) -> None:
        """移除下一个节点"""
        if node_id in self.next_nodes:
            self.next_nodes.remove(node_id)
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """更新节点参数"""
        self.parameters.update(parameters)
    
    def update_description(self, description: str) -> None:
        """更新节点描述"""
        self.description = description


class TaskChain:
    """任务链类，表示一组有序的任务节点"""
    def __init__(self, chain_id: str, name: str, description: str):
        self.chain_id = chain_id
        self.name = name
        self.description = description
        self.nodes: Dict[str, TaskNode] = {}  # 存储所有节点，键为节点ID
        self.start_nodes: List[str] = []  # 存储起始节点ID列表
        self.data_path: str = ""  # 存储数据路径
        
    def add_node(self, node: TaskNode) -> None:
        """添加节点到任务链"""
        self.nodes[node.node_id] = node
        
    def remove_node(self, node_id: str) -> None:
        """从任务链中移除节点"""
        if node_id in self.nodes:
            # 从所有节点的next_nodes中移除该节点
            for node in self.nodes.values():
                if node_id in node.next_nodes:
                    node.next_nodes.remove(node_id)
            
            # 从start_nodes中移除该节点
            if node_id in self.start_nodes:
                self.start_nodes.remove(node_id)
                
            # 删除节点
            del self.nodes[node_id]
    
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """添加节点之间的连接"""
        if from_node_id in self.nodes and to_node_id in self.nodes:
            self.nodes[from_node_id].add_next_node(to_node_id)
    
    def remove_edge(self, from_node_id: str, to_node_id: str) -> None:
        """移除节点之间的连接"""
        if from_node_id in self.nodes:
            self.nodes[from_node_id].remove_next_node(to_node_id)
    
    def get_node(self, node_id: str) -> Optional[TaskNode]:
        """获取指定ID的节点"""
        return self.nodes.get(node_id)
    
    def set_data_path(self, data_path: str) -> None:
        """设置数据路径"""
        self.data_path = data_path
        
        # 更新读取图像节点的参数
        for node in self.nodes.values():
            if node.function_name == "readImg":
                if "strFilePath" in node.parameters:
                    # 如果数据路径是目录，直接使用
                    if os.path.isdir(data_path):
                        node.parameters["strFilePath"] = data_path
                    # 如果数据路径是文件，使用文件
                    elif os.path.isfile(data_path):
                        node.parameters["strFilePath"] = data_path
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务链转换为字典格式"""
        return {
            "chain_id": self.chain_id,
            "name": self.name,
            "description": self.description,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "start_nodes": self.start_nodes,
            "data_path": self.data_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskChain':
        """从字典创建任务链"""
        chain = cls(
            chain_id=data["chain_id"],
            name=data["name"],
            description=data["description"]
        )
        
        # 添加节点
        for node_id, node_data in data["nodes"].items():
            node = TaskNode.from_dict(node_data)
            chain.add_node(node)
            
        # 设置起始节点
        chain.start_nodes = data["start_nodes"]
        
        # 设置数据路径
        if "data_path" in data:
            chain.data_path = data["data_path"]
        
        return chain
    
    def save_to_file(self, file_path: str) -> None:
        """将任务链保存到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'TaskChain':
        """从文件加载任务链"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class TaskChainBuilder:
    """任务链构建器，使用大模型自动构建任务链"""
    def __init__(self):
        # 设置环境变量
        os.environ["DASHSCOPE_API_KEY"] = "sk-5d8c49c360074c78ac5f04c99285575b"
        
        # 初始化 Qwen-Agent 的 Assistant 实例
        self.assistant = Assistant(
            llm={'model': 'qwen-max'},
            function_list=['code_interpreter'],
            system_message="你是一个地理空间数据处理专家。用户会提供处理需求，你需要设计一个任务链来完成这些需求。"
        )
        
        # 初始化图像管理器
        self.image_manager = ImageManager()
        
        # 可用函数映射
        self.available_functions = {
            "readImg": self.image_manager.readImg,
            "saveImg": self.image_manager.saveImg,
            "cropImg": self.image_manager.cropImg,
            "stitchImg": self.image_manager.stitchImg,
            "truncatedLinearStretch": self.image_manager.truncatedLinearStretch,
            "assignGeoreference": self.image_manager.assignGeoreference,
            "tif2shp": self.image_manager.tif2shp,
            "appendImagesFrom": self.image_manager.appendImagesFrom
        }
        
        # 函数描述
        self.function_descriptions = {
            "readImg": "读取图像文件或目录中的图像",
            "saveImg": "保存图像到指定路径（自动选择处理流程中的数据源，如裁剪、拼接、拉伸等）",
            "cropImg": "裁剪图像为小块",
            "stitchImg": "将裁剪的小块图像拼接回原图",
            "truncatedLinearStretch": "对图像进行线性拉伸处理",
            "assignGeoreference": "为图像赋予地理参考信息",
            "tif2shp": "将TIF影像转换为Shapefile格式",
            "appendImagesFrom": "从指定字典追加图像"
        }
        
        # 函数参数映射（确保参数名称正确）
        self.function_parameters = {
            "readImg": ["strFilePath"],
            "saveImg": ["strSavePath", "strOutFormat", "formEE"],
            "cropImg": ["intWidth", "intHeight", "intStep", "intStartGroup"],
            "stitchImg": ["intWidth", "intHeight", "intStep"],
            "truncatedLinearStretch": ["dblPercentile"],
            "assignGeoreference": [],
            "tif2shp": ["output_dir"],
            "appendImagesFrom": ["sourceDictName"]
        }
        
        # 创建存储目录
        os.makedirs("task_chains", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("output", exist_ok=True)
    
    def build_chain_from_requirement(self, requirement: str) -> TaskChain:
        """根据用户需求构建任务链"""
        # 生成唯一ID
        chain_id = str(uuid.uuid4())
        chain_name = f"任务链_{chain_id[:8]}"
        
        # 构建提示
        prompt = f"""
        请根据以下需求设计一个地理空间数据处理任务链：
        
        需求：{requirement}
        
        可用的函数有：
        {json.dumps(self.function_descriptions, ensure_ascii=False, indent=2)}
        
        函数参数说明：
        {json.dumps(self.function_parameters, ensure_ascii=False, indent=2)}
        
        请设计一个任务链，包括以下内容：
        1. 每个任务节点的名称、使用的函数、参数和描述
        2. 任务节点之间的连接关系
        
        请以JSON格式返回，格式如下：
        {{
            "chain_name": "任务链名称",
            "chain_description": "任务链描述",
            "nodes": [
                {{
                    "node_id": "唯一ID",
                    "task_name": "任务名称",
                    "function_name": "函数名称",
                    "parameters": {{"参数名": "参数值"}},
                    "description": "任务描述"
                }},
                ...
            ],
            "edges": [
                {{"from": "源节点ID", "to": "目标节点ID"}},
                ...
            ]
        }}
        
        注意：
        1. 参数名称必须与函数参数说明中的名称完全匹配。
        2. 对于saveImg函数，strOutFormat参数必须以点开头，例如：'.jpg'、'.tif'，不要写成'jpg'或'tif'。
        3. saveImg函数支持的文件格式有：'.jpg'、'.jpeg'、'.png'、'.bmp'、'.tif'、'.tiff'
        4. saveImg函数的strSavePath参数是可选的，如果不提供则默认保存到桌面的result文件夹。
        """
        
        # 获取大模型响应
        messages = [{'role': 'user', 'content': prompt}]
        chain_data = None
        
        for response in self.assistant.run(messages):
            if isinstance(response, list):
                for resp in response:
                    if isinstance(resp, dict) and 'content' in resp:
                        try:
                            # 尝试从响应中提取JSON
                            content = resp['content']
                            # 查找JSON开始和结束位置
                            start_idx = content.find('{')
                            end_idx = content.rfind('}') + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = content[start_idx:end_idx]
                                chain_data = json.loads(json_str)
                                break
                        except json.JSONDecodeError:
                            print("JSON解析错误")
        
        # 如果无法从响应中提取JSON，使用默认任务链
        if not chain_data:
            print("无法从大模型响应中提取任务链数据，使用默认任务链")
            chain_data = {
                "chain_name": chain_name,
                "chain_description": "默认任务链",
                "nodes": [
                    {
                        "node_id": "1",
                        "task_name": "读取图像",
                        "function_name": "readImg",
                        "parameters": {"strFilePath": "data/input.tif"},
                        "description": "读取输入图像"
                    },
                    {
                        "node_id": "2",
                        "task_name": "保存图像",
                        "function_name": "saveImg",
                        "parameters": {"strOutFormat": ".tif"},
                        "description": "保存处理后的图像到桌面的result文件夹"
                    }
                ],
                "edges": [
                    {"from": "1", "to": "2"}
                ]
            }
        
        # 创建任务链
        chain = TaskChain(chain_id, chain_data["chain_name"], chain_data["chain_description"])
        
        # 添加节点
        for node_data in chain_data["nodes"]:
            # 确保参数名称正确
            function_name = node_data["function_name"]
            parameters = node_data["parameters"]
            
            # 检查参数名称是否与函数参数匹配
            if function_name in self.function_parameters:
                expected_params = self.function_parameters[function_name]
                corrected_params = {}
                
                # 处理特殊参数名称
                if function_name == "readImg" and "path" in parameters:
                    corrected_params["strFilePath"] = parameters["path"]
                elif function_name == "saveImg" and "path" in parameters:
                    corrected_params["strSavePath"] = parameters["path"]
                elif function_name == "saveImg" and "format" in parameters:
                    # 确保格式以点开头
                    format_value = parameters["format"]
                    if format_value and not format_value.startswith('.'):
                        format_value = '.' + format_value
                    corrected_params["strOutFormat"] = format_value
                else:
                    # 保留原始参数
                    corrected_params = parameters
                
                # 如果是saveImg函数，确保strOutFormat参数以点开头
                if function_name == "saveImg" and "strOutFormat" in corrected_params:
                    format_value = corrected_params["strOutFormat"]
                    if format_value and not str(format_value).startswith('.'):
                        corrected_params["strOutFormat"] = '.' + format_value
                
                # 更新参数
                node_data["parameters"] = corrected_params
            
            node = TaskNode(
                node_id=node_data["node_id"],
                task_name=node_data["task_name"],
                function_name=node_data["function_name"],
                parameters=node_data["parameters"],
                description=node_data["description"]
            )
            chain.add_node(node)
        
        # 添加边
        for edge in chain_data["edges"]:
            chain.add_edge(edge["from"], edge["to"])
        
        # 确保所有没有前驱的节点都被添加为起始节点
        for node_id, node in chain.nodes.items():
            if not any(node_id in n.next_nodes for n in chain.nodes.values()):
                chain.start_nodes.append(node_id)
        
        return chain
    
    def execute_chain(self, chain: TaskChain) -> Dict[str, Any]:
        """执行任务链"""
        results = {}
        visited = set()
        
        def execute_node(node_id: str) -> Any:
            """递归执行节点及其后续节点"""
            if node_id in visited:
                return results.get(node_id)
            
            visited.add(node_id)
            node = chain.get_node(node_id)
            
            if not node:
                return None
            
            # 获取函数
            func = self.available_functions.get(node.function_name)
            if not func:
                print(f"函数 {node.function_name} 不存在")
                return None
            
            # 执行函数
            try:
                print(f"执行节点 {node.task_name}...")
                result = func(**node.parameters)
                results[node_id] = result
                print(f"节点 {node.task_name} 执行完成")
                
                # 执行后续节点
                for next_node_id in node.next_nodes:
                    execute_node(next_node_id)
                
                return result
            except Exception as e:
                print(f"执行节点 {node_id} 时出错: {str(e)}")
                return None
        
        # 从所有起始节点开始执行
        for start_node_id in chain.start_nodes:
            execute_node(start_node_id)
        
        return results
    
    def edit_node(self, chain: TaskChain, node_id: str, 
                 task_name: Optional[str] = None, 
                 function_name: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None) -> TaskNode:
        """编辑任务链中的节点"""
        node = chain.get_node(node_id)
        if not node:
            raise ValueError(f"节点 {node_id} 不存在")
        
        if task_name:
            node.task_name = task_name
        
        if function_name:
            if function_name not in self.available_functions:
                raise ValueError(f"函数 {function_name} 不存在")
            node.function_name = function_name
        
        if parameters:
            node.update_parameters(parameters)
        
        if description:
            node.update_description(description)
        
        return node
    
    def add_node_to_chain(self, chain: TaskChain, 
                         task_name: str, 
                         function_name: str,
                         parameters: Dict[str, Any],
                         description: str,
                         after_node_id: Optional[str] = None) -> TaskNode:
        """向任务链添加新节点"""
        if function_name not in self.available_functions:
            raise ValueError(f"函数 {function_name} 不存在")
        
        # 创建新节点
        node_id = str(uuid.uuid4())
        node = TaskNode(node_id, task_name, function_name, parameters, description)
        
        # 添加到任务链
        chain.add_node(node)
        
        # 如果指定了前驱节点，添加连接
        if after_node_id:
            if after_node_id not in chain.nodes:
                raise ValueError(f"节点 {after_node_id} 不存在")
            chain.add_edge(after_node_id, node_id)
        else:
            # 否则添加为起始节点
            chain.start_nodes.append(node_id)
        
        return node
    
    def remove_node_from_chain(self, chain: TaskChain, node_id: str) -> None:
        """从任务链中移除节点"""
        if node_id not in chain.nodes:
            raise ValueError(f"节点 {node_id} 不存在")
        
        chain.remove_node(node_id)
    
    def add_edge_to_chain(self, chain: TaskChain, from_node_id: str, to_node_id: str) -> None:
        """向任务链添加连接"""
        if from_node_id not in chain.nodes:
            raise ValueError(f"源节点 {from_node_id} 不存在")
        
        if to_node_id not in chain.nodes:
            raise ValueError(f"目标节点 {to_node_id} 不存在")
        
        chain.add_edge(from_node_id, to_node_id)
        
        # 如果目标节点是起始节点，将其从起始节点列表中移除
        if to_node_id in chain.start_nodes:
            chain.start_nodes.remove(to_node_id)
    
    def remove_edge_from_chain(self, chain: TaskChain, from_node_id: str, to_node_id: str) -> None:
        """从任务链中移除连接"""
        if from_node_id not in chain.nodes:
            raise ValueError(f"源节点 {from_node_id} 不存在")
        
        chain.remove_edge(from_node_id, to_node_id)
        
        # 如果目标节点没有前驱节点，将其添加为起始节点
        if not any(to_node_id in node.next_nodes for node in chain.nodes.values()):
            chain.start_nodes.append(to_node_id)
    
    def upload_data(self, data_path: str, chain: TaskChain) -> str:
        """上传数据到指定路径，并更新任务链中的数据路径"""
        # 创建数据目录
        os.makedirs("data", exist_ok=True)
        
        # 目标路径
        if os.path.isfile(data_path):
            # 如果是文件，复制到data目录
            filename = os.path.basename(data_path)
            dest_path = os.path.join("data", filename)
            shutil.copy2(data_path, dest_path)
            chain.set_data_path(dest_path)
            return dest_path
        elif os.path.isdir(data_path):
            # 如果是目录，创建子目录并复制内容
            dirname = os.path.basename(data_path)
            dest_path = os.path.join("data", dirname)
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(data_path, dest_path)
            chain.set_data_path(dest_path)
            return dest_path
        else:
            raise ValueError(f"数据路径 {data_path} 不存在或无效")


# 示例用法
if __name__ == "__main__":
    # 创建任务链构建器
    builder = TaskChainBuilder()
    
    # 根据需求构建任务链
    requirement = "读取一个TIF图像，进行裁剪，然后拼接回原图，最后保存为JPG格式"
    chain = builder.build_chain_from_requirement(requirement)
    
    # 保存任务链到文件
    chain.save_to_file("task_chains/task_chain.json")
    
    # 编辑任务链中的节点
    node_id = list(chain.nodes.keys())[0]
    builder.edit_node(chain, node_id, parameters={"strFilePath": "data/custom_input.tif"})
    
    # 添加新节点
    builder.add_node_to_chain(
        chain,
        "图像增强",
        "truncatedLinearStretch",
        {"dblPercentile": 5},
        "对图像进行线性拉伸增强",
        after_node_id=list(chain.nodes.keys())[-1]
    )
    
    # 上传数据
    # builder.upload_data("path/to/data", chain)
    
    # 执行任务链
    # results = builder.execute_chain(chain)
    
    print("任务链执行完成") 