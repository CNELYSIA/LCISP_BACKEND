import io
import zipfile
import os
import json


def createZip(target_dir, additional_file_paths, json_data, zip_filename='config.json'):
    """
    在内存中创建一个ZIP文件，包含目标目录下的所有文件、额外指定的文件列表中的文件和JSON数据。

    :param target_dir: 目标目录路径，该目录下的所有文件都将被添加到ZIP文件中
    :param additional_file_paths: 额外文件路径列表，这些文件也将被添加到ZIP文件中
    :param json_data: 要添加到ZIP文件中的JSON数据
    :param zip_filename: ZIP文件内的JSON文件名
    :return: 包含ZIP文件数据的BytesIO对象
    """
    # 创建一个内存中的字节流对象
    in_memory_zip = io.BytesIO()

    # 将JSON数据转换为字符串，并设置ensure_ascii=False以保证中文字符能够被正确处理
    json_str = json.dumps(json_data, ensure_ascii=False)

    # 创建一个新的ZIP文件，写入到内存中的字节流
    with zipfile.ZipFile(in_memory_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 将JSON数据作为一个字符串写入ZIP文件中的一个新文件
        zipf.writestr(zip_filename, json_str)  # writestr默认使用utf-8编码

        # 遍历目标目录下的所有文件并添加到ZIP文件
        for foldername, subfolders, filenames in os.walk(target_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, start=target_dir)
                zipf.write(file_path, arcname)

        # 添加额外的文件列表
        if isinstance(additional_file_paths, list):
            for file_path in additional_file_paths:
                if os.path.exists(file_path):
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)

    # 回到字节流的开头
    in_memory_zip.seek(0)
    return in_memory_zip


# # 使用示例
# json_data = {
#     "ExtractArgs": {
#         "FileName": "20250203131440209665.jpg",
#         "ModelName": "LangSAM",
#         "Extraction": "道路",
#         "Weights": None
#     }
# }
# target_directory = './assets'
# additional_files = ['main.py', 'script2.py']  # 示例：多个额外文件路径
#
# zip_stream = createZip(target_directory, additional_files, json_data)
#
# # 你可以进一步处理这个zip_stream，例如保存到磁盘或发送到网络
# with open('./ceshi.zip', 'wb') as f:
#     f.write(zip_stream.getbuffer())