import json
import numpy as np
import torch

def print_json_structure(data, indent='', level=0):
    # 处理字典
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}|-- {key}:")
            print_json_structure(value, indent + "    ", level + 1)
    
    # 处理列表
    elif isinstance(data, list):
        print(f"{indent}|-- List of length {len(data)}")
        if len(data) > 0:
            print(f"{indent}    |-- Example element:")
            print_json_structure(data[0], indent + "    ", level + 1)
    
    # 处理 NumPy 数组
    elif isinstance(data, np.ndarray):
        print(f"{indent}|-- np.ndarray with shape {data.shape}")
    
    # 处理 PyTorch 张量
    elif isinstance(data, torch.Tensor):
        print(f"{indent}|-- torch.Tensor with shape {data.shape}")
    
    # 处理其他类型
    else:
        print(f"{indent}|-- {type(data).__name__}: {data}")

# 示例用法
if __name__ == "__main__":
    # 假设你的JSON结构如下
    json_file = '/data1/liangzhijia//datasets/coco/annotations/instances_img10000_novel17_train2017.json'
    data = json.load(open(json_file, 'r'))

    # 打印结构
    print_json_structure(data)