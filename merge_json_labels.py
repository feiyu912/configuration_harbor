#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并多个COCO格式的JSON标签文件
"""

import json
import os

def merge_multiple_coco_json(json_files, output_file):
    """
    合并多个COCO格式的JSON文件
    """
    if not json_files:
        raise ValueError("至少需要提供一个JSON文件")
    
    # 读取第一个JSON文件作为基础
    with open(json_files[0], 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    # 创建合并后的数据
    merged_data = {
        "info": base_data.get("info", {"description": "merged dataset"}),
        "images": [],
        "annotations": [],
        "categories": base_data.get("categories", [])
    }
    
    # 初始化最大ID
    max_image_id = 0
    max_annotation_id = 0
    
    # 处理所有JSON文件
    for file_idx, json_file in enumerate(json_files):
        print(f"处理文件 {file_idx + 1}/{len(json_files)}: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是第一个文件，直接添加
        if file_idx == 0:
            # 添加images
            for img in data.get("images", []):
                merged_data["images"].append(img)
                if img["id"] > max_image_id:
                    max_image_id = img["id"]
            
            # 添加annotations
            for ann in data.get("annotations", []):
                merged_data["annotations"].append(ann)
                if ann["id"] > max_annotation_id:
                    max_annotation_id = ann["id"]
        else:
            # 对于后续文件，需要重新编号
            image_id_map = {}  # 原始ID到新ID的映射
            
            # 处理images，重新编号
            for img in data.get("images", []):
                old_id = img["id"]
                new_id = max_image_id + 1
                image_id_map[old_id] = new_id
                
                new_img = img.copy()
                new_img["id"] = new_id
                merged_data["images"].append(new_img)
                max_image_id = new_id
            
            # 处理annotations，重新编号并更新image_id
            for ann in data.get("annotations", []):
                old_image_id = ann["image_id"]
                if old_image_id in image_id_map:
                    new_image_id = image_id_map[old_image_id]
                    
                    new_ann = ann.copy()
                    new_ann["id"] = max_annotation_id + 1
                    new_ann["image_id"] = new_image_id
                    merged_data["annotations"].append(new_ann)
                    max_annotation_id += 1
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    return merged_data

def main():
    # 定义要合并的四个JSON文件路径
    json_files = [
        "extra.json",
        "ship_annotations.json", 
        "crane_annotation.json",
        "container_annotations(1).json"
    ]
    output_file = "merged_annotations.json"
    
    # 检查所有文件是否存在
    missing_files = []
    for json_file in json_files:
        if not os.path.exists(json_file):
            missing_files.append(json_file)
    
    if missing_files:
        print(f"错误: 以下文件不存在:")
        for missing in missing_files:
            print(f"  - {missing}")
        return
    
    print("开始合并JSON文件...")
    print(f"输入文件: {len(json_files)} 个")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {file}")
    print(f"输出文件: {output_file}")
    
    try:
        # 合并JSON文件
        merged_data = merge_multiple_coco_json(json_files, output_file)
        
        # 打印统计信息
        print("\n合并完成!")
        print(f"总图片数: {len(merged_data['images'])}")
        print(f"总标注数: {len(merged_data['annotations'])}")
        print(f"类别数: {len(merged_data['categories'])}")
        
    except Exception as e:
        print(f"合并过程中出现错误: {e}")

if __name__ == "__main__":
    main()