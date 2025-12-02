import json
import os

def compare_labels_and_json():
    # 读取JSON文件
    with open('labels_merged.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 从JSON中提取所有ship文件名
    json_files = set()
    for image in json_data.get('images', []):
        file_name = image.get('file_name', '')
        if file_name.startswith('ship_') and file_name.endswith('.jpg'):
            # 转换为对应的txt文件名
            txt_name = file_name.replace('.jpg', '.txt')
            json_files.add(txt_name)
    
    print(f"JSON文件中记录的ship文件数量: {len(json_files)}")
    
    # 读取labels目录中的文件
    labels_dir = 'raw_private_ship/labels'
    actual_files = set()
    if os.path.exists(labels_dir):
        for file in os.listdir(labels_dir):
            if file.startswith('ship_') and file.endswith('.txt'):
                actual_files.add(file)
    
    print(f"labels目录中实际的ship文件数量: {len(actual_files)}")
    
    # 找出差异
    missing_in_labels = json_files - actual_files  # JSON中有但labels目录中没有
    extra_in_labels = actual_files - json_files    # labels目录中有但JSON中没有
    
    print(f"\n=== 分析结果 ===")
    print(f"JSON中有但labels目录中没有的文件: {len(missing_in_labels)}个")
    if missing_in_labels:
        print("缺失的文件:")
        for file in sorted(missing_in_labels)[:10]:  # 显示前10个
            print(f"  - {file}")
        if len(missing_in_labels) > 10:
            print(f"  ... 还有{len(missing_in_labels) - 10}个文件")
    
    print(f"\nlabels目录中有但JSON中没有的文件: {len(extra_in_labels)}个")
    if extra_in_labels:
        print("额外的文件:")
        for file in sorted(extra_in_labels)[:10]:  # 显示前10个
            print(f"  - {file}")
        if len(extra_in_labels) > 10:
            print(f"  ... 还有{len(extra_in_labels) - 10}个文件")
    
    # 对比具体的文件名差异
    print(f"\n=== 详细对比 ===")
    
    # 获取数字编号进行对比
    json_numbers = set()
    for file in json_files:
        # 从 ship_XXXXXX.txt 提取数字
        import re
        match = re.search(r'ship_(\d+)\.txt', file)
        if match:
            json_numbers.add(int(match.group(1)))
    
    labels_numbers = set()
    for file in actual_files:
        # 从 ship_XXXXXX.txt 提取数字
        import re
        match = re.search(r'ship_(\d+)\.txt', file)
        if match:
            labels_numbers.add(int(match.group(1)))
    
    missing_numbers = json_numbers - labels_numbers
    extra_numbers = labels_numbers - json_numbers
    
    print(f"JSON中有但labels目录中缺失的编号: {len(missing_numbers)}个")
    if missing_numbers:
        print("缺失的编号:")
        for num in sorted(list(missing_numbers))[:20]:
            print(f"  - ship_{num:06d}.txt")
        if len(missing_numbers) > 20:
            print(f"  ... 还有{len(missing_numbers) - 20}个编号")
    
    print(f"\nlabels目录中有但JSON中没有的编号: {len(extra_numbers)}个")
    if extra_numbers:
        print("额外的编号:")
        for num in sorted(list(extra_numbers))[:20]:
            print(f"  - ship_{num:06d}.txt")
        if len(extra_numbers) > 20:
            print(f"  ... 还有{len(extra_numbers) - 20}个编号")

if __name__ == "__main__":
    compare_labels_and_json()