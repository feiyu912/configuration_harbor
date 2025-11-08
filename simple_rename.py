import os

# 目录路径
labels_dir = r"f:\python\configuration_harbor\raw_private_ship\labels"
images_dir = r"f:\python\configuration_harbor\raw_private_ship\images"

# 直接重命名文件函数
def rename_all_chinese_files():
    renamed_count = 0
    
    # 遍历labels目录中的所有文件
    for filename in os.listdir(labels_dir):
        if filename.startswith("船_") and filename.endswith(".txt"):
            # 创建新的英文文件名
            new_filename = "ship_0" + filename[2:]
            old_label_path = os.path.join(labels_dir, filename)
            new_label_path = os.path.join(labels_dir, new_filename)
            
            # 对应的图像文件名
            img_filename = filename.replace(".txt", ".jpg")
            new_img_filename = new_filename.replace(".txt", ".jpg")
            old_img_path = os.path.join(images_dir, img_filename)
            new_img_path = os.path.join(images_dir, new_img_filename)
            
            # 执行重命名，处理可能存在的已存在文件
            try:
                # 重命名标签文件
                if os.path.exists(new_label_path):
                    os.remove(new_label_path)
                os.rename(old_label_path, new_label_path)
                print(f"已重命名标签: {filename} -> {new_filename}")
                
                # 重命名对应的图像文件（如果存在）
                if os.path.exists(old_img_path):
                    if os.path.exists(new_img_path):
                        os.remove(new_img_path)
                    os.rename(old_img_path, new_img_path)
                    print(f"已重命名图像: {img_filename} -> {new_img_filename}")
                
                renamed_count += 1
            except Exception as e:
                print(f"重命名失败 {filename}: {e}")
    
    return renamed_count

# 执行重命名
print("开始重命名中文文件...")
count = rename_all_chinese_files()
print(f"\n重命名完成！成功处理了 {count} 个文件对。")