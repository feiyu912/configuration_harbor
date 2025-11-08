#!/usr/bin/env python3
"""
港口目标检测数据集下载脚本
支持从多个来源下载完整数据集
"""

import os
import gdown
import zipfile
import requests
from pathlib import Path

def download_from_google_drive():
    """从Google Drive下载数据集"""
    print("正在准备Google Drive下载链接...")
    
    # 这里替换为你的Google Drive分享链接
    dataset_links = {
        "raw_public": "YOUR_GOOGLE_DRIVE_LINK",
        "raw_private": "YOUR_GOOGLE_DRIVE_LINK",
        "dataset_yolo_mixed": "YOUR_GOOGLE_DRIVE_LINK"
    }
    
    for dataset_name, drive_link in dataset_links.items():
        if drive_link and drive_link != "YOUR_GOOGLE_DRIVE_LINK":
            output_path = f"{dataset_name}.zip"
            print(f"正在下载 {dataset_name}...")
            try:
                gdown.download(drive_link, output_path, quiet=False)
                
                # 解压文件
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
                
                # 清理压缩包
                os.remove(output_path)
                print(f"✅ {dataset_name} 下载完成")
                
            except Exception as e:
                print(f"❌ 下载 {dataset_name} 失败: {e}")

def download_from_onedrive():
    """从OneDrive下载数据集"""
    print("正在准备OneDrive下载...")
    
    # OneDrive直链下载示例
    onedrive_links = {
        "port_detection_dataset": "YOUR_ONEDRIVE_DIRECT_LINK"
    }
    
    for name, link in onedrive_links.items():
        if link and link != "YOUR_ONEDRIVE_DIRECT_LINK":
            print(f"正在从OneDrive下载 {name}...")
            try:
                response = requests.get(link, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                output_file = f"{name}.zip"
                with open(output_file, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"下载进度: {progress:.1f}%", end='\r')
                
                print(f"\n✅ {name} 下载完成")
                
            except Exception as e:
                print(f"❌ OneDrive下载失败: {e}")

def setup_project_data():
    """设置项目数据结构"""
    print("正在设置项目数据结构...")
    
    # 创建必要的目录结构
    directories = [
        "raw_public/images",
        "raw_public/labels", 
        "raw_private/images",
        "raw_private/labels",
        "raw_private_ship/images",
        "raw_private_ship/labels",
        "raw_private_container/images", 
        "raw_private_container/labels",
        "raw_private_crane/images",
        "raw_private_crane/labels",
        "dataset_yolo_public/images/train",
        "dataset_yolo_public/images/val",
        "dataset_yolo_public/labels/train",
        "dataset_yolo_public/labels/val",
        "dataset_yolo_private/images/train",
        "dataset_yolo_private/images/val", 
        "dataset_yolo_private/labels/train",
        "dataset_yolo_private/labels/val",
        "dataset_yolo_mixed_test/images/test",
        "dataset_yolo_mixed_test/labels/test"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ 项目数据结构设置完成")

def main():
    """主函数"""
    print("🚀 港口目标检测数据集下载工具")
    print("=" * 50)
    
    # 设置项目结构
    setup_project_data()
    
    print("\n📥 请选择数据来源:")
    print("1. Google Drive")
    print("2. OneDrive") 
    print("3. 手动下载 (推荐)")
    print("4. 使用示例数据")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        download_from_google_drive()
    elif choice == "2":
        download_from_onedrive()
    elif choice == "3":
        print("\n📋 手动下载步骤:")
        print("1. 准备数据集压缩包")
        print("2. 解压到项目根目录")
        print("3. 确保目录结构正确")
        print("4. 运行 python data_download_scripts/verify_data.py 验证")
    elif choice == "4":
        print("\n✅ 使用示例数据模式")
        print("项目已设置为示例数据模式，可以运行Streamlit应用查看效果")
        print("访问: http://localhost:8503")
    else:
        print("❌ 无效选择，使用示例数据模式")
    
    print("\n🎉 数据集准备完成!")
    print("下一步: 运行 'streamlit run app/streamlit_app.py' 启动展示系统")

if __name__ == "__main__":
    main()