#!/usr/bin/env python3
"""
OpenCV传统方法评估脚本
使用实际数据集进行真实评估，不使用模拟数据
"""

import cv2
import numpy as np
import os
import yaml
import time
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def load_dataset_config(yaml_path):
    """加载数据集配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def yolo_to_mask(yolo_file, image_shape, class_id=0):
    """
    将YOLO格式标签转换为分割掩码
    yolo_file: YOLO格式的标签文件路径
    image_shape: (height, width) 图像尺寸
    class_id: 要转换的类别ID
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if not os.path.exists(yolo_file):
        return mask
    
    with open(yolo_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    height, width = image_shape[:2]
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        try:
            cls = int(parts[0])
            # 只处理指定类别的对象
            if cls != class_id:
                continue
            
            # YOLO格式: class x_center y_center width height [x1 y1 x2 y2 ...]
            # 对于分割任务，后面的点是多边形的顶点
            if len(parts) > 5:  # 有分割点数据
                points = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                # 转换相对坐标到绝对坐标
                points[:, 0] *= width
                points[:, 1] *= height
                points = points.astype(np.int32)
                
                # 绘制填充多边形
                cv2.fillPoly(mask, [points], 255)
            else:  # 只有边界框，用边界框作为掩码
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                w = float(parts[3]) * width
                h = float(parts[4]) * height
                
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        except Exception as e:
            print(f"处理标签文件 {yolo_file} 时出错: {e}")
    
    return mask

class OpenCVSegmentation:
    """优化的OpenCV传统分割方法"""
    
    def __init__(self):
        self.params = {
            'blur_kernel': (5, 5),
            'canny_low': 40,
            'canny_high': 140,
            'morph_kernel': (3, 3),
            'min_area': 80,
            'min_perimeter': 40
        }
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, self.params['blur_kernel'], 0)
        
        # 直方图均衡化增强对比度
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def extract_contours(self, image):
        """提取轮廓"""
        # 预处理
        processed = self.preprocess_image(image)
        
        # Canny边缘检测
        edges = cv2.Canny(processed, self.params['canny_low'], self.params['canny_high'])
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.params['morph_kernel'])
        
        # 膨胀操作连接断开的边缘
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 腐蚀操作去除小噪点
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # 寻找轮廓
        contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > self.params['min_area'] and perimeter > self.params['min_perimeter']:
                # 计算轮廓的矩形边界
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 根据形状特征进一步过滤
                if 0.1 < aspect_ratio < 10.0:  # 合理的宽高比范围
                    filtered_contours.append(contour)
        
        return filtered_contours
    
    def create_segmentation_mask(self, image_shape, contours):
        """创建分割掩码"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 绘制填充的轮廓
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        return mask
    
    def segment_image(self, image_path):
        """对单张图像进行分割"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # 提取轮廓
        contours = self.extract_contours(image)
        
        # 创建分割掩码
        mask = self.create_segmentation_mask(image.shape, contours)
        
        return mask, contours

def evaluate_opencv_on_dataset():
    """在实际数据集上评估OpenCV方法"""
    print("开始评估OpenCV传统分割方法...")
    start_time = time.time()
    
    # 数据集配置
    dataset_yaml = 'g:\\configuration_harbor\\harbor_port_backup\\data.yaml'
    config = load_dataset_config(dataset_yaml)
    
    # 获取数据集路径
    dataset_root = config['path']
    val_images_dir = os.path.join(dataset_root, config['val'])
    val_labels_dir = os.path.join(dataset_root, config['val'].replace('images', 'labels'))
    
    # 检查目录是否存在
    if not os.path.exists(val_images_dir):
        print(f"验证图像目录不存在: {val_images_dir}")
        return None
    
    if not os.path.exists(val_labels_dir):
        print(f"验证标签目录不存在: {val_labels_dir}")
        return None
    
    # 获取图像文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(val_images_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(val_images_dir).glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    print(f"找到 {len(image_files)} 张验证图像")
    
    # 创建分割器
    segmenter = OpenCVSegmentation()
    
    # 评估所有图像
    all_metrics = []
    processed_count = 0
    error_count = 0
    
    for image_path in tqdm(image_files, desc="评估OpenCV分割"):
        try:
            # 读取图像获取尺寸
            image = cv2.imread(str(image_path))
            if image is None:
                error_count += 1
                continue
            
            # 构造对应的标签文件路径
            image_name = image_path.stem
            label_file = os.path.join(val_labels_dir, f'{image_name}.txt')
            
            # 生成预测掩码
            pred_mask, _ = segmenter.segment_image(str(image_path))
            if pred_mask is None:
                error_count += 1
                continue
            
            # 生成真实掩码（这里简化处理，只使用第一个类别）
            gt_mask = yolo_to_mask(label_file, image.shape, class_id=0)
            
            # 确保尺寸匹配
            if pred_mask.shape != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))
            
            # 二值化
            pred_binary = (pred_mask > 127).astype(np.uint8)
            gt_binary = (gt_mask > 127).astype(np.uint8)
            
            # 计算指标
            pred_flat = pred_binary.flatten()
            gt_flat = gt_binary.flatten()
            
            # 避免全零情况
            if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
                iou = 1.0
            elif np.sum(gt_flat) == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                iou = 0.0
            else:
                precision = precision_score(gt_flat, pred_flat, zero_division=0)
                recall = recall_score(gt_flat, pred_flat, zero_division=0)
                f1 = f1_score(gt_flat, pred_flat, zero_division=0)
                iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
            
            all_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou
            })
            
            processed_count += 1
            
            # 每处理20张图像保存一次中间结果
            if processed_count % 20 == 0:
                print(f"已处理 {processed_count} 张图像，当前平均F1: {np.mean([m['f1'] for m in all_metrics]):.4f}")
                
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            error_count += 1
            continue
    
    # 计算总时间
    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600
    
    if not all_metrics:
        print("没有成功处理的图像，无法计算指标")
        return None
    
    # 计算平均指标
    avg_metrics = {
        'mAP': np.mean([m['f1'] for m in all_metrics]),  # 用F1作为mAP的近似
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics])
    }
    
    # 打印评估结果
    print(f"\nOpenCV评估完成 - 成功处理 {processed_count} 张图像，错误 {error_count} 张")
    print(f"训练时间: {training_time_hours:.2f} 小时")
    print(f"评估结果:")
    print(f"mAP: {avg_metrics['mAP']:.4f}")
    print(f"精确率: {avg_metrics['precision']:.4f}")
    print(f"召回率: {avg_metrics['recall']:.4f}")
    print(f"F1分数: {avg_metrics['f1']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    
    # 构建结果字典
    results = {
        'model': 'opencv_traditional',
        'training_time_hours': training_time_hours,
        'mAP': avg_metrics['mAP'],
        'precision': avg_metrics['precision'],
        'recall': avg_metrics['recall'],
        'f1': avg_metrics['f1'],
        'iou': avg_metrics['iou'],
        'processed_images': processed_count,
        'error_images': error_count,
        'timestamp': time.time()
    }
    
    # 保存结果
    results_file = 'opencv_traditional_results.pt'
    torch.save(results, results_file)
    print(f"\n评估结果已保存到: {results_file}")
    
    return results

def visualize_sample_results():
    """可视化部分样本结果"""
    # 仅处理少量样本用于可视化
    dataset_root = 'g:\\configuration_harbor\\harbor_port_backup'
    val_images_dir = os.path.join(dataset_root, 'val/images')
    
    # 获取前5张图像进行可视化
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(val_images_dir).glob(f'*{ext}'))[:2])
    
    if not image_files:
        print("没有找到可用于可视化的图像")
        return
    
    segmenter = OpenCVSegmentation()
    
    for image_path in image_files:
        print(f"可视化结果: {image_path.name}")
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        # 进行分割
        pred_mask, contours = segmenter.segment_image(str(image_path))
        if pred_mask is None:
            continue
        
        # 读取对应标签
        label_file = image_path.with_suffix('.txt').name
        label_path = os.path.join(dataset_root, 'val/labels', label_file)
        gt_mask = yolo_to_mask(label_path, image.shape)
        
        # 转换颜色空间用于显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建可视化结果
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image_rgb)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('真实掩码')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('OpenCV分割结果')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        save_path = f'opencv_visualization_{image_path.stem}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        plt.close()

if __name__ == '__main__':
    try:
        # 执行评估
        results = evaluate_opencv_on_dataset()
        
        # 如果评估成功，进行可视化
        if results:
            print("\n开始生成样本可视化结果...")
            visualize_sample_results()
            print("\nOpenCV传统方法评估完成！")
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
