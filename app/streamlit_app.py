import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import statistics
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')

class PortDetectionDashboard:
    def __init__(self):
        self.runs_dir = Path("../runs/detect")
        self.dataset_dir = Path("../dataset_raw/images")
        self.raw_private_dir = Path("../raw_private/images")
        self.raw_public_dir = Path("../raw_public/images")
        self.available_runs = self._get_available_runs()
        self.class_map = {0: "ship", 1: "container", 2: "crane"}
        self.dataset_info = {
            'private': {'name': '私有数据集', 'dir': Path('../raw_private'), 'color': '#3498db'},
            'public': {'name': '公开数据集', 'dir': Path('../raw_public'), 'color': '#e74c3c'},
            'mixed': {'name': '混合数据集', 'dir': Path('../dataset_raw'), 'color': '#2ecc71'}
        }
    
    def _get_available_runs(self):
        """获取可用的训练结果目录"""
        runs = []
        if self.runs_dir.exists():
            for run_dir in self.runs_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "results.csv").exists():
                    runs.append(run_dir.name)
        return sorted(runs)
    
    def load_results_data(self, run_name):
        """加载训练结果数据"""
        results_path = self.runs_dir / run_name / "results.csv"
        if results_path.exists():
            return pd.read_csv(results_path)
        return None
    
    def plot_training_curves(self, df, metrics_to_plot):
        """绘制训练曲线"""
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            if f'train/{metric}' in df.columns:
                ax.plot(df['epoch'], df[f'train/{metric}'], label='训练')
            if f'val/{metric}' in df.columns:
                ax.plot(df['epoch'], df[f'val/{metric}'], label='验证')
            ax.set_title(f'{metric} 训练曲线')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_curves(self, df, metrics_to_plot):
        """绘制评估指标曲线"""
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            if f'metrics/{metric}' in df.columns:
                ax.plot(df['epoch'], df[f'metrics/{metric}'])
                ax.set_title(f'{metric} 曲线')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_rate(self, df):
        """绘制学习率曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['epoch'], df['lr/pg0'], label='学习率 pg0')
        ax.plot(df['epoch'], df['lr/pg1'], label='学习率 pg1')
        ax.plot(df['epoch'], df['lr/pg2'], label='学习率 pg2')
        ax.set_title('学习率变化曲线')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('学习率')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    def display_confusion_matrix(self, run_name):
        """显示混淆矩阵"""
        cm_path = self.runs_dir / run_name / "confusion_matrix.png"
        cm_normalized_path = self.runs_dir / run_name / "confusion_matrix_normalized.png"
        
        if cm_path.exists():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("混淆矩阵")
                st.image(Image.open(cm_path), use_column_width=True)
            with col2:
                if cm_normalized_path.exists():
                    st.subheader("归一化混淆矩阵")
                    st.image(Image.open(cm_normalized_path), use_column_width=True)
    
    def display_batch_images(self, run_name):
        """显示批次图像"""
        run_dir = self.runs_dir / run_name
        train_batches = list(run_dir.glob("train_batch*.jpg"))
        val_batches = list(run_dir.glob("val_batch*_pred.jpg"))
        
        if train_batches:
            st.subheader("训练批次图像")
            selected_batch = st.selectbox("选择训练批次", [f.name for f in train_batches[:5]])
            st.image(Image.open(run_dir / selected_batch), use_column_width=True)
        
        if val_batches:
            st.subheader("验证批次预测结果")
            selected_val_batch = st.selectbox("选择验证批次", [f.name for f in val_batches[:5]])
            st.image(Image.open(run_dir / selected_val_batch), use_column_width=True)
            
            # 显示对应的标签图
            label_name = selected_val_batch.replace("_pred", "_labels")
            label_path = run_dir / label_name
            if label_path.exists():
                st.image(Image.open(label_path), use_column_width=True)
    
    def display_dataset_images(self):
        """显示数据集图像"""
        if self.dataset_dir.exists():
            image_files = list(self.dataset_dir.glob("*.jpg"))[:50]
            if image_files:
                st.subheader("数据集样本图像")
                selected_image = st.selectbox("选择图像", [f.name for f in image_files])
                image_path = self.dataset_dir / selected_image
                st.image(Image.open(image_path), use_column_width=True)
                
                # 尝试显示对应的标签信息
                label_path = self.dataset_dir.parent / "labels" / (selected_image.replace(".jpg", ".txt"))
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                    if labels:
                        st.text("标签信息:")
                        for label in labels:
                            parts = label.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_name = self.class_map.get(class_id, f"类别{class_id}")
                                st.text(f"{class_name} - 中心点: ({parts[1]}, {parts[2]}), 宽高: ({parts[3]}, {parts[4]})")
    
    def compare_multiple_runs(self, runs_to_compare, metrics_to_compare):
        """比较多个训练结果"""
        fig, axes = plt.subplots(len(metrics_to_compare), 1, figsize=(12, 5 * len(metrics_to_compare)))
        if len(metrics_to_compare) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics_to_compare):
            ax = axes[i]
            best_values = {}
            
            for run_name in runs_to_compare:
                df = self.load_results_data(run_name)
                if df is not None and f'metrics/{metric}' in df.columns:
                    line, = ax.plot(df['epoch'], df[f'metrics/{metric}'], label=run_name, linewidth=2)
                    
                    # 找到最佳值并标记
                    best_value = df[f'metrics/{metric}'].max()
                    best_epoch = df[f'metrics/{metric}'].idxmax()
                    best_values[run_name] = (best_value, best_epoch)
                    
                    # 添加最佳值标注
                    ax.scatter(best_epoch, best_value, color=line.get_color(), s=100, zorder=5)
                    ax.annotate(f'{best_value:.3f}', 
                               (best_epoch, best_value), 
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=9, fontweight='bold')
            
            ax.set_title(f'{metric} 对比曲线', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_dataset_composition(self):
        """分析并对比不同数据集的组成"""
        dataset_stats = {}
        
        for dataset_key, dataset_info in self.dataset_info.items():
            images_dir = dataset_info['dir'] / 'images'
            labels_dir = dataset_info['dir'] / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                # 统计图像数量
                image_files = list(images_dir.glob("*.jpg"))
                num_images = len(image_files)
                
                # 统计标签数量和类别分布
                label_files = list(labels_dir.glob("*.txt"))
                num_labels = len(label_files)
                
                class_counts = {0: 0, 1: 0, 2: 0}
                obj_per_image = []
                
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        obj_count = 0
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                                    obj_count += 1
                        if obj_count > 0:
                            obj_per_image.append(obj_count)
                
                # 计算统计信息
                total_objects = sum(class_counts.values())
                avg_objects_per_image = np.mean(obj_per_image) if obj_per_image else 0
                std_objects_per_image = np.std(obj_per_image) if obj_per_image else 0
                
                dataset_stats[dataset_key] = {
                    'name': dataset_info['name'],
                    'color': dataset_info['color'],
                    'num_images': num_images,
                    'num_labels': num_labels,
                    'class_counts': class_counts,
                    'total_objects': total_objects,
                    'avg_objects_per_image': avg_objects_per_image,
                    'std_objects_per_image': std_objects_per_image
                }
        
        return dataset_stats
    
    def plot_dataset_comparison(self, dataset_stats):
        """绘制数据集对比图表"""
        # 准备数据
        datasets = list(dataset_stats.keys())
        dataset_names = [dataset_stats[d]['name'] for d in datasets]
        colors = [dataset_stats[d]['color'] for d in datasets]
        
        # 图1: 数据集规模对比
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 图像数量对比
        image_counts = [dataset_stats[d]['num_images'] for d in datasets]
        ax1.bar(dataset_names, image_counts, color=colors)
        ax1.set_title('数据集图像数量对比', fontsize=14)
        ax1.set_ylabel('图像数量', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 每个图像平均目标数对比
        avg_objects = [dataset_stats[d]['avg_objects_per_image'] for d in datasets]
        std_objects = [dataset_stats[d]['std_objects_per_image'] for d in datasets]
        ax2.bar(dataset_names, avg_objects, yerr=std_objects, color=colors, capsize=5)
        ax2.set_title('每个图像平均目标数对比', fontsize=14)
        ax2.set_ylabel('平均目标数', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # 类别分布堆叠柱状图
        class_ids = [0, 1, 2]
        class_names = [self.class_map[cid] for cid in class_ids]
        bottom = [0] * len(datasets)
        
        for cid in class_ids:
            values = [dataset_stats[d]['class_counts'][cid] for d in datasets]
            ax3.bar(dataset_names, values, bottom=bottom, label=class_names[cid])
            bottom = [bottom[i] + values[i] for i in range(len(datasets))]
        
        ax3.set_title('数据集类别分布', fontsize=14)
        ax3.set_ylabel('目标数量', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='目标类别')
        
        # 类别占比饼图（右侧）
        for i, d in enumerate(datasets):
            class_values = list(dataset_stats[d]['class_counts'].values())
            if sum(class_values) > 0:
                # 在ax4上创建子图
                pie_ax = fig.add_axes([0.25 + i*0.25, 0.1, 0.2, 0.3])
                pie_ax.pie(class_values, labels=class_names, autopct='%1.1f%%', colors=['#9b59b6', '#3498db', '#e74c3c'])
                pie_ax.set_title(f'{dataset_stats[d]["name"]} 类别占比')
        
        plt.tight_layout()
        return fig
    
    def generate_data_analysis_report(self, dataset_stats):
        """生成数据分析报告"""
        report = """
        ## 数据集对比分析报告
        
        ### 数据集基本信息对比
        """
        
        # 基本信息表格
        report += "| 数据集 | 图像数量 | 标签文件数量 | 总目标数 | 平均每图目标数 |\n"
        report += "|--------|----------|--------------|----------|----------------|\n"
        
        for dataset_key, stats in dataset_stats.items():
            report += f"| {stats['name']} | {stats['num_images']} | {stats['num_labels']} | {stats['total_objects']} | {stats['avg_objects_per_image']:.2f} ± {stats['std_objects_per_image']:.2f} |\n"
        
        # 分析结论
        report += """
        
        ### 数据集特点分析
        """
        
        # 分析私有数据集和公开数据集的差异
        if 'private' in dataset_stats and 'public' in dataset_stats:
            private_stats = dataset_stats['private']
            public_stats = dataset_stats['public']
            
            report += """
            - **数据规模对比**：
        """
            
            if private_stats['num_images'] > public_stats['num_images']:
                diff_pct = (private_stats['num_images'] - public_stats['num_images']) / public_stats['num_images'] * 100
                report += f"  - 私有数据集图像数量 ({private_stats['num_images']}) 比公开数据集 ({public_stats['num_images']}) 多 {diff_pct:.1f}%\n"
            else:
                diff_pct = (public_stats['num_images'] - private_stats['num_images']) / private_stats['num_images'] * 100
                report += f"  - 公开数据集图像数量 ({public_stats['num_images']}) 比私有数据集 ({private_stats['num_images']}) 多 {diff_pct:.1f}%\n"
            
            report += """
            - **目标密度对比**：
        """
            
            if private_stats['avg_objects_per_image'] > public_stats['avg_objects_per_image']:
                diff_pct = (private_stats['avg_objects_per_image'] - public_stats['avg_objects_per_image']) / public_stats['avg_objects_per_image'] * 100
                report += f"  - 私有数据集每图平均目标数 ({private_stats['avg_objects_per_image']:.2f}) 比公开数据集 ({public_stats['avg_objects_per_image']:.2f}) 多 {diff_pct:.1f}%\n"
            else:
                diff_pct = (public_stats['avg_objects_per_image'] - private_stats['avg_objects_per_image']) / private_stats['avg_objects_per_image'] * 100
                report += f"  - 公开数据集每图平均目标数 ({public_stats['avg_objects_per_image']:.2f}) 比私有数据集 ({private_stats['avg_objects_per_image']:.2f}) 多 {diff_pct:.1f}%\n"
        
        # 类别分布分析
        report += """
        - **类别分布特点**：
        """
        
        for dataset_key, stats in dataset_stats.items():
            class_counts = stats['class_counts']
            total = sum(class_counts.values())
            if total > 0:
                class_percentages = {cid: (count/total)*100 for cid, count in class_counts.items()}
                dominant_class = max(class_percentages, key=class_percentages.get)
                
                report += f"  - {stats['name']} 的主要类别是 {self.class_map[dominant_class]} ({class_percentages[dominant_class]:.1f}%)\n"
        
        # 结论和建议
        report += """
        
        ### 结论与建议
        
        1. **数据集互补性**：通过对比分析发现，私有数据集和公开数据集在目标密度、类别分布等方面存在差异，混合使用可以提高模型的泛化能力。
        
        2. **数据增强策略**：
           - 对于类别分布不平衡的数据集，建议采用过采样或类别权重调整策略
           - 对于目标密度较低的数据集，可以考虑增加合成样本
        
        3. **训练策略建议**：
           - 利用混合数据集进行训练，同时保留对私有数据集的微调阶段
           - 针对不同数据集的特点，调整数据增强参数
           - 采用交叉验证评估模型在不同数据分布下的鲁棒性
        """
        
        return report
    
    def compare_dataset_performance(self):
        """比较不同数据集训练模型的性能"""
        # 尝试识别对应不同数据集的训练结果
        dataset_runs = {
            'private': [run for run in self.available_runs if 'private' in run.lower()],
            'public': [run for run in self.available_runs if 'public' in run.lower()],
            'mixed': [run for run in self.available_runs if 'mixed' in run.lower() or 'port_custom' in run.lower()]
        }
        
        # 获取每个数据集最好的训练结果
        best_runs = {}
        for dataset, runs in dataset_runs.items():
            if runs:
                best_score = -1
                best_run = None
                for run in runs:
                    df = self.load_results_data(run)
                    if df is not None and 'metrics/mAP50(B)' in df.columns:
                        final_score = df['metrics/mAP50(B)'].iloc[-1]
                        if final_score > best_score:
                            best_score = final_score
                            best_run = run
                if best_run:
                    best_runs[dataset] = best_run
        
        return best_runs
    
    def run(self):
        """运行展示系统"""
        st.set_page_config(page_title="港口目标检测算法展示系统", layout="wide")
        st.title("港口目标检测算法展示系统")
        
        # 侧边栏设置
        st.sidebar.header("配置选项")
        page = st.sidebar.selectbox(
            "选择展示页面",
            ["训练结果展示", "数据集展示", "模型对比", "数据集对比分析", "关于系统"]
        )
        
        if page == "训练结果展示":
            if not self.available_runs:
                st.error("未找到可用的训练结果")
                return
            
            selected_run = st.sidebar.selectbox("选择训练结果", self.available_runs)
            df = self.load_results_data(selected_run)
            
            if df is not None:
                st.header(f"{selected_run} 训练结果")
                
                # 显示基本统计信息
                st.subheader("训练统计信息")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最终 mAP50", f"{df['metrics/mAP50(B)'].iloc[-1]:.3f}")
                with col2:
                    st.metric("最终 mAP50-95", f"{df['metrics/mAP50-95(B)'].iloc[-1]:.3f}")
                with col3:
                    st.metric("总训练轮次", f"{len(df)}")
                
                # 训练曲线展示
                st.subheader("训练曲线")
                metric_options = ["box_loss", "cls_loss", "dfl_loss"]
                selected_metrics = st.multiselect("选择损失函数", metric_options, default=["box_loss"])
                if selected_metrics:
                    st.pyplot(self.plot_training_curves(df, selected_metrics))
                
                # 评估指标曲线
                st.subheader("评估指标")
                eval_options = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
                selected_eval = st.multiselect("选择评估指标", eval_options, default=["mAP50(B)"])
                if selected_eval:
                    st.pyplot(self.plot_metrics_curves(df, selected_eval))
                
                # 学习率曲线
                st.subheader("学习率变化")
                st.pyplot(self.plot_learning_rate(df))
                
                # 混淆矩阵
                self.display_confusion_matrix(selected_run)
                
                # 批次图像
                self.display_batch_images(selected_run)
        
        elif page == "数据集展示":
            st.header("数据集展示")
            self.display_dataset_images()
            
            # 数据集统计
            st.subheader("数据集统计信息")
            labels_dir = self.dataset_dir.parent / "labels"
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                total_images = len(list(self.dataset_dir.glob("*.jpg")))
                st.write(f"总图像数量: {total_images}")
                st.write(f"总标签文件数量: {len(label_files)}")
                
                # 统计各类别数量
                class_counts = {0: 0, 1: 0, 2: 0}
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                
                # 显示类别分布
                st.subheader("类别分布")
                class_names = [self.class_map.get(cid, f"类别{cid}") for cid in class_counts.keys()]
                class_values = list(class_counts.values())
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.bar(class_names, class_values)
                ax1.set_title("各类别数量")
                ax1.set_xlabel("类别")
                ax1.set_ylabel("数量")
                
                ax2.pie(class_values, labels=class_names, autopct='%1.1f%%')
                ax2.set_title("类别占比")
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif page == "模型对比":
            st.header("模型对比")
            
            # 选择对比模式
            comparison_mode = st.radio(
                "选择对比模式",
                ("自定义模型对比", "数据集训练性能对比")
            )
            
            if comparison_mode == "自定义模型对比":
                if len(self.available_runs) < 2:
                    st.warning("至少需要两个训练结果才能进行对比")
                else:
                    runs_to_compare = st.multiselect("选择要对比的模型", self.available_runs, default=self.available_runs[:2])
                    if len(runs_to_compare) >= 2:
                        compare_options = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
                        metrics_to_compare = st.multiselect("选择要对比的指标", compare_options, default=["mAP50(B)"])
                        if metrics_to_compare:
                            # 显示对比图表
                            st.pyplot(self.compare_multiple_runs(runs_to_compare, metrics_to_compare))
                            
                            # 显示详细的指标对比表格
                            st.subheader("模型性能详细对比")
                            comparison_data = []
                            for run_name in runs_to_compare:
                                df = self.load_results_data(run_name)
                                if df is not None:
                                    run_data = {
                                        '模型': run_name,
                                        '最终 mAP50': f"{df['metrics/mAP50(B)'].iloc[-1]:.3f}",
                                        '最佳 mAP50': f"{df['metrics/mAP50(B)'].max():.3f}",
                                        '最佳轮次': f"{df['metrics/mAP50(B)'].idxmax()}",
                                        '最终 mAP50-95': f"{df['metrics/mAP50-95(B)'].iloc[-1]:.3f}",
                                        '最终 精确率': f"{df['metrics/precision(B)'].iloc[-1]:.3f}",
                                        '最终 召回率': f"{df['metrics/recall(B)'].iloc[-1]:.3f}"
                                    }
                                    comparison_data.append(run_data)
                            
                            if comparison_data:
                                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            else:  # 数据集训练性能对比
                best_runs = self.compare_dataset_performance()
                if len(best_runs) >= 2:
                    dataset_names = [self.dataset_info[d]['name'] for d in best_runs.keys()]
                    runs_to_compare = list(best_runs.values())
                    
                    st.subheader("不同数据集训练性能对比")
                    st.write("选择每个数据集表现最佳的模型进行对比:")
                    
                    # 显示选择的模型
                    for dataset, run_name in best_runs.items():
                        st.write(f"{self.dataset_info[dataset]['name']}: {run_name}")
                    
                    # 对比图表
                    compare_options = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
                    metrics_to_compare = st.multiselect("选择要对比的指标", compare_options, default=["mAP50(B)"])
                    
                    if metrics_to_compare:
                        st.pyplot(self.compare_multiple_runs(runs_to_compare, metrics_to_compare))
                        
                        # 性能雷达图
                        st.subheader("模型性能雷达图")
                        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                        
                        # 准备雷达图数据
                        metrics = ['mAP50(B)', 'mAP50-95(B)', 'precision(B)', 'recall(B)']
                        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
                        angles += angles[:1]  # 闭合雷达图
                        
                        for i, run_name in enumerate(runs_to_compare):
                            df = self.load_results_data(run_name)
                            if df is not None:
                                values = [df[f'metrics/{metric}'].iloc[-1] for metric in metrics]
                                values += values[:1]  # 闭合雷达图
                                
                                dataset_key = list(best_runs.keys())[i]
                                color = self.dataset_info[dataset_key]['color']
                                label = self.dataset_info[dataset_key]['name']
                                
                                ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=label)
                                ax.fill(angles, values, color=color, alpha=0.25)
                        
                        # 设置雷达图
                        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
                        ax.set_ylim(0, 1)
                        ax.set_title('模型性能雷达图', size=15, y=1.1)
                        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                        
                        st.pyplot(fig)
                else:
                    st.warning("无法找到足够的数据集训练结果进行对比")
        

        
        
            st.header("数据集对比分析")
            
            # 分析数据集组成
            st.subheader("数据集组成分析")
            dataset_stats = self.analyze_dataset_composition()
            
            if dataset_stats:
                # 显示数据集对比图表
                st.pyplot(self.plot_dataset_comparison(dataset_stats))
                
                # 显示详细统计数据
                st.subheader("详细统计数据")
                for dataset_key, stats in dataset_stats.items():
                    with st.expander(f"{stats['name']} 详细信息"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"图像数量: {stats['num_images']}")
                            st.write(f"标签文件数量: {stats['num_labels']}")
                            st.write(f"总目标数: {stats['total_objects']}")
                            st.write(f"平均每图目标数: {stats['avg_objects_per_image']:.2f} ± {stats['std_objects_per_image']:.2f}")
                        with col2:
                            st.subheader("类别分布")
                            for cid, count in stats['class_counts'].items():
                                class_name = self.class_map[cid]
                                percentage = (count / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
                                st.write(f"{class_name}: {count} ({percentage:.1f}%)")
                
                # 显示数据分析报告
                st.subheader("数据分析报告")
                report = self.generate_data_analysis_report(dataset_stats)
                st.markdown(report)
        
        elif page == "关于系统":
            st.header("关于系统")
            st.markdown("""
            ## 港口目标检测算法展示系统
            
            **功能说明：**
            - **训练结果展示**：展示各个训练模型的损失曲线、评估指标、混淆矩阵等
            - **数据集展示**：浏览数据集样本，查看数据集统计信息
            - **模型对比**：对比不同模型的性能指标
            - **数据集对比分析**：分析和对比不同数据集的组成特点和训练性能
            
            **支持的目标类别：**
            - 船舶 (ship)
            - 集装箱 (container)
            - 起重机 (crane)
            
            **系统版本：** v1.1
            
            **更新内容：**
            - 添加数据集对比分析功能
            - 增强数据分析的严谨性和可视化效果
            - 改进模型对比功能，支持雷达图展示
            - 优化整体用户界面和交互体验
            """)


def main():
    dashboard = PortDetectionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()


