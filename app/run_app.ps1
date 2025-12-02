# run_app.ps1 - 港口目标检测应用启动脚本
# 设置环境变量以避免OpenMP重复初始化警告
$env:KMP_DUPLICATE_LIB_OK='TRUE'

# 输出启动信息
Write-Host "正在启动港口目标检测Streamlit应用..."
Write-Host "环境变量KMP_DUPLICATE_LIB_OK已设置为TRUE，以避免OpenMP重复初始化警告"
Write-Host "已应用修复：添加numpy模块导入和修复模型加载兼容性问题"

# 运行Streamlit应用
python -m streamlit run streamlit_app.py