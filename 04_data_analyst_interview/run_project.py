#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商销售数据分析与预测项目主脚本
用于运行整个项目的完整流程
"""

import os
import subprocess

def main():
    """
    主函数，运行整个项目流程
    """
    print("=== 电商销售数据分析与预测项目 ===")
    
    # 切换到notebooks目录运行各个脚本
    notebooks_dir = os.path.join(os.path.dirname(__file__), 'notebooks')
    
    print("\n1. 数据获取与预处理")
    subprocess.run(['python', 'data_preprocessing.py'], cwd=notebooks_dir, check=True)
    
    print("\n2. 探索性数据分析")
    subprocess.run(['python', 'exploratory_analysis.py'], cwd=notebooks_dir, check=True)
    
    print("\n3. 特征工程")
    subprocess.run(['python', 'feature_engineering.py'], cwd=notebooks_dir, check=True)
    
    print("\n4. 建模分析")
    subprocess.run(['python', 'modeling_analysis.py'], cwd=notebooks_dir, check=True)
    
    print("\n5. 数据可视化")
    subprocess.run(['python', 'data_visualization.py'], cwd=notebooks_dir, check=True)
    
    print("\n=== 项目运行完成 ===")
    print("\n项目输出文件:")
    print("- 数据文件: data/cleaned_sales_data.csv, data/features_sales_data.csv")
    print("- 模型文件: models/best_sales_prediction_model.joblib")
    print("- 可视化图表: visualizations/")
    print("- 文档: docs/project_overview.md, docs/conclusions_recommendations.md")

if __name__ == "__main__":
    main()
