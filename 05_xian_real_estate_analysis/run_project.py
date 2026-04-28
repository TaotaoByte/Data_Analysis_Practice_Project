#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西安二手房房价影响因素分析与预测项目
主运行脚本
"""

import os
import subprocess

def main():
    """
    主函数，运行项目完整流程
    """
    print("=" * 70)
    print("西安二手房房价影响因素分析与预测项目")
    print("=" * 70)

    notebooks_dir = os.path.join(os.path.dirname(__file__), 'notebooks')

    steps = [
        ("1. 数据清洗", "01_data_cleaning.py"),
        ("2. 探索性分析", "02_exploratory_analysis.py"),
        ("3. 特征工程", "03_feature_engineering.py"),
        ("4. 建模分析", "04_modeling_analysis.py"),
        ("5. 数据可视化", "05_data_visualization.py")
    ]

    for step_name, script_name in steps:
        print(f"\n{step_name}")
        print("-" * 50)
        try:
            result = subprocess.run(
                ['python', script_name],
                cwd=notebooks_dir,
                check=True,
                capture_output=False
            )
            print(f"[SUCCESS] {script_name} 执行完成")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {script_name} 执行失败")
            return

    print("\n" + "=" * 70)
    print("项目运行完成!")
    print("=" * 70)

    print("\n输出文件说明:")
    print("\n[数据文件]")
    print("  data/cleaned_house_data.csv       - 清洗后的房源数据")
    print("  data/cleaned_index_data.csv       - 清洗后的房价指数数据")
    print("  data/features_engineered_data.csv - 特征工程处理后的数据")
    print("  data/feature_importance.csv       - 特征重要性排名")

    print("\n[模型文件]")
    print("  models/best_price_prediction_model.joblib - 最佳预测模型")
    print("  models/model_comparison.csv              - 模型性能对比")

    print("\n[可视化图表]")
    print("  visualizations/price_trend.png       - 房价趋势图")
    print("  visualizations/district_comparison.png - 区域对比图")
    print("  visualizations/feature_importance.png - 特征重要性图")
    print("  visualizations/model_comparison.png   - 模型对比图")
    print("  visualizations/price_distribution.png - 价格分布图")
    print("  visualizations/correlation_heatmap.png - 相关性热力图")
    print("  visualizations/scatter_matrix.png    - 散点矩阵图")

    print("\n[交互式图表]")
    print("  visualizations/*_interactive.html   - 可在浏览器中打开的交互式图表")

    print("\n[文档]")
    print("  docs/project_requirements.md - 项目需求文档")
    print("  docs/project_summary.md      - 项目总结报告")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
