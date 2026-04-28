import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建可视化目录
def create_visualization_dir():
    """
    创建可视化目录
    """
    dir_path = "../visualizations"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

# 加载清洗后的数据
def load_cleaned_data():
    """
    加载清洗后的数据
    返回: 清洗后的数据DataFrame
    """
    file_path = "../data/cleaned_sales_data.csv"
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 1. 销售趋势图
def plot_sales_trend(df, output_dir):
    """
    绘制销售趋势图
    参数: 
        df - 清洗后的数据DataFrame
        output_dir - 输出目录
    """
    # 按日期分组计算销售额
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales['date'], daily_sales['sales'], marker='o', linewidth=2, color='#1f77b4')
    plt.title('每日销售趋势', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('销售额 (元)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'sales_trend.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"销售趋势图已保存至: {output_path}")

# 2. 产品类别销售额分布图
def plot_category_distribution(df, output_dir):
    """
    绘制产品类别销售额分布图
    参数: 
        df - 清洗后的数据DataFrame
        output_dir - 输出目录
    """
    # 按类别分组计算销售额
    category_sales = df.groupby('category')['sales'].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff7f0e', '#2ca02c']
    plt.pie(category_sales['sales'], labels=category_sales['category'], autopct='%1.1f%%', 
            startangle=90, colors=colors, wedgeprops={'edgecolor': 'white'})
    plt.title('产品类别销售额分布', fontsize=16)
    plt.axis('equal')  # 确保饼图是圆形
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'category_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"产品类别分布图已保存至: {output_path}")

# 3. 区域销售对比图
def plot_region_comparison(df, output_dir):
    """
    绘制区域销售对比图
    参数: 
        df - 清洗后的数据DataFrame
        output_dir - 输出目录
    """
    # 按区域分组计算销售额
    region_sales = df.groupby('region')['sales'].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='region', y='sales', data=region_sales, palette='viridis')
    plt.title('区域销售对比', fontsize=16)
    plt.xlabel('区域', fontsize=12)
    plt.ylabel('销售额 (元)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值
    for i, v in enumerate(region_sales['sales']):
        plt.text(i, v + 1000, f'{v:.0f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'region_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"区域销售对比图已保存至: {output_path}")

# 4. 产品销售额Top10
def plot_top_products(df, output_dir):
    """
    绘制产品销售额Top10
    参数: 
        df - 清洗后的数据DataFrame
        output_dir - 输出目录
    """
    # 按产品分组计算销售额并排序
    product_sales = df.groupby('product')['sales'].sum().reset_index()
    top_10_products = product_sales.sort_values('sales', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='sales', y='product', data=top_10_products, palette='magma')
    plt.title('产品销售额Top10', fontsize=16)
    plt.xlabel('销售额 (元)', fontsize=12)
    plt.ylabel('产品', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值
    for i, v in enumerate(top_10_products['sales']):
        plt.text(v + 1000, i, f'{v:.0f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'top_10_products.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"产品销售额Top10图已保存至: {output_path}")

# 5. 相关性热力图
def plot_correlation_heatmap(df, output_dir):
    """
    绘制相关性热力图
    参数: 
        df - 清洗后的数据DataFrame
        output_dir - 输出目录
    """
    # 选择数值型列计算相关性
    numeric_cols = ['quantity', 'price', 'sales']
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('变量相关性热力图', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"相关性热力图已保存至: {output_path}")

# 6. 月度销售趋势图
def plot_monthly_sales(df, output_dir):
    """
    绘制月度销售趋势图
    参数: 
        df - 清洗后的数据DataFrame
        output_dir - 输出目录
    """
    # 按月分组计算销售额
    df['month'] = df['date'].dt.month
    monthly_sales = df.groupby(['month', 'category'])['sales'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='month', y='sales', hue='category', data=monthly_sales, marker='o', linewidth=2)
    plt.title('月度销售趋势', fontsize=16)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('销售额 (元)', fontsize=12)
    plt.xticks([1, 2, 3], ['1月', '2月', '3月'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='产品类别')
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'monthly_sales.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"月度销售趋势图已保存至: {output_path}")

# 主函数
def main():
    """
    主函数，执行可视化呈现流程
    """
    # 创建可视化目录
    output_dir = create_visualization_dir()
    
    # 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 绘制各种图表
    plot_sales_trend(df, output_dir)
    plot_category_distribution(df, output_dir)
    plot_region_comparison(df, output_dir)
    plot_top_products(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_monthly_sales(df, output_dir)
    
    print("\n所有可视化图表已生成并保存")

if __name__ == "__main__":
    main()
