import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
        print(f"成功加载清洗后的数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 单变量分析
def univariate_analysis(df):
    """
    单变量分析
    参数: df - 清洗后的数据DataFrame
    """
    print("\n=== 单变量分析 ===")
    
    # 1. 销售趋势分析
    print("\n1. 销售趋势分析")
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    print(f"日均销售额: {daily_sales['sales'].mean():.2f} 元")
    print(f"销售额标准差: {daily_sales['sales'].std():.2f} 元")
    print(f"最高销售额日期: {daily_sales.loc[daily_sales['sales'].idxmax(), 'date']}")
    print(f"最高销售额: {daily_sales['sales'].max():.2f} 元")
    
    # 2. 产品类别分布
    print("\n2. 产品类别分布")
    category_dist = df.groupby('category').agg({
        'quantity': 'sum',
        'sales': 'sum'
    }).reset_index()
    print(category_dist)
    
    # 3. 区域分布
    print("\n3. 区域分布")
    region_dist = df.groupby('region').agg({
        'quantity': 'sum',
        'sales': 'sum'
    }).reset_index()
    print(region_dist)
    
    # 4. 产品分布
    print("\n4. 产品分布")
    product_dist = df.groupby('product').agg({
        'quantity': 'sum',
        'sales': 'sum'
    }).reset_index().sort_values('sales', ascending=False)
    print(product_dist.head(10))

# 双变量分析
def bivariate_analysis(df):
    """
    双变量分析
    参数: df - 清洗后的数据DataFrame
    """
    print("\n=== 双变量分析 ===")
    
    # 1. 产品类别与销售额关系
    print("\n1. 产品类别与销售额关系")
    category_sales = df.groupby('category')['sales'].sum().reset_index()
    print(category_sales)
    
    # 2. 区域与销售额关系
    print("\n2. 区域与销售额关系")
    region_sales = df.groupby('region')['sales'].sum().reset_index()
    print(region_sales)
    
    # 3. 产品与销售额关系
    print("\n3. 产品与销售额关系")
    product_sales = df.groupby('product')['sales'].sum().reset_index().sort_values('sales', ascending=False)
    print(product_sales.head(10))
    
    # 4. 价格与销售额关系
    print("\n4. 价格与销售额关系")
    price_sales_corr = df[['price', 'sales']].corr().iloc[0, 1]
    print(f"价格与销售额相关系数: {price_sales_corr:.4f}")
    
    # 5. 数量与销售额关系
    print("\n5. 数量与销售额关系")
    quantity_sales_corr = df[['quantity', 'sales']].corr().iloc[0, 1]
    print(f"数量与销售额相关系数: {quantity_sales_corr:.4f}")

# 多变量分析
def multivariate_analysis(df):
    """
    多变量分析
    参数: df - 清洗后的数据DataFrame
    """
    print("\n=== 多变量分析 ===")
    
    # 1. 区域-类别-销售额分析
    print("\n1. 区域-类别-销售额分析")
    region_category_sales = df.groupby(['region', 'category'])['sales'].sum().reset_index()
    print(region_category_sales)
    
    # 2. 时间-类别-销售额分析
    print("\n2. 时间-类别-销售额分析")
    # 按月分组
    df['month'] = df['date'].dt.month
    monthly_category_sales = df.groupby(['month', 'category'])['sales'].sum().reset_index()
    print(monthly_category_sales)
    
    # 3. 时间-区域-销售额分析
    print("\n3. 时间-区域-销售额分析")
    monthly_region_sales = df.groupby(['month', 'region'])['sales'].sum().reset_index()
    print(monthly_region_sales)

# 主函数
def main():
    """
    主函数，执行探索性数据分析流程
    """
    # 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 单变量分析
    univariate_analysis(df)
    
    # 双变量分析
    bivariate_analysis(df)
    
    # 多变量分析
    multivariate_analysis(df)

if __name__ == "__main__":
    main()
