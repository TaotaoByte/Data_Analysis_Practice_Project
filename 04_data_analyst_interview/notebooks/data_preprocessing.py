import pandas as pd
import numpy as np

# 数据获取函数
def load_data():
    """
    加载销售数据
    返回: 原始销售数据DataFrame
    """
    # 读取原始销售数据
    file_path = "../data/sales_data.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 数据质量检查函数
def data_quality_check(df):
    """
    数据质量检查
    参数: df - 原始销售数据DataFrame
    返回: 检查结果字典
    """
    check_results = {}
    
    # 基本信息
    check_results['基本信息'] = {
        '行数': len(df),
        '列数': len(df.columns),
        '列名': list(df.columns)
    }
    
    # 缺失值检查
    missing_values = df.isnull().sum()
    check_results['缺失值检查'] = missing_values[missing_values > 0].to_dict()
    
    # 数据类型检查
    check_results['数据类型'] = df.dtypes.to_dict()
    
    # 描述性统计
    check_results['数值型描述统计'] = df.describe().to_dict()
    
    return check_results

# 数据清洗函数
def clean_data(df):
    """
    数据清洗
    参数: df - 原始销售数据DataFrame
    返回: 清洗后的数据DataFrame
    """
    # 创建副本避免修改原始数据
    cleaned_df = df.copy()
    
    # 处理日期格式
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    
    # 处理缺失值
    # 对于数值型列，使用均值填充
    numeric_cols = ['quantity', 'price']
    for col in numeric_cols:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # 对于分类列，使用众数填充
    categorical_cols = ['product', 'category', 'region']
    for col in categorical_cols:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    # 处理异常值
    # 使用IQR方法检测并处理异常值
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 替换异常值为边界值
        cleaned_df[col] = np.where(cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col])
        cleaned_df[col] = np.where(cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col])
    
    # 计算销售额
    cleaned_df['sales'] = cleaned_df['quantity'] * cleaned_df['price']
    
    return cleaned_df

# 主函数
def main():
    """
    主函数，执行数据获取与预处理流程
    """
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 数据质量检查
    print("\n数据质量检查结果:")
    check_results = data_quality_check(df)
    for key, value in check_results.items():
        print(f"\n{key}:")
        print(value)
    
    # 数据清洗
    cleaned_df = clean_data(df)
    print("\n数据清洗完成")
    print(f"清洗后数据形状: {cleaned_df.shape}")
    
    # 保存清洗后的数据
    output_path = "../data/cleaned_sales_data.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\n清洗后的数据已保存至: {output_path}")
    
    return cleaned_df

if __name__ == "__main__":
    main()
