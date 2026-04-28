import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

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

# 创建时间相关特征
def create_time_features(df):
    """
    创建时间相关特征
    参数: df - 清洗后的数据DataFrame
    返回: 添加时间特征后的DataFrame
    """
    df['day_of_week'] = df['date'].dt.dayofweek  # 周几（0-6）
    df['month'] = df['date'].dt.month  # 月份
    df['day_of_month'] = df['date'].dt.day  # 当月第几天
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 是否周末
    return df

# 创建产品相关特征
def create_product_features(df):
    """
    创建产品相关特征
    参数: df - 清洗后的数据DataFrame
    返回: 添加产品特征后的DataFrame
    """
    # 产品价格区间
    df['price_range'] = pd.cut(df['price'], bins=[0, 100, 500, 2000, 10000], 
                              labels=['low', 'medium', 'high', 'premium'])
    
    # 产品类别编码
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # 产品编码
    le_product = LabelEncoder()
    df['product_encoded'] = le_product.fit_transform(df['product'])
    
    return df

# 创建区域相关特征
def create_region_features(df):
    """
    创建区域相关特征
    参数: df - 清洗后的数据DataFrame
    返回: 添加区域特征后的DataFrame
    """
    # 区域编码
    le_region = LabelEncoder()
    df['region_encoded'] = le_region.fit_transform(df['region'])
    
    return df

# 创建交互特征
def create_interaction_features(df):
    """
    创建交互特征
    参数: df - 清洗后的数据DataFrame
    返回: 添加交互特征后的DataFrame
    """
    # 类别与区域的交互
    df['category_region_interaction'] = df['category_encoded'] * 10 + df['region_encoded']
    
    # 价格与数量的交互
    df['price_quantity_interaction'] = df['price'] * df['quantity']
    
    return df

# 特征选择
def select_features(X, y, k=10):
    """
    特征选择
    参数: 
        X - 特征矩阵
        y - 目标变量
        k - 选择的特征数量
    返回: 选择的特征名称
    """
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features

# 特征标准化
def standardize_features(X):
    """
    特征标准化
    参数: X - 特征矩阵
    返回: 标准化后的特征矩阵
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# 主函数
def main():
    """
    主函数，执行特征工程流程
    """
    # 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 创建时间特征
    df = create_time_features(df)
    print("\n创建时间特征完成")
    
    # 创建产品特征
    df = create_product_features(df)
    print("创建产品特征完成")
    
    # 创建区域特征
    df = create_region_features(df)
    print("创建区域特征完成")
    
    # 创建交互特征
    df = create_interaction_features(df)
    print("创建交互特征完成")
    
    # 准备特征矩阵和目标变量
    feature_columns = ['quantity', 'price', 'day_of_week', 'month', 'day_of_month', 
                      'is_weekend', 'category_encoded', 'product_encoded', 
                      'region_encoded', 'category_region_interaction', 
                      'price_quantity_interaction']
    
    X = df[feature_columns]
    y = df['sales']
    
    # 特征选择
    selected_features = select_features(X, y, k=8)
    print(f"\n选择的特征: {list(selected_features)}")
    
    # 特征标准化
    X_scaled = standardize_features(X[selected_features])
    print("\n特征标准化完成")
    print(f"标准化后特征形状: {X_scaled.shape}")
    
    # 保存特征工程后的数据
    df_with_features = pd.concat([df, X_scaled.add_suffix('_scaled')], axis=1)
    output_path = "../data/features_sales_data.csv"
    df_with_features.to_csv(output_path, index=False)
    print(f"\n特征工程后的数据已保存至: {output_path}")
    
    return df_with_features

if __name__ == "__main__":
    main()
