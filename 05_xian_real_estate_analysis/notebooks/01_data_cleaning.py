import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    加载原始数据
    """
    print("=" * 60)
    print("数据加载")
    print("=" * 60)

    house_df = pd.read_csv("../data/xian_second_hand_houses.csv")
    index_df = pd.read_csv("../data/xian_house_price_index.csv")
    district_df = pd.read_csv("../data/xian_district_price.csv")

    print(f"二手房数据: {house_df.shape[0]} 条记录, {house_df.shape[1]} 个字段")
    print(f"房价指数数据: {index_df.shape[0]} 条记录, {index_df.shape[1]} 个字段")
    print(f"区域均价数据: {district_df.shape[0]} 条记录, {district_df.shape[1]} 个字段")

    return house_df, index_df, district_df

def clean_house_data(df):
    """
    清洗二手房数据
    """
    print("\n" + "=" * 60)
    print("二手房数据清洗")
    print("=" * 60)

    cleaned = df.copy()

    print(f"\n原始数据形状: {cleaned.shape}")

    print("\n1. 数据类型检查:")
    print(cleaned.dtypes)

    print("\n2. 缺失值检查:")
    missing = cleaned.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "无缺失值")

    print("\n3. 异常值检测:")
    for col in ['area', 'total_price', 'unit_price', 'floor', 'total_floors']:
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = cleaned[(cleaned[col] < lower) | (cleaned[col] > upper)][col]
        print(f"{col}: {len(outliers)} 个异常值 (范围: {lower:.2f} - {upper:.2f})")

    print("\n4. 重复值检查:")
    duplicates = cleaned.duplicated().sum()
    print(f"重复记录: {duplicates} 条")

    print("\n5. 数据一致性检查:")
    invalid_floor = cleaned[cleaned['floor'] > cleaned['total_floors']]
    print(f"楼层异常(楼层>总楼层): {len(invalid_floor)} 条")

    invalid_price = cleaned[cleaned['total_price'] <= 0]
    print(f"价格异常(总价<=0): {len(invalid_price)} 条")

    cleaned['unit_price_calc'] = (cleaned['total_price'] * 10000 / cleaned['area']).round(0)
    price_diff = abs(cleaned['unit_price'] - cleaned['unit_price_calc'])
    print(f"\n单价计算差异(元/㎡):")
    print(f"  平均差异: {price_diff.mean():.2f}")
    print(f"  最大差异: {price_diff.max():.2f}")

    cleaned = cleaned.drop('unit_price_calc', axis=1)

    print("\n" + "=" * 60)
    print("数据清洗完成")
    print("=" * 60)
    print(f"清洗后数据形状: {cleaned.shape}")

    return cleaned

def clean_index_data(df):
    """
    清洗房价指数数据
    """
    print("\n" + "=" * 60)
    print("房价指数数据清洗")
    print("=" * 60)

    cleaned = df.copy()

    cleaned['month'] = pd.to_datetime(cleaned['month'])

    cleaned = cleaned.sort_values('month')

    print(f"\n时间范围: {cleaned['month'].min()} 至 {cleaned['month'].max()}")
    print(f"数据条数: {len(cleaned)}")

    print("\n数据统计:")
    print(cleaned.describe())

    return cleaned

def clean_district_data(df):
    """
    清洗区域均价数据
    """
    print("\n" + "=" * 60)
    print("区域均价数据清洗")
    print("=" * 60)

    cleaned = df.copy()

    print(f"\n区域列表: {cleaned['district'].tolist()}")

    return cleaned

def save_cleaned_data(house_df, index_df, district_df):
    """
    保存清洗后的数据
    """
    house_df.to_csv("../data/cleaned_house_data.csv", index=False)
    index_df.to_csv("../data/cleaned_index_data.csv", index=False)
    district_df.to_csv("../data/cleaned_district_data.csv", index=False)

    print("\n" + "=" * 60)
    print("数据保存完成")
    print("=" * 60)
    print("文件路径:")
    print("  - ../data/cleaned_house_data.csv")
    print("  - ../data/cleaned_index_data.csv")
    print("  - ../data/cleaned_district_data.csv")

def main():
    """
    主函数
    """
    house_df, index_df, district_df = load_data()

    house_df = clean_house_data(house_df)
    index_df = clean_index_data(index_df)
    district_df = clean_district_data(district_df)

    save_cleaned_data(house_df, index_df, district_df)

    return house_df, index_df, district_df

if __name__ == "__main__":
    main()
