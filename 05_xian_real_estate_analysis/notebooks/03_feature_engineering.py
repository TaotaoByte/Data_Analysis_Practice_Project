import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """
    加载清洗后的数据
    """
    house_df = pd.read_csv("../data/cleaned_house_data.csv")
    return house_df

def create_time_features(df):
    """
    创建时间相关特征
    """
    df['age'] = 2026 - df['year_built']
    df['age_range'] = pd.cut(df['age'], bins=[0, 5, 10, 15, 20, 100],
                              labels=['新房(0-5年)', '次新房(5-10年)', '中等房龄(10-15年)', '老房(15-20年)', '老旧房(20年+)'])

    df['floor_ratio'] = df['floor'] / df['total_floors']
    df['floor_ratio_range'] = pd.cut(df['floor_ratio'], bins=[0, 0.2, 0.5, 0.8, 1],
                                      labels=['低楼层', '中低楼层', '中高楼层', '高楼层'])

    return df

def create_area_features(df):
    """
    创建面积相关特征
    """
    df['area_range'] = pd.cut(df['area'], bins=[0, 60, 90, 120, 150, 500],
                               labels=['小户型(60以下)', '中小户型(60-90)', '中户型(90-120)', '中大户型(120-150)', '大户型(150+)'])

    rooms = df['layout'].str.extract(r'(\d+)室')[0].astype(float)
    df['area_per_room'] = df['area'] / rooms

    return df

def create_layout_features(df):
    """
    创建户型相关特征
    """
    rooms = df['layout'].str.extract(r'(\d+)室')[0].astype(int)
    halls = df['layout'].str.extract(r'(\d+)厅')[0].astype(int)
    df['rooms'] = rooms
    df['halls'] = halls
    df['rooms_halls_ratio'] = df['rooms'] / df['halls']

    return df

def create_encoding_features(df):
    """
    创建编码特征
    """
    le_district = LabelEncoder()
    df['district_encoded'] = le_district.fit_transform(df['district'])

    le_decoration = LabelEncoder()
    df['decoration_encoded'] = le_decoration.fit_transform(df['decoration'])

    le_building = LabelEncoder()
    df['building_type_encoded'] = le_building.fit_transform(df['building_type'])

    le_elevator = LabelEncoder()
    df['elevator_encoded'] = le_elevator.fit_transform(df['elevator'])

    le_orientation = LabelEncoder()
    df['orientation_encoded'] = le_orientation.fit_transform(df['orientation'])

    return df

def create_interaction_features(df):
    """
    创建交互特征
    """
    df['district_decoration'] = df['district_encoded'] * 10 + df['decoration_encoded']
    df['area_price_per_room'] = df['area'] * df['decoration_encoded']
    df['floor_elevator'] = df['floor_ratio'] * df['elevator_encoded']

    return df

def feature_selection(X, y, k=10):
    """
    特征选择
    """
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    scores = selector.scores_[selector.get_support()]
    feature_importance = list(zip(selected_features, scores))
    return selected_features, feature_importance

def standardize_features(X):
    """
    特征标准化
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler

def main():
    """
    主函数
    """
    print("=" * 60)
    print("特征工程")
    print("=" * 60)

    df = load_cleaned_data()
    print(f"\n原始数据形状: {df.shape}")

    df = create_time_features(df)
    print("时间特征创建完成")

    df = create_area_features(df)
    print("面积特征创建完成")

    df = create_layout_features(df)
    print("户型特征创建完成")

    df = create_encoding_features(df)
    print("编码特征创建完成")

    df = create_interaction_features(df)
    print("交互特征创建完成")

    print(f"\n特征工程后数据形状: {df.shape}")
    print(f"\n新增特征列表:")
    new_features = ['age', 'age_range', 'floor_ratio', 'floor_ratio_range', 'area_range',
                     'area_per_room', 'rooms', 'halls', 'rooms_halls_ratio',
                     'district_encoded', 'decoration_encoded', 'building_type_encoded',
                     'elevator_encoded', 'orientation_encoded', 'district_decoration',
                     'area_price_per_room', 'floor_elevator']
    for f in new_features:
        print(f"  - {f}")

    feature_columns = ['area', 'floor', 'total_floors', 'year_built', 'age', 'floor_ratio',
                       'area_per_room', 'rooms', 'halls', 'rooms_halls_ratio',
                       'district_encoded', 'decoration_encoded', 'building_type_encoded',
                       'elevator_encoded', 'orientation_encoded', 'district_decoration',
                       'area_price_per_room', 'floor_elevator']

    X = df[feature_columns]
    y = df['unit_price']

    print("\n" + "=" * 60)
    print("特征选择")
    print("=" * 60)

    selected_features, feature_importance = feature_selection(X, y, k=10)

    print("\n选择的Top 10特征:")
    for i, (feature, score) in enumerate(sorted(feature_importance, key=lambda x: x[1], reverse=True)):
        print(f"  {i+1}. {feature}: {score:.2f}")

    print("\n" + "=" * 60)
    print("特征标准化")
    print("=" * 60)

    X_selected = X[selected_features]
    X_scaled, scaler = standardize_features(X_selected)
    print(f"\n标准化后特征形状: {X_scaled.shape}")

    print("\n" + "=" * 60)
    print("保存特征工程数据")
    print("=" * 60)

    df.to_csv("../data/features_engineered_data.csv", index=False)
    print("特征工程数据已保存: ../data/features_engineered_data.csv")

    feature_info = pd.DataFrame({
        'feature': selected_features,
        'importance_score': [score for _, score in feature_importance]
    })
    feature_info.to_csv("../data/feature_importance.csv", index=False)
    print("特征重要性已保存: ../data/feature_importance.csv")

    return df, selected_features, X_scaled

if __name__ == "__main__":
    main()
