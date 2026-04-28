import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 加载特征工程后的数据
def load_features_data():
    """
    加载特征工程后的数据
    返回: 特征工程后的数据DataFrame
    """
    file_path = "../data/features_sales_data.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载特征工程后的数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 准备训练数据
def prepare_training_data(df):
    """
    准备训练数据
    参数: df - 特征工程后的数据DataFrame
    返回: X_train, X_test, y_train, y_test
    """
    # 选择特征和目标变量
    feature_columns = ['quantity', 'price', 'day_of_week', 'is_weekend', 
                      'category_encoded', 'product_encoded', 'region_encoded', 
                      'category_region_interaction']
    
    X = df[feature_columns]
    y = df['sales']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# 训练线性回归模型
def train_linear_regression(X_train, y_train):
    """
    训练线性回归模型
    参数: X_train, y_train - 训练数据
    返回: 训练好的模型
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 训练随机森林模型
def train_random_forest(X_train, y_train):
    """
    训练随机森林模型
    参数: X_train, y_train - 训练数据
    返回: 训练好的模型
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 训练梯度提升模型
def train_gradient_boosting(X_train, y_train):
    """
    训练梯度提升模型
    参数: X_train, y_train - 训练数据
    返回: 训练好的模型
    """
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 模型评估
def evaluate_model(model, X_test, y_test, model_name):
    """
    模型评估
    参数: 
        model - 训练好的模型
        X_test, y_test - 测试数据
        model_name - 模型名称
    返回: 评估指标字典
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        '模型': model_name,
        '均方误差 (MSE)': mse,
        '均方根误差 (RMSE)': rmse,
        '平均绝对误差 (MAE)': mae,
        'R² 评分': r2
    }
    
    print(f"\n{model_name} 模型评估结果:")
    for key, value in metrics.items():
        if key != '模型':
            print(f"{key}: {value:.4f}")
    
    return metrics

# 模型优化
def optimize_model(X_train, y_train):
    """
    模型优化
    参数: X_train, y_train - 训练数据
    返回: 优化后的模型
    """
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证评分: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 主函数
def main():
    """
    主函数，执行建模分析流程
    """
    # 加载数据
    df = load_features_data()
    if df is None:
        return
    
    # 准备训练数据
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    
    # 训练多个模型
    models = {
        '线性回归': train_linear_regression(X_train, y_train),
        '随机森林': train_random_forest(X_train, y_train),
        '梯度提升': train_gradient_boosting(X_train, y_train)
    }
    
    # 评估所有模型
    metrics_list = []
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        metrics_list.append(metrics)
    
    # 模型比较
    metrics_df = pd.DataFrame(metrics_list)
    print("\n模型性能比较:")
    print(metrics_df.sort_values('R² 评分', ascending=False))
    
    # 模型优化
    optimized_model = optimize_model(X_train, y_train)
    
    # 评估优化后的模型
    optimize_metrics = evaluate_model(optimized_model, X_test, y_test, "优化后的随机森林")
    
    # 保存最佳模型
    best_model = optimized_model
    model_path = "../models/best_sales_prediction_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n最佳模型已保存至: {model_path}")
    
    return best_model

if __name__ == "__main__":
    main()
