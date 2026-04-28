import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    加载特征工程后的数据
    """
    df = pd.read_csv("../data/features_engineered_data.csv")
    return df

def prepare_data(df):
    """
    准备训练数据
    """
    feature_columns = ['area', 'floor', 'total_floors', 'year_built', 'age', 'floor_ratio',
                       'area_per_room', 'rooms', 'halls', 'rooms_halls_ratio',
                       'district_encoded', 'decoration_encoded', 'building_type_encoded',
                       'elevator_encoded', 'orientation_encoded']

    X = df[feature_columns]
    y = df['unit_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """
    训练线性回归模型
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train):
    """
    训练岭回归模型
    """
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train):
    """
    训练LASSO回归模型
    """
    model = Lasso(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    训练随机森林模型
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """
    训练梯度提升模型
    """
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """
    评估模型
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} 模型评估结果:")
    print(f"  均方误差 (MSE): {mse:,.2f}")
    print(f"  均方根误差 (RMSE): {rmse:,.2f}")
    print(f"  平均绝对误差 (MAE): {mae:,.2f}")
    print(f"  R² 评分: {r2:.4f}")

    return {'model_name': model_name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def cross_validate_model(model, X, y, cv=5):
    """
    交叉验证
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"\n交叉验证 R² 评分: {scores}")
    print(f"平均 R² 评分: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores

def optimize_random_forest(X_train, y_train):
    """
    优化随机森林模型
    """
    print("\n" + "=" * 60)
    print("随机森林模型优化")
    print("=" * 60)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证评分: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def get_feature_importance(model, feature_names):
    """
    获取特征重要性
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    return None

def main():
    """
    主函数
    """
    print("=" * 60)
    print("建模分析")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    models = {
        '线性回归': train_linear_regression(X_train, y_train),
        '岭回归': train_ridge_regression(X_train, y_train),
        'LASSO回归': train_lasso_regression(X_train, y_train),
        '随机森林': train_random_forest(X_train, y_train),
        '梯度提升': train_gradient_boosting(X_train, y_train)
    }

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    results = []
    for model_name, model in models.items():
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)

    print("\n" + "=" * 60)
    print("模型比较")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2', ascending=False)
    print("\n模型性能排名:")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("模型优化")
    print("=" * 60)

    best_rf = optimize_random_forest(X_train, y_train)

    print("\n优化后随机森林模型评估:")
    optimize_result = evaluate_model(best_rf, X_test, y_test, "优化后随机森林")

    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)

    feature_importance = get_feature_importance(best_rf, X_train.columns.tolist())
    if feature_importance is not None:
        print("\n随机森林特征重要性 Top 10:")
        print(feature_importance.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)

    joblib.dump(best_rf, "../models/best_price_prediction_model.joblib")
    print("最佳模型已保存: ../models/best_price_prediction_model.joblib")

    results_df.to_csv("../models/model_comparison.csv", index=False)
    print("模型比较结果已保存: ../models/model_comparison.csv")

    feature_importance.to_csv("../models/feature_importance.csv", index=False)
    print("特征重要性已保存: ../models/feature_importance.csv")

    return best_rf, results_df, feature_importance

if __name__ == "__main__":
    main()
