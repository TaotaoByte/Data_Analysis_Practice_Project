import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("线性回归 - 房价预测")
print("=" * 50)

df = pd.read_csv('../data/housing_data.csv')
print(f"\n数据形状: {df.shape}")
print("\n前5行数据:")
print(df.head())

print("\n数据基本信息:")
print(df.info())
print("\n描述性统计:")
print(df.describe())

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='area', y='price')
plt.title('面积与房价关系')

plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='bedrooms', y='price')
plt.title('卧室数量与房价关系')

plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='age', y='price')
plt.title('房龄与房价关系')

plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('特征相关性热力图')

plt.tight_layout()
plt.savefig('../data/visualization.png')
print("\n可视化图表已保存到 data/visualization.png")

X = df[['area', 'bedrooms', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n模型系数: {model.coef_}")
print(f"模型截距: {model.intercept_}")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"R² 分数: {r2:.4f}")

new_house = pd.DataFrame({
    'area': [100, 150, 80],
    'bedrooms': [3, 4, 2],
    'age': [5, 3, 10]
})

predictions = model.predict(new_house)

print("\n" + "=" * 50)
print("新房预测结果:")
print("=" * 50)
for i, pred in enumerate(predictions):
    print(f"\n房屋 {i+1}:")
    print(f"  面积: {new_house['area'][i]}㎡")
    print(f"  卧室: {new_house['bedrooms'][i]}个")
    print(f"  房龄: {new_house['age'][i]}年")
    print(f"  预测房价: {pred:.2f}万元")

print("\n" + "=" * 50)
print("分析完成!")
print("=" * 50)
