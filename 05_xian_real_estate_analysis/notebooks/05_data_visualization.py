import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """
    加载数据
    """
    house_df = pd.read_csv("../data/cleaned_house_data.csv")
    index_df = pd.read_csv("../data/cleaned_index_data.csv")
    index_df['month'] = pd.to_datetime(index_df['month'])
    district_df = pd.read_csv("../data/cleaned_district_data.csv")
    feature_importance = pd.read_csv("../data/feature_importance.csv")
    model_comparison = pd.read_csv("../models/model_comparison.csv")

    return house_df, index_df, district_df, feature_importance, model_comparison

def plot_price_trend(index_df):
    """
    绘制房价趋势图
    """
    print("绘制房价趋势图...")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(index_df['month'], index_df['new_house_index'], 'b-', linewidth=2, label='新房价格指数')
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('新房价格指数 (上月=100)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(index_df['month'], index_df['old_house_index'], 'r-', linewidth=2, label='二手房价格指数')
    ax2.set_ylabel('二手房价格指数 (上月=100)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('西安住宅价格指数趋势 (2017-2026)', fontsize=14)
    fig.tight_layout()
    plt.savefig('../visualizations/price_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: price_trend.png")

def plot_district_comparison(house_df, district_df):
    """
    绘制区域对比图
    """
    print("绘制区域对比图...")

    district_stats = house_df.groupby('district').agg({
        'unit_price': ['mean', 'median', 'std'],
        'total_price': 'mean',
        'area': 'mean'
    }).round(0)
    district_stats.columns = ['均价', '中位数', '标准差', '平均总价', '平均面积']
    district_stats = district_stats.sort_values('均价', ascending=False).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(district_stats)))

    axes[0, 0].bar(district_stats['district'], district_stats['均价'], color=colors)
    axes[0, 0].set_title('各区域二手房均价', fontsize=12)
    axes[0, 0].set_xlabel('区域')
    axes[0, 0].set_ylabel('单价 (元/㎡)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    district_order = district_stats['district'].head(6).tolist()
    data_to_plot = [house_df[house_df['district'] == d]['unit_price'].values for d in district_order]
    bp = axes[0, 1].boxplot(data_to_plot, labels=district_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:6]):
        patch.set_facecolor(color)
    axes[0, 1].set_title('各区域房价分布 (Top 6)', fontsize=12)
    axes[0, 1].set_xlabel('区域')
    axes[0, 1].set_ylabel('单价 (元/㎡)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 0].bar(district_stats['district'], district_stats['平均总价'], color='steelblue')
    axes[1, 0].set_title('各区域平均总价', fontsize=12)
    axes[1, 0].set_xlabel('区域')
    axes[1, 0].set_ylabel('总价 (万元)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    axes[1, 1].bar(district_stats['district'], district_stats['平均面积'], color='coral')
    axes[1, 1].set_title('各区域平均面积', fontsize=12)
    axes[1, 1].set_xlabel('区域')
    axes[1, 1].set_ylabel('面积 (㎡)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('../visualizations/district_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: district_comparison.png")

def plot_feature_importance(feature_importance):
    """
    绘制特征重要性图
    """
    print("绘制特征重要性图...")

    fig, ax = plt.subplots(figsize=(10, 6))

    feature_importance_sorted = feature_importance.sort_values('importance_score', ascending=True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_sorted)))
    ax.barh(feature_importance_sorted['feature'], feature_importance_sorted['importance_score'], color=colors)

    ax.set_title('房价预测特征重要性 Top 10', fontsize=14)
    ax.set_xlabel('重要性分数', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)

    plt.tight_layout()
    plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: feature_importance.png")

def plot_model_comparison(model_comparison):
    """
    绘制模型对比图
    """
    print("绘制模型对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_comparison)))

    axes[0].bar(model_comparison['model_name'], model_comparison['R2'], color=colors)
    axes[0].set_title('模型R²评分对比', fontsize=12)
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel('R² 评分')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(model_comparison['R2']):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(model_comparison['model_name'], model_comparison['RMSE'], color=colors)
    axes[1].set_title('模型RMSE对比', fontsize=12)
    axes[1].set_xlabel('模型')
    axes[1].set_ylabel('RMSE (元/㎡)')
    for i, v in enumerate(model_comparison['RMSE']):
        axes[1].text(i, v + 20, f'{v:.0f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('../visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: model_comparison.png")

def plot_price_distribution(house_df):
    """
    绘制价格分布图
    """
    print("绘制价格分布图...")

    house_df['age'] = 2026 - house_df['year_built']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(house_df['unit_price'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('单价分布', fontsize=12)
    axes[0, 0].set_xlabel('单价 (元/㎡)')
    axes[0, 0].set_ylabel('频数')

    axes[0, 1].hist(house_df['total_price'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('总价分布', fontsize=12)
    axes[0, 1].set_xlabel('总价 (万元)')
    axes[0, 1].set_ylabel('频数')

    axes[1, 0].hist(house_df['area'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('面积分布', fontsize=12)
    axes[1, 0].set_xlabel('面积 (㎡)')
    axes[1, 0].set_ylabel('频数')

    axes[1, 1].hist(house_df['age'], bins=20, color='gold', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('房龄分布', fontsize=12)
    axes[1, 1].set_xlabel('房龄 (年)')
    axes[1, 1].set_ylabel('频数')

    plt.tight_layout()
    plt.savefig('../visualizations/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: price_distribution.png")

def plot_correlation_heatmap(house_df):
    """
    绘制相关性热力图
    """
    print("绘制相关性热力图...")

    house_df['age'] = 2026 - house_df['year_built']

    numeric_cols = ['area', 'floor', 'total_floors', 'total_price', 'unit_price', 'year_built', 'age']
    corr_matrix = house_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)

    ax.set_title('房价影响因素相关性矩阵', fontsize=14)

    plt.tight_layout()
    plt.savefig('../visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: correlation_heatmap.png")

def plot_scatter_matrix(house_df):
    """
    绘制散点矩阵图
    """
    print("绘制散点矩阵图...")

    house_df['age'] = 2026 - house_df['year_built']
    plot_df = house_df[['area', 'unit_price', 'floor', 'age', 'district']].head(100)

    g = sns.pairplot(plot_df, hue='district', diag_kind='kde', palette='husl')
    g.fig.suptitle('房价影响因素散点矩阵', y=1.02, fontsize=14)

    plt.savefig('../visualizations/scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: scatter_matrix.png")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    house_df, index_df, district_df, feature_importance, model_comparison = load_data()

    plot_price_trend(index_df)
    plot_district_comparison(house_df, district_df)
    plot_feature_importance(feature_importance)
    plot_model_comparison(model_comparison)
    plot_price_distribution(house_df)
    plot_correlation_heatmap(house_df)
    plot_scatter_matrix(house_df)

    print("\n" + "=" * 60)
    print("可视化图表生成完成")
    print("=" * 60)
    print("\n生成的图表文件:")
    print("  - price_trend.png (房价趋势图)")
    print("  - district_comparison.png (区域对比图)")
    print("  - feature_importance.png (特征重要性图)")
    print("  - model_comparison.png (模型对比图)")
    print("  - price_distribution.png (价格分布图)")
    print("  - correlation_heatmap.png (相关性热力图)")
    print("  - scatter_matrix.png (散点矩阵图)")

if __name__ == "__main__":
    main()
