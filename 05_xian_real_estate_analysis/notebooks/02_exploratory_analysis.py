import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_cleaned_data():
    """
    加载清洗后的数据
    """
    house_df = pd.read_csv("../data/cleaned_house_data.csv")
    index_df = pd.read_csv("../data/cleaned_index_data.csv")
    index_df['month'] = pd.to_datetime(index_df['month'])
    district_df = pd.read_csv("../data/cleaned_district_data.csv")

    return house_df, index_df, district_df

def univariate_analysis(house_df, index_df):
    """
    单变量分析
    """
    print("=" * 60)
    print("单变量分析")
    print("=" * 60)

    print("\n1. 二手房单价分布统计:")
    print(house_df['unit_price'].describe())

    print("\n2. 二手房总价分布统计:")
    print(house_df['total_price'].describe())

    print("\n3. 二手房面积分布统计:")
    print(house_df['area'].describe())

    print("\n4. 户型分布:")
    print(house_df['layout'].value_counts().head(10))

    print("\n5. 区域分布:")
    print(house_df['district'].value_counts())

    print("\n6. 装修程度分布:")
    print(house_df['decoration'].value_counts())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(house_df['unit_price'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('二手房单价分布', fontsize=14)
    axes[0, 0].set_xlabel('单价 (元/㎡)')
    axes[0, 0].set_ylabel('频数')

    axes[0, 1].hist(house_df['total_price'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('二手房总价分布', fontsize=14)
    axes[0, 1].set_xlabel('总价 (万元)')
    axes[0, 1].set_ylabel('频数')

    axes[1, 0].hist(house_df['area'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title('二手房面积分布', fontsize=14)
    axes[1, 0].set_xlabel('面积 (㎡)')
    axes[1, 0].set_ylabel('频数')

    layout_counts = house_df['layout'].value_counts().head(8)
    axes[1, 1].barh(layout_counts.index, layout_counts.values, color='steelblue')
    axes[1, 1].set_title('户型分布', fontsize=14)
    axes[1, 1].set_xlabel('数量')

    plt.tight_layout()
    plt.savefig('../visualizations/univariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n图表已保存: ../visualizations/univariate_analysis.png")

def bivariate_analysis(house_df, district_df):
    """
    双变量分析
    """
    print("\n" + "=" * 60)
    print("双变量分析")
    print("=" * 60)

    print("\n1. 各区域二手房均价:")
    district_avg = house_df.groupby('district')['unit_price'].mean().sort_values(ascending=False)
    print(district_avg.round(0))

    print("\n2. 各区域二手房总价均值:")
    district_total = house_df.groupby('district')['total_price'].mean().sort_values(ascending=False)
    print(district_total.round(0))

    print("\n3. 装修程度与单价关系:")
    decoration_price = house_df.groupby('decoration')['unit_price'].mean()
    print(decoration_price.round(0))

    print("\n4. 房龄与单价关系:")
    house_df['age'] = 2026 - house_df['year_built']
    age_corr = house_df[['age', 'unit_price']].corr().iloc[0, 1]
    print(f"房龄与单价相关系数: {age_corr:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    district_avg = house_df.groupby('district')['unit_price'].mean().sort_values(ascending=False)
    axes[0, 0].bar(district_avg.index, district_avg.values, color='coral')
    axes[0, 0].set_title('各区域二手房均价', fontsize=14)
    axes[0, 0].set_xlabel('区域')
    axes[0, 0].set_ylabel('单价 (元/㎡)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    district_total = house_df.groupby('district')['total_price'].mean().sort_values(ascending=False)
    axes[0, 1].bar(district_total.index, district_total.values, color='skyblue')
    axes[0, 1].set_title('各区域二手房总价均值', fontsize=14)
    axes[0, 1].set_xlabel('区域')
    axes[0, 1].set_ylabel('总价 (万元)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    decoration_price = house_df.groupby('decoration')['unit_price'].mean()
    axes[1, 0].bar(decoration_price.index, decoration_price.values, color='lightgreen')
    axes[1, 0].set_title('装修程度与单价关系', fontsize=14)
    axes[1, 0].set_xlabel('装修程度')
    axes[1, 0].set_ylabel('单价 (元/㎡)')

    scatter = axes[1, 1].scatter(house_df['area'], house_df['unit_price'],
                                   c=house_df['age'], cmap='coolwarm', alpha=0.6)
    axes[1, 1].set_title('面积与单价关系 (颜色=房龄)', fontsize=14)
    axes[1, 1].set_xlabel('面积 (㎡)')
    axes[1, 1].set_ylabel('单价 (元/㎡)')
    plt.colorbar(scatter, ax=axes[1, 1], label='房龄(年)')

    plt.tight_layout()
    plt.savefig('../visualizations/bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n图表已保存: ../visualizations/bivariate_analysis.png")

def multivariate_analysis(house_df, index_df, district_df):
    """
    多变量分析
    """
    print("\n" + "=" * 60)
    print("多变量分析")
    print("=" * 60)

    print("\n1. 数值变量相关性矩阵:")
    numeric_cols = ['area', 'floor', 'total_floors', 'total_price', 'unit_price', 'year_built']
    corr_matrix = house_df[numeric_cols].corr()
    print(corr_matrix.round(3))

    print("\n2. 房价指数趋势分析:")
    index_df['year'] = index_df['month'].dt.year
    yearly_avg = index_df.groupby('year')[['new_house_index', 'old_house_index']].mean()
    print(yearly_avg)

    print("\n3. 区域-装修交叉分析:")
    cross_tab = pd.crosstab(house_df['district'], house_df['decoration'], values=house_df['unit_price'], aggfunc='mean')
    print(cross_tab.round(0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    corr_matrix = house_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0, 0], fmt='.2f')
    axes[0, 0].set_title('数值变量相关性矩阵', fontsize=14)

    axes[0, 1].plot(index_df['month'], index_df['old_house_index'], label='二手房指数', color='coral', linewidth=2)
    axes[0, 1].set_title('西安二手房价格指数趋势', fontsize=14)
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('价格指数 (上月=100)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    yearly_avg = index_df.groupby('year')[['new_house_index', 'old_house_index']].mean()
    x = range(len(yearly_avg))
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], yearly_avg['new_house_index'], width, label='新房指数', color='steelblue')
    axes[1, 0].bar([i + width/2 for i in x], yearly_avg['old_house_index'], width, label='二手房指数', color='coral')
    axes[1, 0].set_title('年度房价指数对比', fontsize=14)
    axes[1, 0].set_xlabel('年份')
    axes[1, 0].set_ylabel('价格指数')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(yearly_avg.index)
    axes[1, 0].legend()

    district_decoration = house_df.pivot_table(values='unit_price', index='district', columns='decoration', aggfunc='mean')
    district_decoration.plot(kind='bar', ax=axes[1, 1], colormap='viridis')
    axes[1, 1].set_title('各区域不同装修程度单价对比', fontsize=14)
    axes[1, 1].set_xlabel('区域')
    axes[1, 1].set_ylabel('单价 (元/㎡)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='装修程度')

    plt.tight_layout()
    plt.savefig('../visualizations/multivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n图表已保存: ../visualizations/multivariate_analysis.png")

def main():
    """
    主函数
    """
    house_df, index_df, district_df = load_cleaned_data()

    univariate_analysis(house_df, index_df)

    house_df['age'] = 2026 - house_df['year_built']
    bivariate_analysis(house_df, district_df)

    multivariate_analysis(house_df, index_df, district_df)

    print("\n" + "=" * 60)
    print("探索性数据分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
