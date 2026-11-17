import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================= 设置中文字体，防止图表乱码 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

print(">>> 1. 正在读取数据...")
try:
    df_products = pd.read_excel('products.xlsx')
    df_orders = pd.read_excel('orders.xlsx')
except FileNotFoundError:
    print("错误：找不到文件！请确保 products.xlsx 和 orders.xlsx 与代码在同一个文件夹下。")
    exit()

# ================= 二、数据预处理 =================
print(">>> 2. 正在进行数据预处理...")
df_orders['订单编号'] = df_orders['订单编号'].ffill()

# 建立货架坐标映射 (用于K-Means聚类)
# 根据你们的平面图：A-D左上(1,1), E-H左下(1,2), I-L右上(2,1), M-P右下(2,2)
zone_coords = {
    'A': [1, 1], 'B': [1.2, 1], 'C': [1, 1.2], 'D': [1.2, 1.2],
    'E': [1, 2], 'F': [1.2, 2], 'G': [1, 2.2], 'H': [1.2, 2.2],
    'I': [2, 1], 'J': [2.2, 1], 'K': [2, 1.2], 'L': [2.2, 1.2],
    'M': [2, 2], 'N': [2.2, 2], 'O': [2, 2.2], 'P': [2.2, 2.2]
}

# ================= 模块 1：K-Means 聚类 (片区访问次序) =================
print("\n>>> 3. [对应报告阶段1] 正在运行 K-Means 进行商品空间聚类...")
# 提取商品货架坐标
df_products['X'] = df_products['货架编号'].map(lambda x: zone_coords.get(x, [0,0])[0])
df_products['Y'] = df_products['货架编号'].map(lambda x: zone_coords.get(x, [0,0])[1])
X_coords = df_products[['X', 'Y']]

# 聚类为 4 个片区 (A-D, E-H, I-L, M-P)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_products['片区聚类'] = kmeans.fit_predict(X_coords)

# 打印聚类中心结果
print("【K-Means 聚类中心】(对应物理片区):")
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    zone_name = "未知"
    if center[0] < 1.5 and center[1] < 1.5: zone_name = "蔬果区 A-D"
    elif center[0] < 1.5 and center[1] >= 1.5: zone_name = "日用品区 E-H"
    elif center[0] >= 1.5 and center[1] < 1.5: zone_name = "冷藏区 I-L"
    else: zone_name = "零食区 M-P"
    print(f"类别 {i}: 坐标({center[0]:.1f}, {center[1]:.1f}) -> 映射至: {zone_name}")

# ================= 模块 2：Apriori 关联规则 (片区内部拣货) =================
print("\n>>> 4. [对应报告阶段1&3] 正在运行 Apriori 优化拣货次序...")
basket = (df_orders.groupby(['订单编号', '商品名称'])['商品ID']
          .count().unstack().reset_index().fillna(0).set_index('订单编号'))
basket_sets = basket.map(lambda x: True if x >= 1 else False)

frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
top_rules = rules.sort_values(by='lift', ascending=False).head(5)

print("【挖掘出的Top 5强关联规则】(用于指导同片区连续拣取):")
for index, row in top_rules.iterrows():
    antecedents = ", ".join(list(row['antecedents']))
    consequents = ", ".join(list(row['consequents']))
    print(f"如果买了 [{antecedents}] -> 优先拣取 [{consequents}] (提升度:{row['lift']:.2f})")

# ================= 模块 3：决策树分类 (动态避障策略) =================
print("\n>>> 5. [对应报告阶段2] 正在训练 决策树 障碍物分类模型...")
# 生成仿真障碍物数据：特征 [尺寸大小(m), 移动速度(m/s), 停留时间(s)]
np.random.seed(42)
# 0: 静态障碍物/故障货架(尺寸大, 速度0, 停留长) | 1: 动态行人(尺寸中, 速度快, 停留短) | 2: 临时杂物(尺寸小, 速度0, 停留短)
X_obs = np.vstack([
    np.random.normal([1.5, 0, 60], [0.2, 0.1, 5], (50, 3)),   # 静态障碍
    np.random.normal([0.5, 1.2, 2], [0.1, 0.2, 0.5], (50, 3)), # 动态行人
    np.random.normal([0.3, 0, 5], [0.1, 0.1, 2], (50, 3))     # 临时杂物
])
y_obs = np.array([0]*50 + [1]*50 + [2]*50)

dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_obs, y_obs)

print("【决策树避障规则提取】(提取关键规则):")
tree_rules = export_text(dt_classifier, feature_names=['尺寸', '速度', '停留时间'])
print(tree_rules[:400] + "\n... (部分截取)")

# ================= 模块 4：多元线性回归 (客流密度预测) =================
print("\n>>> 6. [对应报告阶段1&4] 正在训练 回归模型 预测片区客流...")
# 生成仿真客流数据：特征 [是否周末(0/1), 时间段(1-12早8点到晚8点)]
X_flow = []
y_flow = []
for is_weekend in [0, 1]:
    for hour in range(1, 13):
        X_flow.append([is_weekend, hour])
        # 仿真公式：工作日客流低但中午高；周末全天客流暴涨
        base = 5 if is_weekend == 0 else 20
        peak = 10 if (hour in [4,5,6]) else 0 # 中午11-1点
        noise = np.random.randint(-2, 3)
        y_flow.append(base + peak + noise)

X_flow = np.array(X_flow)
y_flow = np.array(y_flow)

lr_model = LinearRegression()
lr_model.fit(X_flow, y_flow)

# 预测周末下午2点的客流
pred_weekend_14pm = lr_model.predict([[1, 8]])[0]
pred_workday_14pm = lr_model.predict([[0, 8]])[0]
print(f"【回归模型预测】周末下午14点客流指数: {pred_weekend_14pm:.1f} (预测值较高)")
print(f"【回归模型预测】工作日14点客流指数: {pred_workday_14pm:.1f} (预测值较低)")
print("-> 结论：结合回归预测，周末时段需优先规避高拥堵片区并下调速度。")

# ================= 五、可视化出图 =================
print("\n>>> 7. 正在生成组合版可视化大图...")
plt.figure(figsize=(16, 10))

# 图1：K-Means 空间聚类
plt.subplot(2, 2, 1)
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
for i in range(4):
    cluster_data = df_products[df_products['片区聚类'] == i]
    plt.scatter(cluster_data['X'], cluster_data['Y'], c=colors[i], label=f'聚类 {i}', alpha=0.7, s=80)
plt.title('商品空间坐标 K-Means 聚类 (映射四大片区)')
plt.xlabel('X 坐标 (横向)')
plt.ylabel('Y 坐标 (纵向)')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

# 图2：Apriori 关联规则支持度置信度
plt.subplot(2, 2, 2)
scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.7, s=60)
plt.colorbar(scatter, label='提升度 (Lift)')
plt.title('Apriori 关联规则分布')
plt.xlabel('支持度 (Support)')
plt.ylabel('置信度 (Confidence)')
plt.grid(True, linestyle='--', alpha=0.5)

# 图3：决策树分类可视化
plt.subplot(2, 2, 3)
plot_tree(dt_classifier, feature_names=['尺寸', '速度', '停留时间'], 
          class_names=['静态障碍', '动态行人', '临时杂物'], filled=True, rounded=True)
plt.title('决策树避障分类模型')

# 图4：线性回归客流预测
plt.subplot(2, 2, 4)
plt.scatter(range(len(y_flow)), y_flow, color='gray', alpha=0.5, label='实际仿真数据')
plt.plot(range(len(y_flow)), lr_model.predict(X_flow), color='red', linewidth=2, label='回归模型拟合线')
plt.title('分时段客流密度多元线性回归预测')
plt.xlabel('时间序列 (工作日->周末)')
plt.ylabel('客流密度指数')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('四大算法综合图表.png', dpi=300)
print("图表已生成并保存为 '四大算法综合图表.png'，请直接插入报告！")

# 保存处理后的数据
df_products.to_excel('商品片区聚类结果.xlsx', index=False)
print("商品聚类结果已保存为 '商品片区聚类结果.xlsx'。")
print("\n>>> 完美结束！这版代码完全契合你们的新版报告逻辑！")