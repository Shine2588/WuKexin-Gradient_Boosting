# ====================================
# Gradient Boosting回归实验
# California Housing Dataset
# 任务：加州房价预测
# ====================================

# 1. 导入相关库
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ====================================
# 2. 数据加载
# ====================================

data = fetch_california_housing()

X = data.data
y = data.target

# 构造DataFrame，方便查看数据
df = pd.DataFrame(
    X,
    columns=data.feature_names
)

df["target"] = y

print("数据集维度：")
print(df.shape)

print("\n前5行数据：")
print(df.head())

print("\n目标变量基本统计信息：")
print(df["target"].describe())

# ====================================
# 3. 数据预处理
# ====================================

# 3.1 检查缺失值
print("\n缺失值统计：")
print(df.isnull().sum())

# 3.2 划分特征变量和目标变量
X = df.drop("target", axis=1)
y = df["target"]

# 3.3 划分训练集和测试集
# 80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))

# 3.4 特征标准化
# 虽然树模型对特征尺度不敏感，
# 但为了保持实验流程一致性，这里仍进行标准化处理

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================================
# 4. 构建Gradient Boosting回归模型
# ====================================

gb_reg = GradientBoostingRegressor(
    n_estimators=200,      # 弱学习器数量
    learning_rate=0.1,     # 学习率
    max_depth=3,           # 每棵树最大深度
    random_state=42
)

# ====================================
# 5. 模型训练
# ====================================

gb_reg.fit(X_train_scaled, y_train)

# ====================================
# 6. 模型预测
# ====================================

y_pred = gb_reg.predict(X_test_scaled)

# ====================================
# 7. 模型评价
# ====================================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("Gradient Boosting回归结果")
print("==============================")

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R^2  : {r2:.4f}")

# ====================================
# 8. 特征重要性分析
# ====================================

importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": gb_reg.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n特征重要性排序：")
print(importance)

# ====================================
# 9. 预测结果示例
# ====================================

result = pd.DataFrame({
    "True Value": y_test.values[:10],
    "Predicted Value": y_pred[:10],
    "Error": y_test.values[:10] - y_pred[:10]
})

print("\n预测结果示例：")
print(result)