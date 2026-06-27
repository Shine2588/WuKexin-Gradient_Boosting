# ====================================
# Gradient Boosting 与 Linear Regression 回归对比实验
# 数据集：California Housing
# 任务：房价预测
# ====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ====================================
# 1. 加载数据
# ====================================

data = fetch_california_housing()

X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

print("数据集维度：")
print(df.shape)

print("\n缺失值统计：")
print(df.isnull().sum())

print("\n目标变量统计信息：")
print(df["target"].describe())

# ====================================
# 2. 数据预处理
# ====================================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))

# ====================================
# 3. 构建模型
# ====================================

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

lr_model = LinearRegression()

# ====================================
# 4. 模型训练
# ====================================

gb_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# ====================================
# 5. 模型预测
# ====================================

gb_pred = gb_model.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test_scaled)

# ====================================
# 6. 模型评价函数
# ====================================

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mae, r2


gb_mse, gb_rmse, gb_mae, gb_r2 = regression_metrics(
    y_test,
    gb_pred
)

lr_mse, lr_rmse, lr_mae, lr_r2 = regression_metrics(
    y_test,
    lr_pred
)

# ====================================
# 7. 输出结果对比表
# ====================================

result_df = pd.DataFrame({
    "Model": [
        "Gradient Boosting",
        "Linear Regression"
    ],
    "MSE": [
        gb_mse,
        lr_mse
    ],
    "RMSE": [
        gb_rmse,
        lr_rmse
    ],
    "MAE": [
        gb_mae,
        lr_mae
    ],
    "R2": [
        gb_r2,
        lr_r2
    ]
})

print("\n模型回归性能对比：")
print(result_df)

# ====================================
# 8. 指标对比柱状图
# ====================================

metrics = ["MSE", "RMSE", "MAE", "R2"]

gb_values = [
    gb_mse,
    gb_rmse,
    gb_mae,
    gb_r2
]

lr_values = [
    lr_mse,
    lr_rmse,
    lr_mae,
    lr_r2
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    gb_values,
    width,
    label="Gradient Boosting"
)

plt.bar(
    x + width / 2,
    lr_values,
    width,
    label="Linear Regression"
)

plt.xticks(x, metrics)
plt.ylabel("Metric Value")
plt.title("Regression Metrics Comparison")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("regression_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ====================================
# 9. 真实值与预测值对比折线图
# ====================================

# 为了图像清晰，只展示前100个测试样本
sample_num = 100

plt.figure(figsize=(12, 6))

plt.plot(
    range(sample_num),
    y_test.values[:sample_num],
    label="True Value",
    linestyle="-"
)

plt.plot(
    range(sample_num),
    gb_pred[:sample_num],
    label="Gradient Boosting Prediction",
    linestyle="--"
)

plt.plot(
    range(sample_num),
    lr_pred[:sample_num],
    label="Linear Regression Prediction",
    linestyle=":"
)

plt.xlabel("Test Sample Index")
plt.ylabel("House Value")
plt.title("True Values vs Predicted Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_prediction_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ====================================
# 10. 预测误差对比折线图
# ====================================

gb_error = y_test.values - gb_pred
lr_error = y_test.values - lr_pred

plt.figure(figsize=(12, 6))

plt.plot(
    range(sample_num),
    gb_error[:sample_num],
    label="Gradient Boosting Error"
)

plt.plot(
    range(sample_num),
    lr_error[:sample_num],
    label="Linear Regression Error"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Test Sample Index")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_error_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ====================================
# 11. 特征重要性对比补充
# ====================================

gb_importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": gb_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nGradient Boosting特征重要性：")
print(gb_importance)
