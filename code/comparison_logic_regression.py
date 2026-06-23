# ====================================
# Gradient Boosting 与 Logistic Regression 分类对比实验
# 数据集：Breast Cancer Wisconsin
# ====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# ====================================
# 1. 加载数据
# ====================================

data = load_breast_cancer()

X = data.data

# 标签转换：
# 原始数据中 0=Malignant, 1=Benign
# 转换后 1=Malignant, 0=Benign
y = 1 - data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

print("数据集维度：")
print(df.shape)

print("\n类别分布：")
print(df["target"].value_counts())

print("\n缺失值统计：")
print(df.isnull().sum().sum())

# ====================================
# 2. 数据预处理
# ====================================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))

# ====================================
# 3. 构建模型
# ====================================

# Gradient Boosting模型
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# 对比模型：逻辑回归
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

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
# 6. 整体评价指标对比
# ====================================

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

result_df = pd.DataFrame({
    "Model": ["Gradient Boosting", "Logistic Regression"],
    "Accuracy": [gb_accuracy, lr_accuracy],
    "Precision": [gb_precision, lr_precision],
    "Recall": [gb_recall, lr_recall]
})

print("\n模型整体评价指标对比：")
print(result_df)

print("\nGradient Boosting分类报告：")
print(
    classification_report(
        y_test,
        gb_pred,
        target_names=["Benign", "Malignant"]
    )
)

print("\nLogistic Regression分类报告：")
print(
    classification_report(
        y_test,
        lr_pred,
        target_names=["Benign", "Malignant"]
    )
)

print("\nGradient Boosting混淆矩阵：")
print(confusion_matrix(y_test, gb_pred))

print("\nLogistic Regression混淆矩阵：")
print(confusion_matrix(y_test, lr_pred))

# ====================================
# 7. 计算测试集累计Precision和Recall
# ====================================

def cumulative_metrics(y_true, y_pred):
    precision_list = []
    recall_list = []

    for i in range(1, len(y_true) + 1):
        precision = precision_score(
            y_true[:i],
            y_pred[:i],
            zero_division=0
        )

        recall = recall_score(
            y_true[:i],
            y_pred[:i],
            zero_division=0
        )

        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list


y_test_array = np.array(y_test)

gb_precision_curve, gb_recall_curve = cumulative_metrics(
    y_test_array,
    gb_pred
)

lr_precision_curve, lr_recall_curve = cumulative_metrics(
    y_test_array,
    lr_pred
)

x_axis = np.arange(1, len(y_test_array) + 1)

# ====================================
# 8. 绘制Precision对比折线图
# ====================================

plt.figure(figsize=(10, 6))

plt.plot(
    x_axis,
    gb_precision_curve,
    label="Gradient Boosting Precision"
)

plt.plot(
    x_axis,
    lr_precision_curve,
    label="Logistic Regression Precision"
)

plt.xlabel("Number of Test Samples")
plt.ylabel("Cumulative Precision")
plt.title("Precision Comparison on Test Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================
# 9. 绘制Recall对比折线图
# ====================================

plt.figure(figsize=(10, 6))

plt.plot(
    x_axis,
    gb_recall_curve,
    label="Gradient Boosting Recall"
)

plt.plot(
    x_axis,
    lr_recall_curve,
    label="Logistic Regression Recall"
)

plt.xlabel("Number of Test Samples")
plt.ylabel("Cumulative Recall")
plt.title("Recall Comparison on Test Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()