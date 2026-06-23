# ====================================
# Gradient Boosting分类实验
# Breast Cancer Wisconsin Dataset
# ====================================

# 导入库
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ====================================
# 1. 数据加载
# ====================================

data = load_breast_cancer()

X = data.data
y = 1 - data.target

# 构造DataFrame方便查看
df = pd.DataFrame(
    X,
    columns=data.feature_names
)

df["target"] = y

print("数据集维度：")
print(df.shape)

print("\n前5行数据：")
print(df.head())

print("\n类别分布：")
print(df["target"].value_counts())

# ====================================
# 2. 数据预处理
# ====================================

# 检查缺失值

print("\n缺失值统计：")
print(df.isnull().sum().sum())

# 特征和标签

X = df.drop("target", axis=1)
y = df["target"]

# 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))

# 特征标准化

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================
# 3. 构建Gradient Boosting模型
# ====================================

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# ====================================
# 4. 模型训练
# ====================================

gb_model.fit(X_train, y_train)

# ====================================
# 5. 模型预测
# ====================================

y_pred = gb_model.predict(X_test)

# 恶性肿瘤概率

y_prob = gb_model.predict_proba(X_test)[:, 1]

# ====================================
# 6. 模型评价
# ====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n==============================")
print("Gradient Boosting分类结果")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ====================================
# 7. 混淆矩阵
# ====================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n混淆矩阵：")
print(cm)

# ====================================
# 8. 分类报告
# ====================================

print("\n分类报告：")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )
)

# ====================================
# 9. 特征重要性
# ====================================

importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": gb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n前10个重要特征：")

print(
    importance.head(10)
)

# ====================================
# 10. 输出预测概率示例
# ====================================

result = pd.DataFrame({
    "True Label": y_test.values[:10],
    "Predicted Label": y_pred[:10],
    "Malignant Probability": y_prob[:10]
})

print("\n预测结果示例：")
print(result)