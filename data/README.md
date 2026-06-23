# 数据集说明

本项目使用两个公开数据集，分别用于验证 Gradient Boosting 算法在分类任务和回归任务中的应用效果。

## 1. Breast Cancer Wisconsin (Diagnostic) 数据集

### 数据来源

UCI Machine Learning Repository

### 数据集链接

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic


## 2. California Housing 数据集

### 数据来源

Scikit-Learn 官方数据集

### 数据集说明链接

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 获取方式

该数据集可通过 Scikit-Learn 直接加载：

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

