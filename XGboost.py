import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def read_excel(file_path, y_value):

    # 读取
    selected_data = pd.read_excel(file_path)  
    # 选择指定的列
    selected_data = selected_data.iloc[:, 2:]  # 选择第3列到第9列，Python索引从0开始

    # 替换'--'为NaN
    selected_data.replace('--', np.nan, inplace=True)

    # 计算每列的平均值
    row_means = selected_data.mean(axis=0)

    # 填充NaN为每行的平均值
    selected_data.fillna(row_means, axis=0, inplace=True)

    # 将DataFrame转换为NumPy数组
    selected_data = np.array(selected_data)

    # 创建y数组，填充指定的值，数据类型为整数
    y = np.full(selected_data.shape[0], y_value, dtype=int)

    return selected_data, y
    # print(y.size)       # 输出y的第43个元素
    # print(selected_data[42])


#非st数据
file_path_0 = "./data.xlsx"
x_0 , y_0 = read_excel(file_path_0, 0)
# print(x_0)
# print(y_0)
#st数据
file_path_1 = "./data_st.xlsx"
x_1 , y_1 = read_excel(file_path_1, 1)


# 合并数据和标签
X = np.concatenate((x_0, x_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

# 划分数据集，按照7:3的比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 创建XGBoost分类器
xgb_classifier = XGBClassifier()

# 在训练集上训练模型
xgb_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = xgb_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算精确率
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# 计算召回率
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# 计算F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# 计算AUC值
y_prob = xgb_classifier.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)

# 绘制ROC曲线图
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
