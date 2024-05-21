import pandas as pd
import torch
import numpy as np

# 从Excel文件中读取数据
excel_file = 'Financial texts-data.xlsx'
df = pd.read_excel(excel_file)


# 获取特征列
features = df['Business_Status'].values

# 获取标签列
labels = df['Label'].values

print("第一个样本数据：")
print(features[0])

# 将特征字符串转换为词向量
# 假设你有一个词向量的字典，可以根据特征字符串获取对应的词向量
# 这里只是一个示例，你需要根据自己的实际情况来替换这部分代码



