import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

# 从CNN.py文件导入模型
from CNN import CNN_TextClassifier  
#processtex.py导入文本处理函数
from processtex import process_text
from processtex import process_data

# 从Excel文件中读取财务文本数据
excel_file = 'Financial texts-data.xlsx'
df = pd.read_excel(excel_file)
# 获取特征列
features = df['Business_Status'].values
# 获取标签列
labels = df['Label'].values

# print("第一份样本数据：")
# print(features[0])

# 财务文本数据处理
num_samples = 200
features, labels = process_text(features, labels, num_samples)

# 初始化模型
vocab_size = 10000  # 假设词汇表大小为10000
embedding_dim = 100  # 假设词向量维度为100
max_length = 10  # 假设特征字符串长度为10
model = CNN_TextClassifier(vocab_size, embedding_dim, max_length)

# 假设你有一个词汇表，将特征字符串映射为索引
# 这里只是一个示例，你需要根据自己的情况来构建这个映射
vocab = list(set(''.join(features)))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# 处理数据
features_tensor, labels_tensor = process_data(features, labels, word_to_idx)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
train_losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(features_tensor)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    time.sleep(3)  # 等待5秒钟
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 绘制训练损失函数图
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# 预测
with torch.no_grad():
    predicted = model(features_tensor).argmax(dim=1)
    accuracy = (predicted == labels_tensor).sum().item() / len(labels_tensor)
    print(f"Accuracy: {accuracy}")
