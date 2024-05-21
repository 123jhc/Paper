import torch
import torch.nn as nn
import torch.optim as optim

class CNN_TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(CNN_TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=5)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((max_length - 5 + 1) // 2), 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Conv1d要求输入的维度是(batch_size, embedding_dim, sequence_length)
        conv_out = self.conv1d(embedded)
        pooled = self.maxpool1d(conv_out)
        flattened = self.flatten(pooled)
        out = self.fc1(flattened)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# 示例数据
vocab_size = 10000  # 词汇表大小
embedding_dim = 100  # 词向量维度
max_length = 100  # 最大句子长度

# 初始化模型
model = CNN_TextClassifier(vocab_size, embedding_dim, max_length)
print(model)
# 统计模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")