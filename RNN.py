import torch
import torch.nn as nn
import torch.optim as optim

class RNN_TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN_TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.fc1(rnn_out[:, -1, :])  # 使用最后一个时间步的隐藏状态作为特征
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# 示例数据
vocab_size = 10000  # 词汇表大小
embedding_dim = 100  # 词向量维度
hidden_dim = 128  # RNN隐藏层维度
output_dim = 8  # 输出维度

# 初始化模型
model = RNN_TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# print(model_rnn)
print(model)

# 统计模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")