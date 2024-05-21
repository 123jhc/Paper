import torch
import numpy as np
from collections import Counter

#数据处理：词向量
def word2vec(texts, embedding_dim=100, window_size=5, min_count=5):
    # 构建词汇表和统计词频
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # 过滤低频词
    vocab = [word for word, count in word_counts.items() if count >= min_count]

    # 创建词向量矩阵
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    embeddings = np.random.rand(vocab_size, embedding_dim)  # 随机初始化词向量

    # 训练Word2Vec模型
    for text in texts:
        words = text.split()
        for i, target_word in enumerate(words):
            if target_word not in word_to_idx:
                continue
            target_idx = word_to_idx[target_word]
            context_words = words[max(0, i - window_size):i] + words[i + 1:min(len(words), i + window_size + 1)]
            for context_word in context_words:
                if context_word not in word_to_idx:
                    continue
                context_idx = word_to_idx[context_word]
                embeddings[target_idx] += embeddings[context_idx]
                embeddings[context_idx] += embeddings[target_idx]

    # 归一化词向量
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    return vocab, embeddings, word_to_idx, idx_to_word

#数据处理：分词
def process_text(features, labels, num_samples):
    features = []
    labels = []
    for _ in range(num_samples):
        # 生成随机的特征字符串，假设长度为10
        feature = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=10))
        label = np.random.randint(2)  # 生成随机的0或1作为标签
        features.append(feature)
        labels.append(label)
    return features, labels

# 转换数据为模型能够接受的形式
def process_data(features, labels, word_to_idx):
    # 将特征字符串转换为索引
    features_indices = [[word_to_idx[word] for word in feature] for feature in features]
    features_tensor = torch.tensor(features_indices, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_tensor, labels_tensor
