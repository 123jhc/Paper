import jieba
import string

def process_text(text, max_length=500):
    # 分词处理并去除标点符号
    seg_list = jieba.lcut(text)
    seg_list_no_punct = [word for word in seg_list if word not in string.punctuation]
    
    # 进一步过滤逗号、句号等标点符号
    seg_list_no_punct = [word for word in seg_list_no_punct if word not in ['，', '。', '、', '；', '：']]
    
    # 从文本末尾开始截取指定长度的词汇
    tokenized_text = seg_list_no_punct[-max_length:]
    
    return tokenized_text

# 文件路径
file_path = '财务文本\data.txt'

# 读取txt文件并处理文本
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 调用函数处理文本
tokenized_text = process_text(text)

# 打印处理后的分词结果
print("分词效果如下：")
print(' '.join(tokenized_text))
