import os
import pandas as pd

def read_file(file_path, num=100):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取前100行
        for i in range(num):
            line = file.readline()
            if not line:  # 如果文件行数少于100行，会在这里跳出循环
                break
            # 输出每行的长度
            samples.append(line.strip().split('\t'))
    return samples


if __name__ == '__main__':
    file_path = 'datasets/corpus/psgs_w100.tsv'
    samples = read_file(file_path)
    column_names = samples[0]
    data = samples[1:]  # 获取除了列名之外的所有数据
    df = pd.DataFrame(data, columns=column_names)
    df['text_len']=df['text'].apply(lambda x: len(x.split()))
    print(df.describe([i*0.1 for i in range(10)]))   # 把文件分段，每行100个单词
    import IPython
    IPython.embed()