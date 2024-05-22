from tqdm import tqdm

data_path = 'datasets/corpus/psgs_w100.tsv'
titles = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        title = line.strip().split('\t')[2]
        titles.append(title)

import IPython
IPython.embed()
