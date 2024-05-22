import json
import pandas as pd
import os


def calc_recall(reals, predicts):
    flag = False
    for real in reals:
        for predict in predicts:
            if real in predict:
                flag = True
                break
        if flag:
            break
    return flag


if __name__ == '__main__':
    root_in = 'output_files/top1'
    recall_files = os.listdir(root_in)
    for recall_file in recall_files:
        df = pd.read_json(os.path.join(root_in, recall_file), lines=True)
        # docs = df['titles'].tolist()
        # ctxs = df['ctxs'].tolist()
        # pred_docs = df['ret_titles'].tolist()
        # pred_ctxs = df['ret_docs'].tolist()
        df['doc_in_preddoc'] = df.apply(lambda x: calc_recall(x['titles'], x['ret_titles']), axis=1)
        df['doc_in_predctx'] = df.apply(lambda x: calc_recall(x['titles'], x['ret_docs']), axis=1)
        df['ctx_in_predctx'] = df.apply(lambda x: calc_recall(x['ctxs'], x['ret_docs']), axis=1)
        print(f'current file: {recall_file}')
        print(f': 文档的召回率为：{df["doc_in_preddoc"].sum()/df.shape[0]}')
        print(f': 文档在上下文召回率为：{df["doc_in_predctx"].sum()/df.shape[0]}')
        print(f': 上下文召回率为：{df["ctx_in_predctx"].sum()/df.shape[0]}')