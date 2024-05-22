import argparse
import json
import os
from retriever import BM25, SGPT, BGEReranker
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from data_add_title import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC


def unfold_2nd_list(x):
    if len(x) == 0:
        return ['']
    if isinstance(x[0], list):
        return sum(x, [])
    else:
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='strategyqa', help='config path')
    parser.add_argument("--retriever_type", type=str, default='BGEReranker')
    parser.add_argument("--model_name_or_path", type=str, default='/home/liuyh0916/models/open_models/bge-reranker-base')
    parser.add_argument("--output_dir", type=str, default='/home/liuyh0916/calibration/output_files')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    data_path = f'datasets/{args.dataset}/'
    if args.dataset == "strategyqa":
        data = StrategyQA(data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(data_path)
    elif args.dataset == "iirc":
        data = IIRC(data_path)
    else:
        raise NotImplementedError
    data = data.dataset

    if args.retriever_type == "BM25":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        retriever = BM25(
            tokenizer = tokenizer,
            index_name = "wiki" if "es_index_name" not in args else args.es_index_name,
            engine = "elasticsearch",
        )
    elif args.retriever_type == "BGEReranker":
        retriever = BGEReranker(
            model_name_or_path = args.model_name_or_path,
            index_name = "wiki" if "es_index_name" not in args else args.es_index_name,
        )
    output_path = f'{args.output_dir}/top{args.topk}'
    os.makedirs(output_path, exist_ok=True)

    results = []
    with open(f'{output_path}/{args.retriever_type}_{args.dataset}.jsonl', 'w') as f:
        # 设置batch size
         for i in tqdm(range(0, len(data), args.batch_size)):
            batch_d = data[i:i+args.batch_size]
            queries = batch_d['question']
            _, ret_titles, ret_docs = retriever.retrieve(queries, topk=args.topk)
            for id in range(len(queries)):
                result_q = {}
                result_q['titles'] = unfold_2nd_list(batch_d['titles'][id])
                result_q['ctxs'] = unfold_2nd_list(batch_d['ctxs'][id])
                result_q['question'] = batch_d['question'][id]

                result_q['ret_titles'] = ret_titles[id].tolist()
                result_q['ret_docs'] = ret_docs[id].tolist()
                json.dump(result_q, f)
                f.write('\n')
