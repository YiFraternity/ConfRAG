from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
import spacy
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BaseDataset:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def format(self, fewshot: int = 0):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)

    def get_real_prediction(self, pred):
        return pred


class StrategyQA(BaseDataset):
    def __init__(self, data_path: str):
        logger.info(f"Loading StrategyQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, "strategyqa_train.json"), "r") as fin:
            dataset_1 = json.load(fin)
        with open(os.path.join(data_path, "strategyqa_train_paragraphs.json"), "r") as fin:
            dataset_2 = json.load(fin)
        for data in tqdm(dataset_1):
            example = {
                "qid": data["qid"],
                "question": data["question"],
                "answer": "yes" if data["answer"] == True else "no",
            }
            title = []
            ctxs = []
            for evi in data["evidence"][0]:
                if type(evi) == list:
                    for t in evi:
                        if type(t) == list:
                            title.extend(t)
                        else:
                            title.append(t)
                else:
                    title.append(evi)
            real_titles = []
            for tl in title:
                if tl == "operation" or tl == "no_evidence":
                    continue
                if tl in dataset_2:
                    real_titles.append(dataset_2[tl]["title"])
                    ctxs.append(dataset_2[tl]["content"])
            example["titles"] = real_titles
            example["ctxs"] = ctxs
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""


class WikiMultiHopQA(BaseDataset):
    def __init__(self, data_path: str):
        logger.info(f"Loading WikiMultiHopQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), 'r') as fin:
            js = json.load(fin)
            for example in tqdm(js):
                qid = example['_id']
                question = example['question']
                ans = example['answer']
                ans_id = example['answer_id']
                ctxs = example['context']
                titles = []
                cots = []
                for ctx in ctxs:
                    titles.append(ctx[0])
                    cots.append(ctx[1])
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                    'titles': titles,
                    'ctxs': cots,
                })
        self.dataset = Dataset.from_list(dataset)
        self.init_id_aliases(data_path)

    @classmethod
    def init_id_aliases(cls, data_path):
        cls.id_alias: Dict[str, List[str]] = {}
        with open(os.path.join(data_path, 'id_aliases.json'), 'r') as fin:
            for l in fin:
                l = json.loads(l)
                cls.id_alias[l['Q_id']] = l['aliases']

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        if ground_truth_id and ground_truth_id in cls.id_alias:
            return cls.id_alias[ground_truth_id]
        else:
            return []

    def get_real_prediction(self, pred):
        if "the answer is" in pred:
            beg = pred.find("the answer is") + len("the answer is") + 1
            pred = pred[beg:] # delete final "."
            if pred.endswith("</s>"):
                pred = pred[:len(pred) - len("</s>")]
            if pred.endswith("<|endoftext|>"):
                pred = pred[:len(pred) - len("<|endoftext|>")]
            if pred.endswith("."):
                pred = pred[:-1]
            return pred
        else:
            return pred


class HotpotQA(BaseDataset):
    def __init__(self, data_path: str):
        logger.info(f"Loading HotpotQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'hotpotqa-dev.json'), "r") as fin:
            js = json.load(fin)
            for example in tqdm(js):
                qid = example["_id"]
                question = example["question"]
                answer = example['answer']
                context = example['context']
                titles = []
                cots = []
                for ctx in context:
                    titles.append(ctx[0])
                    cots.append(ctx[1])
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                    'titles': titles,
                    'ctxs': cots,
                })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                return pred
        else:
            return ""


class IIRC(BaseDataset):
    def __init__(self, data_path: str):
        logger.info(f"Loading IIRC dev from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), "r") as fin:
            js = json.load(fin)
            for tmp in tqdm(js):
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']

                    ans = example['answer']
                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]

                    # context = example['context']
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                        # 'ctxs': context,
                    })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                for stop_word in ["</s>", "<|endoftext|>", "\n", "."]:
                    if pred.endswith(stop_word):
                        pred = pred[:len(pred) - len(stop_word)]
                return pred
        else:
            return ""