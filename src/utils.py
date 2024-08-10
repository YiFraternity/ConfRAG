import re
from typing import List
import spacy
nlp = spacy.load("en_core_web_sm")

ANSWER_NEW_TOKEN_NUM = 2048


def split_sentences(text: str) -> List[str]:
    sentences = [sent.text.strip() for sent in nlp(text).sents]
    sentences = [sent for sent in sentences if len(sent) > 0]
    results = []
    i = 0
    while i < len(sentences):
        if re.search(r'\d+\.', sentences[i].strip()) and i < len(sentences) - 1:
            results.append(sentences[i] + " " + sentences[i + 1])
            i += 2
        else:
            results.append(sentences[i])
            i += 1
    return results

def is_complete_sentence(sentence):
    # 检查最后一个字符是否是中英文的句号、问号或感叹号
    return sentence.endswith(('。', '？', '！', '.', '?', '!'))

def process_answer_text(raw_text, pre_answer):
    text = raw_text
    ptns = r'(?i).*?\banswer\s*[:：]\s*'
    pattern = re.compile(ptns, re.DOTALL)
    result = re.sub(pattern, '', text)
    all_texts = split_sentences(result)
    if len(all_texts) > 1:
        last_txt = all_texts[-1]
        all_texts = all_texts if is_complete_sentence(last_txt) else all_texts[:-1]
    not_in_prompt_texts = [text for text in all_texts if text not in pre_answer]
    return ' '.join(not_in_prompt_texts).strip()


def process_confidence_text(raw_text, prompt):
    text = raw_text
    ptns_choice = [
        r'(?i).*?\bconfidence\s*[:：]\s*',
        r'(?i).*?\bmy confidence is',
        r'(?i).*?\ba confidence level',
    ]
    for ptns in ptns_choice:
        pattern = re.compile(ptns, re.DOTALL)
        text = re.sub(pattern, '', text)
    tmp = re.findall(r"\d+\.?\d*", text)
    if len(tmp) > 0:
        confs = float(tmp[0])
        if confs > 1:
            num_digits = len(str(int(confs)))
            scale_factor = 10 ** num_digits
            confs = min(1, confs / scale_factor)
    else:
        confs = 0.0
    return confs

def process_advice_text(raw_text, prompt):
    text = raw_text
    ptns_choice = [
        r'(?i).*?\badvice\s*[:：]\s*',
        r'(?i).*?\bmy advice is',
        r'(?i).*?\ba advice is',
    ]
    for ptns in ptns_choice:
        pattern = re.compile(ptns, re.DOTALL)
        text = re.sub(pattern, '', text)
    text = text.replace('\n', ' ')
    return text

def process_reflect_text(raw_text, prompt):
    text = raw_text
    ptns_choice = [
        r'(?i).*?\bmodified response\s*[:：]\s*',
        r'(?i).*?\bmy modified response is',
        r'(?i).*?\ba modified response is',
    ]
    for ptns in ptns_choice:
        pattern = re.compile(ptns, re.DOTALL)
        text = re.sub(pattern, '', text)
    text = text.replace('\n', ' ')
    return text


def is_ans_unknown(answers: List[str]) -> bool:
    unknown_values = [
        "not provided",
        "cannot definitively",
        "not explicitly",
        "not applicable",
        "not available",
        "cannot be",
        "unknown",
        "unsure",
        "not sure",
        "There's no",
        "don't know",
        "inconclusive",
        "not found",
        "uncertainty",
        "none",
        "unable",
        "not enough",
        "not specified",
        "not determined",
        "not disclosed",
        "not revealed",
        "not mentioned",
        "not known",
        "not stated",
        "not directly",
        "not sufficient data",
        "not sufficient information",
        "isn't sufficient",
        "isn't enough",
        "does not",
        "not specify",
    ]
    for idx, answer in enumerate(answers):
        if any(re.search(r'(?i).*?\b{}\b.*'.format(value), answer) for value in unknown_values):
            return idx, True
    pattern = r'(?i)the answer is[：:]?$'
    if re.search(pattern, answer):
        return -1, True
    return None, False


if __name__ == '__main__':
    text = "1 This is a test. 2. This is another test. 3.This is a third test.\n 4. This is a fourth test."
    # sents = split_sentences(text)
    # print(len(sents), sents)
    text = """Sure, I\'d be happy to help! Based on the context you provided, my response would be:\n\n"1. Seraphim is a concept in Christian theology, referring to a high rank of angels."\n\nMy confidence in this response is 1, as I am familiar with the concept of Seraphim in Christian theology and can provide a correct definition.</s>"""
    test_txts = ["I don't know", "I'm not sure", "The answer is unknown", "The answer is A"]
    unknow_answer = []
    with open('results/SeqRAG/Qwen1.5-7B-Chat/hotpotqa/BGEReranker/output.txt', 'r') as f:
        for line in f:
            text = line.strip()
            import json
            data = json.loads(text)
            pred = data['prediction']
            answer = split_sentences(pred)[-1].strip()
            if is_ans_unknown(answer):
                unknow_answer.append(data['qid'])
    # print(process_confidence_text(text))

