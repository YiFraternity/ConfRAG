import re

ANSWER_NEW_TOKEN_NUM = 2048


def split_sentences(text):
    # 定义正则表达式模式，匹配句子结束的标点符号（包括全角和半角的句号和换行符），但不包括数字后的点和冒号
    pattern = re.compile(r'(?<!\d)([。.\n])|(?<!\d)([:：])(?!\d)')
    # 使用正则表达式替换匹配的部分，同时保留匹配的标点符号，并在每个匹配的标点符号后添加一个特殊标记
    text_with_markers = re.sub(pattern, r'\1\2<split>', text)
    # 根据特殊标记拆分句子
    sentences = text_with_markers.split('<split>')
    # 去除空白句子
    sentences = [sentence.strip() + ('\n' if sentence.endswith('\n') else '') for sentence in sentences if sentence.strip()]
    return sentences



def process_answer_text(text, pre_answer):
    ptns = r'(?i).*?\banswer\s*[:：]\s*'
    pattern = re.compile(ptns, re.DOTALL)
    result = re.sub(pattern, '', text)
    all_texts = split_sentences(result)
    not_in_prompt_texts = [text for text in all_texts if text not in pre_answer]
    return ' '.join(not_in_prompt_texts).strip()


def process_confidence_text(text, prompt):
    ptns_choice = [
        r'(?i).*?\bconfidence\s*[:：]\s*',
        r'(?i).*?\bmy confidence',
        r'(?i).*?\ba confidence level',
    ]
    for ptns in ptns_choice:
        pattern = re.compile(ptns, re.DOTALL)
        text = re.sub(pattern, '', text)
    return text.strip()

if __name__ == '__main__':
    text = "1 This is a test. 2. This is another test. 3.This is a third test.\n 4. This is a fourth test."
    # sents = split_sentences(text)
    # print(len(sents), sents)
    text = """Sure, I\'d be happy to help! Based on the context you provided, my response would be:\n\n"1. Seraphim is a concept in Christian theology, referring to a high rank of angels."\n\nMy confidence in this response is 1, as I am familiar with the concept of Seraphim in Christian theology and can provide a correct definition.</s>"""

    print(process_confidence_text(text))

