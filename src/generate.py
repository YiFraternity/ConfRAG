import re
import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT, BGEReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from examples import TUTOR_ADVICE_EXAMPLES, REFLECT_EXAMPLES
from prompts import *
from utils import (
    process_answer_text,
    process_confidence_text,
    split_sentences,
    ANSWER_NEW_TOKEN_NUM,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code = "falcon" in model_name_or_path,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_chat_template_(self, prompt, add_generation_prompt=True):
        message = [
            {"role": "system", "content": "You are a helpful assistant, please don't reply to any irrelevant words."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text

    def generate(self, input_text, max_length, return_logprobs=False):
        if self.model_config.model_type == "llama":
            input_text = self._apply_chat_template_(input_text)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_length,
                return_dict_in_generate = True,
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs

        else:
            outputs = self.model.generate(
                input_ids = input_ids,
                max_new_tokens = max_length,
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None

    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_length,
            return_dict_in_generate = True,
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max":
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]

        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
        else:
            seqentropies = None

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated,
            "token_count": self.token - other_counter.token,
            "sentence_count": self.sentence - other_counter.sentence
        }


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name,
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path,
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            elif self.retriever_type == "BGEReranker":
                self.retriever = BGEReranker(
                    model_name_or_path = self.bge_model_name_or_path,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name,
                )
            else:
                raise NotImplementedError

        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, _doc_titles, docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
            )
            return docs[0]
        elif self.retriever_type == "BGEReranker":
            _docs_ids, _doc_titles, docs = self.retriever.retrieve(
                queries = [query],
                recall_num = 100,
                topk = topk,
            )
            return docs[0]
        else:
            raise NotImplementedError


    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""

    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            docs = self.retrieve(question, topk=self.retrieve_topk)
            # 对 topk 个 passage 生成 prompt
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.sentence_solver = 'max'

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        if tokens == []:
            tid = 0
        else:
            tid = 1
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):   # 到第一个回车符或者空格为止
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            if len(probs) == 0:
                p = 0.
            else:
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1

        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt,
                self.generate_max_length,
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                curr = curr.replace("[xxx]", "")
                if self.query_formulation == "direct":
                    retrieve_question = curr
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, ptext, curr]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)

        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)

        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)]
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0

                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    # def fetch_forward(self, prev_text, curr_tokens, curr_hit):
    #     curr_text = " ".join(curr_tokens)

    #     all_text = prev_text + " " + curr_text
    #     input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
    #     input_length = input_ids.shape[1]
    #     tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

    #     atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

    #     # merge tokens
    #     range_ = []
    #     for i, t in enumerate(tokens_tmp):
    #         if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
    #             range_.append([i, i])
    #         else:
    #             range_[-1][-1] += 1
    #     tokens = []
    #     for r in range_:
    #         tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
    #         tokens.append(tokenseq)

    #     curr_st = len(tokens) - len(curr_tokens)
    #     curr_ed = len(tokens)
    #     tl, tr = 0, len(tokens)
    #     if "retrieve_query_type" in self.__dict__:
    #         if self.retrieve_query_type == "only_forward":
    #             tr = curr_st
    #         elif self.retrieve_query_type == "current":
    #             tl, tr = curr_st, curr_ed
    #         elif self.retrieve_query_type == "top_k_and_current":
    #             tr = curr_st

    #     attns = []
    #     for r in range_:
    #         att = torch.zeros(atten_tmp.shape[0], input_length)
    #         for i in range(r[0], r[1] + 1):
    #             att += atten_tmp[:, i]
    #         att /= (r[1] - r[0] + 1)
    #         att = torch.mean(att, dim=0)
    #         att = att[tl:tr]
    #         if att.shape[0] > 1:
    #             att = att / sum(att[1:]).item()
    #         attns.append(att)

    #     # 计算每个超过阈值的 token 在前文的 attentions
    #     forward_attns = torch.zeros(tr - tl)
    #     hit_cnt = 0
    #     for i in range(len(curr_hit)):
    #         if curr_hit[i] == 1:
    #             forward_attns += attns[curr_st + i]
    #             hit_cnt += 1
    #     forward_attns /= hit_cnt
    #     forward_attns = forward_attns.tolist()

    #     if "retrieve_keep_weight" in self.__dict__:
    #         topk_token = []
    #         for tok, att in zip(tokens[tl:tr], forward_attns):
    #             if att * (tr - tl + 1) >= self.retrieve_keep_weight:
    #                 topk_token.append(tok)

    #     else:
    #         topk_attn = sorted(forward_attns, reverse=True)
    #         if "retrieve_keep_top_k" in self.__dict__:
    #             top_k = min(self.retrieve_keep_top_k, tr - tl)
    #         elif "retrieve_keep_ratio" in self.__dict__:
    #             top_k = int((tr - tl) * self.retrieve_keep_ratio)
    #         else:
    #             raise NotImplementedError
    #         topk_attn = topk_attn[:top_k]
    #         topk_token = []
    #         for tok, att in zip(tokens[tl:tr], forward_attns):
    #             if att in topk_attn:
    #                 topk_token.append(tok)

    #     final_text = " ".join(topk_token)
    #     if "retrieve_query_type" in self.__dict__ and self.retrieve_query_type == "top_k_and_current":
    #         mask_curr = " ".join(
    #             list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
    #         )
    #         return final_text + " " + mask_curr
    #     else:
    #         return final_text

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        tl, tr = 0, len(tokens)
        curr_st = len(tokens) - len(curr_tokens)
        attns = []
        for r in range_:
            att = torch.zeros(atten_tmp.shape[0], input_length)
            for i in range(r[0], r[1] + 1):
                att += atten_tmp[:, i]
            att /= (r[1] - r[0] + 1)
            att = torch.mean(att, dim=0)
            att = att[tl:tr]
            if att.shape[0] > 1:
                att = att / sum(att[1:]).item()
            attns.append(att)

        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(tr - tl)
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if match(tok):
                real_pairs.append((att, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # print('####', prompt)
            # prompt += case + " " + text
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_max_length,
                # self.attention_solver,
                use_entropy = self.method == "dragin",
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)

            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all

                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)

                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)

                elif self.query_formulation == "real_words":
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext,
                        curr_tokens = curr_tokens,
                        curr_hit = curr_hit,
                    )
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

                # print("### retrieve_question ###")
                # print(retrieve_question)
                # context = "### Context: ###\n"
                # for i, doc in enumerate(docs):
                #     context += f"[{i+1}] {doc}\n"
                # print(context)
                # print(text)

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        # print("#" * 20)
        return text


class SeqConfidenceRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def _generate_text_(self, prompt, generate_max_length, return_logprobs=False, gen_type='answer'):
        gen_text, _, _ = self.generator.generate(
            prompt,
            generate_max_length,
            return_logprobs=return_logprobs,
        )
        if gen_type == 'answer':
            process_text = process_answer_text
        elif gen_type == 'confidence':
            process_text = process_confidence_text
        text = process_text(gen_text)
        return text

    def _get_seq_confs_(self, question, context, response):
        contxt = ' '.join([question.strip(), context.strip()])
        conf_prompt = CONFIDENCE_TEMPLATE.format(
            context=contxt,
            response=response,
        )
        confs_text = self._generate_text_(
            conf_prompt,
            500,
            return_logprobs=False,
            gen_type='confidence',
        )
        tmp = re.findall(r"\d+\.?\d*", confs_text)
        if len(tmp) > 0:
            confs = float(tmp[0])
            if confs > 1:
                num_digits = len(str(int(confs)))
                scale_factor = 10 ** num_digits
                confs = min(1, confs / scale_factor)
        else:
            confs = 0.0
        return confs

    def _reflection_(self, question, context, response, advice_examples='', reflect_examples=''):
        """
        # 反思需要两次生成
        # 1. tutor-advice: 用于指导从哪个层面思考
        # 2. Refine: 用于提升回复的质量
        """

        tutor_data = {
            "header": TUTOR_ADVICE_HEADER,
            "examples": advice_examples,
            "middle": TUTOR_ADVICE_MIDDLE,
            "question": question,
            "context": context,
            "response": response,
        }

        advice_prompt = ADVICE_TEMPLATE.format(**tutor_data)
        advice, _, _ = self.generator.generate(
            advice_prompt,
            500,
            return_logprobs=False
        )
        advice = advice.strip()

        reft_prompt = {
            "header": REFLECTION_HEADER,
            "examples": reflect_examples,
            "middle": REFLECTION_MIDDLE,
            "question": question,
            "context": context,
            "response": response,
            "tutor_ins": advice,
        }
        reflect_prompt = REFLECTION_TEMPLATE.format(**reft_prompt)
        reflect, _, _ = self.generator.generate(
            reflect_prompt,
            500,
            return_logprobs=False,
        )
        return reflect.strip()

    def modifier(self, question, ptext, text, adv_examples='', ref_examples=''):
        """
        按模型对新生成的内容判断自信度进行修改。删除置信度不高的文本

        return:
            confs_socres: list of float, the confidence score for each sentence
            new_text: str, the new text with the highest confidence score
            hallucination: bool, whether the new text is hallucinated
        """
        hallucination = False

        sentences = split_sentences(text)
        confs_socres = []
        context = ptext + ' '
        modified_texts = []
        for i, sent in enumerate(sentences):
            modify_text = sent
            if i > 0:
                context += sentences[i-1]
            confs = self._get_seq_confs_(question, context, sent)
            if confs < self.reflection_threshold and confs >= self.hallucination_threshold:
            # 根据置信度进行幻觉判断，如果需要反思，则调用self._reflection生成反思后的文本，之后再对生成的新文本进行置信度的判断，如果置信度比之前的文本高则进行置信度的替换（为什么不更改文本？reft_text）
            # 若置信度过低，则将该句子mask掉
                print(f'cur confs:{confs}, performed reflect')
                reft_text = self._reflection_(question, context, sent, advice_examples=adv_examples, reflect_examples=ref_examples)
                reft_cons = self._get_seq_confs_(question, context, reft_text)
                if reft_cons >= confs:
                    modify_text = reft_text
                    confs = reft_cons
            elif confs < self.hallucination_threshold:
                print(f'cur confs:{confs}, performed hullucination')
                hallucination = True
                modify_text = "[xxx]."

            modified_texts.append(modify_text)
            confs_socres.append(confs)
        modified_text = ' '.join(modified_texts)

        return sentences, confs_socres, modified_text, hallucination

    def inference(self, question, demo, case):
        text = ""    # 用于存储置信度高的序列，以及后续不可提升序列置信度的句子
        ptext = ""   # 用于存储之前生成所有的可用句子
        docs = []
        pre_seq_confidence = 0.
        while True:
            examples = "".join([d["case"]+"\n" for d in demo])
            doc_str = ''
            if len(docs) > 0:
                doc_str += "Douments:\n"
                for i, doc in enumerate(docs):
                    doc_str += f"[{i+1}] {doc}\n"
                doc_str += "Please answer the question based on the documents and the previous response.\n"

            prompt = ANSWER_QUESTION_TEMPLETE.format(
                demo=examples,
                docs=doc_str,
                question=question,
            )

            new_text = self._generate_text_(prompt, self.generate_max_length)
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)

            all_texts, seq_confidences, modified_texts, hallucination = self.modifier(question, ptext, new_text, adv_examples=TUTOR_ADVICE_EXAMPLES, ref_examples=REFLECT_EXAMPLES)
            if hallucination:
                forward_all = [question, ptext.strip(), modified_texts]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)
                forward_all = forward_all.replace("[xxx].", " ")
                if self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                docs = docs.tolist()
                if self.use_counter == True:
                    self.counter.hallucinated += 1

            for i, sent in enumerate(all_texts):
                if seq_confidences[i] >= self.reflection_threshold:
                    ptext += sent + " "
                    text += sent + " "
                else:
                    pre_seq_confidence = seq_confidences[i]
                    break

            try:
                if "the answer is" in text or seq_confidences[0] <= pre_seq_confidence:
                    for i, sent in enumerate(all_texts):
                        if sent not in text:
                            text += sent + " "
                    # IPython.embed()
                    break
            except:
                break
        return text