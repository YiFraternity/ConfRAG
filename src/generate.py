import random
import re
import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT, BGEReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from examples import TUTOR_ADVICE_EXAMPLES, REFLECT_EXAMPLES
from prompts import *
from utils import *
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BasicGenerator:

    def __update_generate_config__(self, params):
        for k in self.generate_config.keys():
            if k in params:
                self.generate_config[k] = params[k]
        if self.generate_config["do_sample"] == False:
            self.generate_config["top_k"] = -1
            self.generate_config['temperature'] = 0
            self.generate_config['top_p'] = 1.0

    def __init__(self, model_name_or_path, params={}):
        logger.info(f"Loading model from {model_name_or_path}")

        self.generate_config = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "num_beams": 1,
            "do_sample": True,
        }
        self.__update_generate_config__(params)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if 'use_vllm' in params and not params['use_vllm']:
            self.model_config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            if self.model_config.model_type in ["llama", "qwen2"]:
                self.space_token = "Ġ"  # Llama3为`Ġ`，Llama2为`▁`
            else:
                self.space_token = self.tokenizer.tokenize(' ')[0]

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.use_vllm = False
        else:
            self.model = LLM(model=model_name_or_path)
            self.use_vllm = True

    def _get_chat_message_(self, prompt):
        message = [
            {"role": "system", "content": "You are a concise and efficient assistant. Please proceed directly with reasoning and answering, avoiding any unrelated phrases such as 'Let me help,', 'Let’s analyze the information,', or similar expressions. You must not repeat the reasoning and response in Enlish."},
            {"role": "user", "content": prompt}
        ]
        return message

    def generate(
            self,
            input_text,
            max_new_tokens,
            return_logprobs=False,
            gen_type="answer",
            process_gen_text=False,
        ):
        # 对生成的内容进行一步处理
        if gen_type == 'answer':
            process_text = process_answer_text
        elif gen_type == 'confidence':
            process_text = process_confidence_text
        elif gen_type == 'advice':
            process_text = process_advice_text
        elif gen_type == 'reflection':
            process_text = process_reflect_text
        elif gen_type == 'keywords':
            process_text = process_keywords_text
        elif gen_type == 'retr_info':
            process_text = process_retr_info_text
        else:
            raise ValueError(f"gen_type {gen_type} is not supported")
        """
        Args:
            gen_type (str): [`answer`, `confidence`, `advice`, `reflection`, `keywords`]
        """
        message = self._get_chat_message_(input_text)
        if not self.use_vllm:
            input_text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)
            input_length = input_ids.shape[1]
            attention_mask = torch.ones_like(input_ids)

            if return_logprobs:
                outputs = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens,
                    return_dict_in_generate = True,
                    output_scores = True,
                    **self.generate_config,
                )
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )

                generated_tokens = outputs.sequences[:, input_length:]
                text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True) # text = "".join(tokens)
                tokens = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens[0]]
                special_tokens_index = [idx for idx, t in enumerate(tokens) if t == '']
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
                tokens = [ t for idx, t in enumerate(tokens) if idx not in special_tokens_index]     # remove sepical token
                logprobs = [p for idx, p in enumerate(logprobs) if idx not in special_tokens_index]     # remove sepical token
                return text, tokens, logprobs

            else:
                outputs = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens,
                    **self.generate_config,
                )
                generated_tokens = outputs[:, input_length:]
                text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        else:
            sampling_params = SamplingParams(
                temperature=self.generate_config['temperature'],
                top_p=self.generate_config['top_p'],
                top_k=self.generate_config['top_k'],
                min_p=0.0 if self.generate_config['temperature']==0 else 1,
                best_of=1,
                max_tokens=max_new_tokens,
            )
            outputs = self.model.chat(
                message,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            text = outputs[0].outputs[0].text
        if process_gen_text:
            processed_text = process_text(text, input_text)
        else:
            processed_text = text
        return text, processed_text, None, None

    def generate_attn(
            self,
            input_text,
            max_new_tokens,
            solver="max",
            use_entropy = False,
            use_logprob = False,
        ):
        message = self._get_chat_message_(input_text)
        input_text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            return_dict_in_generate = True,
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens[0]]
        text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        special_tokens_idx = [idx for idx, t in enumerate(tokens) if t == '']
        special_tokens_idx_t = torch.tensor(special_tokens_idx, dtype=torch.int)
        tokens = [t for idx, t in enumerate(tokens) if idx not in special_tokens_idx]     # remove sepical token
        mask = torch.ones(generated_tokens.shape[1], dtype=torch.bool)
        mask[special_tokens_idx_t] = False
        generated_tokens = generated_tokens[:, mask]
        assert len(tokens) == generated_tokens.shape[1]
        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            # if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
            if i == 0 or t.startswith(' '):
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
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(' ', "")
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
        self.reflect = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "reflect_count": self.reflect - other_counter.reflect,
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
        self.generator = BasicGenerator(self.model_name_or_path, self.__dict__)
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
            separator = np.array([[' | '] * len(docs[0])])
            result = np.char.add(np.char.add(_doc_titles, separator), docs)
            return result[0]
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
            separator = np.array([[' | '] * len(docs[0])])
            result = np.char.add(np.char.add(_doc_titles, separator), docs)
            return result[0]
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

    def inference(self, question, demo):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = get_answer_prompt([], demo=demo, question=question, text="")
        text, _, _, _ = self.generator.generate(
            prompt,
            max_new_tokens=self.max_length,
        )
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = get_answer_prompt(docs=docs, demo=demo, question=question, text="")
        text, _, _, _ = self.generator.generate(
            prompt,
            self.max_length,
        )
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def _get_retr_docs_(self, question, ptext):
        if self.query_formulation == "forward_all":
            tmp_all = [question, ptext]
            retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
        elif self.query_formulation == "last_sentence":
            tmp_all = [question, ptext]
            retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
            retrieve_question = retrieve_question.strip()
            retrieve_question = self.get_last_sentence(retrieve_question)
            if len(retrieve_question) == 0:
                retrieve_question = question
        else:
            retrieve_question = question
        docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
        docs = docs.tolist()
        return docs

    def inference(self, question, demo):
        ptext = ""
        ptexts = []
        docs = []
        old_len = -1
        while True:
            # 对 topk 个 passage 生成 prompt
            prompt = get_answer_prompt(docs=docs, demo=demo, question=question, text=ptext)
            text, answer, _, _ = self.generator.generate(
                prompt,
                self.generate_length,
                process_gen_text=True,
            )
            if self.use_counter == True:
                self.counter.add_generate(text, self.generator.tokenizer)
            if self.method == "fix-sentence-retrieval":
                # fix sentence
                sentences = list(nlp(answer).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                answer = sentences[0]
                ptexts.append(answer)
            if self.method == "random-sentence-retrieval":
                sentences = list(nlp(answer).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                first_n = random.randint(1, len(sentences))
                first_n_sents = sentences[:first_n]
                answer = ' '.join(first_n_sents)
                ptexts.extend(first_n_sents)
            ptext += (" " + answer.strip())
            ptext = ptext.strip()
            # 判断 token 的个数要少于 max_length
            tokens_count = len(self.generator.tokenizer.encode(ptext))
            if tokens_count >= self.max_length or tokens_count <= old_len or "the answer is" in ptext:
                if len(ptexts)==0 or is_ans_unknown(ptexts[-1]):
                    ptext = ' '.join(ptexts[:-1])
                    ptext = ptext.strip()
                    docs = self._get_retr_docs_(question, ptext)
                    prompt = get_answer_prompt(
                        docs,
                        demo,
                        question,
                        text=ptext,
                    )
                    text, new_text, _, _ = self.generator.generate(
                        prompt,
                        max_new_tokens=self.max_length,
                        return_logprobs=False,
                        gen_type='answer',
                        process_gen_text=True,
                    )
                    ptext += (' ' + new_text)
                    ptext = ptext.strip()
                break
            old_len = tokens_count

            docs = self._get_retr_docs_(question, ptext)
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
            probs = [1 - exp(v) for v in logprobs[tid:tr]]
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

    def inference(self, question, demo):
        # assert self.query_formulation == "direct"
        ptext = ""
        old_len = -1
        while True:
            docs = []
            prompt = get_answer_prompt(
                docs=docs,
                demo=demo,
                question=question,
                text=ptext
            )
            new_text, tokens, logprobs = self.generator.generate(
                prompt,
                self.generate_length,
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext_, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            ptext += " " + ptext_.strip()
            if hallucination:
                curr = curr.replace("[xxx]", "")
                if self.query_formulation == "direct":
                    retrieve_question = curr
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, ptext_, curr]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                elif self.query_formulation == "last_sentence":
                    ptext = ptext.strip()
                    txt = ptext if len(ptext) > 0 else question
                    retrieve_question = self.get_last_sentence(txt)
                else:
                    raise NotImplemented
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = get_answer_prompt(
                    docs = docs,
                    demo = demo,
                    question = question,
                    text = ptext
                )
                text, new_text, _, _ = self.generator.generate(
                    prompt,
                    self.generate_length,
                )
                if self.use_counter == True:
                    self.counter.add_generate(text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                ptext += (" " + new_text.strip())
            ptext = ptext.strip()
            # 判断 token 的个数要少于 max_length
            tokens_count = len(self.generator.tokenizer.encode(ptext))
            if tokens_count >= self.max_length or tokens_count <= old_len or "the answer is" in ptext:
                break
            old_len = tokens_count
        return ptext


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

    def inference(self, question, demo):
        return super().inference(question, demo)


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

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt").to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]
        atten_tmp = atten_tmp.to('cpu')
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

    def inference(self, question, demo):
        ptext = ""
        docs = []
        old_len = -1
        while True:
            prompt = get_answer_prompt(
                docs=docs,
                demo=demo,
                question=question,
                text=ptext,
            )
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_length,
                # self.attention_solver,
                use_entropy = self.method == "dragin",
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext_, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)

            if not hallucination:
                ptext += (" " + new_text.strip())
                ptext = ptext.strip()
            else:
                ptext += (" " + ptext_.strip())
                ptext = ptext.strip()
                forward_all = [question, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer=self.generator.tokenizer):
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
                    if len(retrieve_question) == 0:
                        retrieve_question = question

                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)

                elif self.query_formulation == "real_words":
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + ptext,
                        curr_tokens = curr_tokens,
                        curr_hit = curr_hit,
                    )
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = get_answer_prompt(
                    docs=docs,
                    demo=demo,
                    question=question,
                    text=ptext,
                )
                new_text, _, _, _ = self.generator.generate(
                    prompt,
                    max_new_tokens = self.generate_length,
                    process_gen_text = False,
                )
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                ptext += (" " + new_text.strip())

            ptext = ptext.strip()
            tokens_count = len(self.generator.tokenizer.encode(ptext))
            if tokens_count >= self.max_length or tokens_count <= old_len or "the answer is" in ptext:
                break
            old_len = tokens_count
        return ptext


class SeqConfidenceRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def _get_seq_confs_value_(self, question:str, history_resp:str, response:str, docs:list):
        conf_prompt = get_conf_value_prompt(
            question=question,
            history_resp=history_resp,
            response=response,
            docs=docs
        )
        text, confs, _, _ = self.generator.generate(
            conf_prompt,
            max_new_tokens=self.generate_confidence_length,
            return_logprobs=False,
            gen_type='confidence',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return confs

    def _get_seq_confs_level_(self, question:str, history_resp:str, response:str, docs:list):

        def __conf_level_in_confs__(confs):
            high_conf_levels = ['very certain', 'fairly certain', '1', '2']
            mid_conf_levels = ['lightly certain', '3']
            low_conf_levels = ['not certain', 'very uncertain', '4', '5']
            for conf_level in high_conf_levels:
                if conf_level in confs.lower():
                    return 'high'
            for conf_level in mid_conf_levels:
                if conf_level in confs.lower():
                    return 'mid'
            for conf_level in low_conf_levels:
                if conf_level in confs.lower():
                    return 'low'

        conf_prompt = get_conf_level_prompt(
            question=question,
            history_resp=history_resp,
            response=response,
            docs=docs
        )
        text, confs, _, _ = self.generator.generate(
            conf_prompt,
            max_new_tokens=self.generate_confidence_length,
            process_gen_text=False,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)

        confs = __conf_level_in_confs__(text)
        if "turbulence" not in self.__dict__ or not self.turbulence:
            if confs == 'low':
                return confs, 'lack knowledge'
            else:
                return confs, 'exact' if confs == 'high' else 'reflect'
        else:
            # Add perturbation to rejudge the confidence level of the model in response
            if confs == 'low':
                return confs, 'lack knowledge'
            else:
                turb_resp = ""
                turb_resp, _, _, _ = self.generator.generate(
                    ENTITY_REPLEACE_TEMPLATE.format(question=question, sentence=response),
                    max_new_tokens=32,
                    return_logprobs=False,
                    process_gen_text=False,
                )
                if turb_resp == '' or 'None' in turb_resp:   # confs is response confs
                    return confs, 'exact' if confs == 'high' else 'reflect'

                turb_conf_prompt = get_conf_level_prompt(
                    question=question,
                    history_resp=history_resp,
                    response=turb_resp,
                    docs=docs
                )
                text, mod_confs, _, _ = self.generator.generate(
                    turb_conf_prompt,
                    max_new_tokens=self.generate_confidence_length,
                    process_gen_text=False,
                )
                mod_confs = __conf_level_in_confs__(text)
                if self.use_counter:
                    self.counter.add_generate(text, self.generator.tokenizer)
                if mod_confs in ['high', 'mid']:
                    return 'low', 'hallucination'
                else:
                    return confs, 'exact' if confs == 'high' else 'reflect'

    def _generate_(self, docs=[], demo=[], question='', ptext='', qtype='answer', generate_length=-1):
        if qtype == 'reason':
            prompt = get_reason_prompt(docs, reason_pth=question)
        else:
            prompt = get_answer_prompt(docs, demo, question, ptext)
        # 当前轮次的新文本
        text, new_text, _, _ = self.generator.generate(
            prompt,
            max_new_tokens=self.generate_length if generate_length==-1 else generate_length,
            return_logprobs=False,
            gen_type='answer',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, new_text

    def _get_keywords_(self, addition_info):
        if addition_info == "":
            return ""
        keyword_prompt = KEYWORDS_TEMPLATE.format(sentence=addition_info)
        _, keywords, _, _ = self.generator.generate(
            keyword_prompt,
            max_new_tokens=10,
            gen_type='keywords',
            process_gen_text=True,
        )
        return keywords

    def _get_retr_query_(self, question, response, resp_conf_type):
        response = response.strip()
        if 'hallucination' == resp_conf_type:
            if len(response) == 0:
                retr_info_prompt = INIT_INFO_TEMPLATE.format(
                    question=question,
                )
            else:
                retr_info_prompt = MISSING_INFO_TEMPLATE.format(
                    question=question,
                    response=response.strip(),
                )
        else:
            if len(response) == 0:
                retr_info_prompt = INIT_INFO_TEMPLATE.format(
                    question=question,
                )
            else:
                retr_info_prompt = MISSING_INFO_TEMPLATE.format(
                    question=question,
                    response=response.strip(),
                )
        text, retr_info, _, _ = self.generator.generate(
            retr_info_prompt,
            max_new_tokens=64,
            gen_type='retr_info',
            process_gen_text=True,
        )
        return retr_info

    def _get_retr_docs_(self, question, hist_resps, cur_step_ptext, cur_ptext_conf_type, retr_type='retr_query'):
        """
        Args:
            question: str
            historys: list of str
            cur_step_ptext: str
            retr_type: str [retr_query, retr_query_keywords]
        """
        history = " ".join(hist_resps).strip()
        if self.query_formulation == "direct":
            retrieve_question = question
        elif self.query_formulation == "forward_all":
            forward_all = [question, cur_step_ptext]
            forward_all = " ".join(s for s in forward_all if len(s) > 0)
            forward_all = forward_all.replace("[xxx].", "")
            retrieve_question = forward_all
        elif self.query_formulation == "last_sentence":
            forward_all = [question, history]
            forward_all = " ".join(s for s in forward_all if len(s) > 0)
            retrieve_question = forward_all
            retrieve_question = self.get_last_sentence(forward_all)
        elif self.query_formulation == "query_and_last_sentence":
            forward_all = [history]
            forward_all = " ".join(s for s in forward_all if len(s) > 0)
            retrieve_question = self.get_last_sentence(forward_all)
            retrieve_question = question + " " + retrieve_question
        elif self.query_formulation == "generate_query":
            if retr_type == "retr_query":
                retrieve_question = self._get_retr_query_(question, history, cur_ptext_conf_type)
            elif retr_type == "retr_query_keywords":
                keywords = self._get_keywords_(cur_step_ptext, cur_ptext_conf_type)
                retrieve_question = question + " " + keywords
        else:
            raise NotImplemented
        retrieve_question = retrieve_question.strip()
        docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
        docs = docs.tolist()
        return docs, retrieve_question

    def _reflection_(self, question, history_resp, response, docs=[]):
        """
        # 反思需要两次生成
        # 1. tutor-advice: 用于指导从哪个层面思考
        # 2. Refine: 用于提升回复的质量
        """
        if self.use_counter:
            self.counter.reflect += 1
        doc_str = get_docstr(docs)
        tutor_data = {
            "header": TUTOR_ADVICE_HEADER,
            "examples": TUTOR_ADVICE_EXAMPLES,
            "docs": (TUTOR_USE_DOCS + '\n' + doc_str) if len(docs) > 0 else doc_str,
            "middle": (TUTOR_USE_DOCS_MIDDLE if len(docs) > 0 else TUTOR_NOT_USE_DOCS_MIDDLE) +  " " + TUTOR_ADVICE_MIDDLE,
            "question": question,
            "history_resp": history_resp.replace('\n', ' '),
            "response": response,
        }
        advice_prompt = ADVICE_TEMPLATE.format(**tutor_data)
        text, advice, _, _ = self.generator.generate(
            advice_prompt,
            max_new_tokens=self.max_length,
            return_logprobs=False,
            gen_type='advice',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)

        reft_prompt = {
            "header": REFLECTION_HEADER,
            "examples": REFLECT_EXAMPLES,
            "docs": (REFLECT_USE_DOC + '\n' + doc_str + '\n') if len(docs) > 0 else doc_str,
            "middle": (REFLECT_USE_DOC_MIDDLE if len(docs) > 0 else REFLECT_NOT_USE_DOC_MIDDLE) + REFLECTION_MIDDLE,
            "question": question,
            "history_resp": history_resp.replace('\n', ' '),
            "response": response,
            "tutor_ins": advice,
        }
        reflect_prompt = REFLECTION_TEMPLATE.format(**reft_prompt)
        text, reflect, _, _ = self.generator.generate(
            reflect_prompt,
            max_new_tokens=self.max_length,
            return_logprobs=False,
            gen_type='reflection',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return reflect

    def _get_confs_class_(self, question, history_resp, sent, docs):
        confs_class = self.confs_class if "confs_class" in self.__dict__ else 'value'
        if confs_class == 'value':
            confs, = self._get_seq_confs_value_(question, history_resp, sent, docs)
        else:
            confs, conf_type = self._get_seq_confs_level_(question, history_resp, sent, docs)

        confs_value = 0
        if self.reflection_threshold < 0:  # no reflect
            if (confs_class == 'value' and confs >= self.hallucination_threshold) or \
            (confs_class=='level' and confs=='high'):
                confs_value = 1
            else:
                confs_value = -1
            return confs_value, conf_type

        if (confs_class == 'value' and confs >= self.reflection_threshold) or \
            (confs_class == 'level' and confs=='high'):
            confs_value = 1
        elif (confs_class == 'value' and confs < self.hallucination_threshold) or \
            (confs_class == 'level' and confs=='low'):
            confs_value = -1
        else:
            confs_value = 0
        return confs_value, conf_type

    def modifier(self, question, ptext, text, docs):
        """
        按模型对新生成的内容判断自信度进行修改。删除置信度不高的文本
        Args:
            confs_class: str, the type of confidence score, 'value' or 'level'
        Returns:
            ptexts_: list of str, sentences that exceed the hallucination threshold for first. If there are none, retain the first sentence.
            pconfs_: list of float, the confidence score for sentence in the ptexts_
            pconf_types_: list of str, the confindence type for sentence in the ptexts_
            hallucination: bool, whether the text is hallucinated
        """
        hallucination = False

        reflect_tag = True   # 仅仅在前面所有句子置信度都高的情况下才可以反思，若前面出现了置信度较低的情况，则反思无效
        sentences = split_sentences(text)
        if len(sentences) == 0:
            return [], [], [], hallucination

        ptexts_, pconfs_, pconf_types_ = [], [], []
        history_resp = ptext
        for i, sent in enumerate(sentences):
            modify_text = sent
            if i > 0:
                history_resp += (' ' + sentences[i-1])
            confs, conf_type = self._get_confs_class_(question, history_resp, sent, docs)
            if 0 == confs and reflect_tag:
                # 根据置信度进行幻觉判断，如果需要反思，则调用self._reflection生成反思后的文本，之后再对生成的新文本进行置信度的判断，如果置信度比之前的文本高则进行置信度的替换
                # 若置信度过低，则将该句子mask掉
                print(f'cur confs:{confs}, performed reflect')
                reft_text = self._reflection_(question, history_resp, sent, docs)
                reft_cons, reft_conf_type = self._get_confs_class_(question, history_resp, reft_text, docs)
                if reft_cons >= 0:
                    modify_text = reft_text
                    confs = reft_cons
                    conf_type = reft_conf_type
            elif confs < 0:
                print(f'cur confs:{confs}, performed hallucination')
                hallucination = True
                reflect_tag = False

            ptexts_.append(modify_text)
            pconfs_.append(confs)
            pconf_types_.append(conf_type)

        assert len(sentences) == len(pconfs_)

        hall_index = find_element_index(pconfs_, -1) + 1
        if hall_index > 0:
            ptexts_ = ptexts_[:hall_index]
            pconfs_ = pconfs_[:hall_index]
            pconf_types_ = pconf_types_[:hall_index]
        return ptexts_, pconfs_, pconf_types_, hallucination

    def inference(self, question, demo):
        ptext = ""     # 用于存储置信度高的序列，以及后续不可提升序列置信度的句子
        ptexts = []
        docs = []
        old_len = -1
        retr_num = 0
        while True:
            _, new_text = self._generate_([], demo, question, ptext)
            ptexts_, pconfs_, pconf_types_, hallucination = self.modifier(
                question,
                ptext,
                new_text,
                docs=docs,
            )
            if not hallucination:
                ptext += (' ' + (' '.join(ptexts_)))
                ptexts.extend(ptexts_)
            else:
                if len(ptexts_) > 0:
                    ptexts.extend(ptexts_[:-1])
                    ptext += (' ' + (' '.join(ptexts_[:-1])))
                    pre_seq = ptexts_[-1]
                    pre_seq_conf_type = pconf_types_[-1]
                else:
                    pre_seq = ""
                retr_num += 1
                docs, retr_quest = self._get_retr_docs_(question, ptexts, pre_seq, pre_seq_conf_type)
                _, new_text = self._generate_(docs=docs, demo=demo, question=question, ptext=ptext, generate_length=128)
                ptexts.append(new_text)
                ptext += (' ' + new_text)
                # docs = []
                """
                if 'retr_accpt' in self.__dict__ and self.retr_accpt:
                    cur_conf = 1
                else:
                    cur_conf, _ = self._get_confs_class_(question, ptext, new_text, _docs_)
                # 判断经过检索后新生成的句子是否置信度更高
                if cur_conf >= pre_seq_conf:
                    ptext += (' ' + new_text)
                    ptexts.append(new_text)
                    pconfs.append(cur_conf)
                    docs = _docs_
                else:
                    ptext += (' ' + pre_seq)
                    ptexts.append(pre_seq)
                    pconfs.append(pre_seq_conf)
                """
            ptext = ptext.strip()
            cur_len = len(self.generator.tokenizer.encode(ptext)) if ptext != "" else 0

            if "the answer is" in ptext or \
                    cur_len >= self.max_length or \
                    cur_len <= old_len or \
                    retr_num >= self.max_retrieve:
                idx, unknown = is_ans_unknown(ptexts)
                if len(ptexts)==0 or unknown:
                    ptext = ' '.join(ptexts[:idx])
                    # ptext = ''
                    unknown_info = ptexts[idx] if idx and idx >= 0 else ptext
                    unknown_info = unknown_info.strip() if len(unknown_info)>0 else ""
                    docs, _ = self._get_retr_docs_(question, [], unknown_info, 'lack knowledge')
                    _, new_text = self._generate_(docs, demo, question, ptext, generate_length=self.max_length)
                    ptext += (' ' + new_text)
                    ptext = ptext.strip()
                break
            old_len = cur_len

        return ptext
