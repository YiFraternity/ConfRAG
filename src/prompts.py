ANSWER_QUESTION_TEMPLETE = """{examples}{docs}{use_docs}{use_demo}{use_continue}
Question: {question}
Answer: {gen_text}"""
ANSWER_USE_DOCS_TEMPLATE = "Please answer the question based on the above documents."
ANSWER_NOT_USE_DOCS_TEMPLATE = """Please answer the question by reasoning step-by-step."""

ANSWER_USE_DEMO_TEMPLATE = """And I hope you can answer the question in the same format as the examples"""
ANSWER_NOT_USE_DEMO_TEMPLATE = """And I expect you to provide answers in a format consistent with the question and to only provide the answer without including the question. Additionally, please prefix each answer with "So, the answer is"."""

CONTINUE_ANSWER_TEMPLATE = """. If more detail is needed, continue reasoning follewed Anaswer and without deviating from the question, and aim to conclude promptly once the answer is fully developed. What's more, if the answer is complete, conclude with "So, the answer is" indicating a definitive response. And Please do not provide anything unrelated to reasoning, such as such as "Let's me help.", "Let's analyze the information provided", "I'd be delighted to assist!" and so on. And please proceed with reasoning based on the provided Answer and do not repeat the content of the Answer. \n"""

NOT_CONTINUE_ANSWER_TEMPLATE = """, (i.e, ending with "So, the answer is"). And Please do not provide anything unrelated to reasoning, such as "Let's me help.", "Let's analyze the information provided", "I'd be delighted to assist!" and so on."""

CONFIDENCE_TEMPLATE = """Confucius said, 'To know what you know and to know what you do not know, that is true knowledge.' I believe you have true knowledge, and I will provide you with "Context" and "Your response" which you generated base on the "Context". {docs}
Please provide your score of confidence in "Your response" to demonstrate your familiarity with the relevant knowledge. Please note that the confidence is between 0 and 1, and the closer the value is to 1, the better your understanding of this knowledge. Please note that your confidence level is related to the "Your Response". Please provide confidence first, and then provide an explanation. {use_docs}

Context:
{context}
Your Response:
{response}
Confidence:"""

CONFIDENCE_CLASS_TEMPLATE = """Confucius said, 'To know what you know and to know what you do not know, that is true knowledge.' I believe you have true knowledge, and I will provide you with "Context" and "Your response" which you generated base on the "Context". {docs}
Analyse its answer given other options. What level of confidence do you have in "Your response".
A. High Confidence
B. Medium Confidence
C. Low Confidence
Please note that your confidence level is related to the "Your Response". Please provide confidence first, and then provide an explanation. {use_docs}

Context:
{context}
Your Response:
{response}
Confidence Level:"""

KEYWORDS_TEMPLATE = """Please use 2 to 3 keywords to express the idea behind this sentence.
Sentence: {sentence}"""

CONFIDENCE_USE_DOCS_SUFFIX = """Above are the documents related to the "Context". """
CONFIDENCE_USE_DOCS_PREFIX = """Below are the documents related to the "Context"."""
CONFIDENCE_USE_DOCS = """When you provide your confidence in "Your Response", please refer to the documents."""
CONFIDENCE_USE_DOCUS_TEMPLATE = "Please provide your confidence in your response based on the above documents.\n"
#"""Confucius said, 'To know what you know and to know what you do not know, that is true knowledge.'
#I believe you have true knowledge, and I will provide you with a specific context and your response. Please provide your score of confidence in your response to demonstrate your familiarity with the relevant knowledge. Please note that the score of confidence is between 0 and 1, and the closer the value is to 1, the better your understanding of this knowledge.
#Context: {context}
#Your Response: {response}
#Confidence:"""

#"""You're a Q&A reasoning master, and you can judge the confidence level of an answer based on context. I'm going to give you a pair of clues next, including "Contexter" and your Respence. And all you have to do is give each sentence based on Contekste's confidence. Please note that the score of confidence is between 0 and 1, and the closer the value is to 1, the better your understanding of this knowledge.
#The template is as follows: Context: {context}
#Your Response: {response}
#Confidence:"""


# CONFIDENCE_INSTRUCTION = """Below you'll find contexts submitted by the user along with the responses your own generated. You should provide a confidence level for your responses, rated on a scale from 0 to 1. A higher score reflects a greater level of confidence in the accuracy of the generated responses. Please include your confidence estimate with each response you provide.
# Question:{question}
# Context:{context}
# Response:{response}
# Confidence:"""


# REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
# 以下反思提供了一个计划，以避免无法像以前那样回答问题。
# 使用它们来改进你正确回答给定问题的策略。
# REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
# 你以前曾尝试回答以下问题，但失败了。以下是你试图回答问题的最后一次尝试。n
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'



# 你以前曾试图回答以下问题，但失败了。以下反思提供了一个计划，
# 以避免无法像以前那样回答问题。使用它们来改进你正确回答给定问题的策略。

# 作为老师，我将为你提供学生提出的问题、他们之前的推理步骤，以及他们最后一次失败的回复。请仔细阅读并分析学生的推理过程和失败的回复。识别出学生在推理过程中出现的错误，并提供建设性的反馈，帮助学生理解错误的根源，从而改进其推理能力。
TUTOR_ADVICE_HEADER = "I hope you can provide advice to help student as a knowledgeable tutor. I will give you a question, the student's previous reasoning content for the question, and their most recent fail responses."
TUTOR_USE_DOCS = """Below are the documents referred to by student while answering questions."""
TUTOR_USE_DOCS_MIDDLE = """Please provide "Advice" in the same format as the examples based on above documents."""
TUTOR_NOT_USE_DOCS_MIDDLE = """Please provide "Advice" in the same format as the examples."""
TUTOR_ADVICE_MIDDLE = "What's more, please carefully analyze the reason of failed responses and offer constructive advice to help them understand the root causes of these errors and improve their response skills."
ADVICE_TEMPLATE = """{header}

Examples:
{examples}

{docs}{middle}
Question: {question}
Previous reasoning: {history_resp}
Fail Response: {response}
Advice:"""



# 你是一个能够自我反思和持续改进的高级推理代理。
# 每道题都会为你提供一个问题和之前试验。仔细阅读问题和之前试验，以及提供的建议，并通过思考来改善试验。
# 思考可以对当前情况进行推理，返回答案并完成任务。
REFLECTION_HEADER = """You’re an advanced reasoning agent capable of self-reflection and continuous improvement. Each problem will provide you with a question, previous excellent responses, and the last failed response."""
REFLECT_USE_DOC = """Below are the documents related to the question."""
REFLECT_USE_DOC_MIDDLE = """Based on the Documents provided above and the Adivce given below, """
REFLECT_NOT_USE_DOC_MIDDLE = """Based on the Adivce given below, """
REFLECTION_MIDDLE = """you can modify the "Fail Response" in the same format as the Examples."""
REFLECTION_TEMPLATE = """{header}

Examples:
{examples}

{docs}{middle}
Question: {question}
Previous reasoning: {history_resp}
Fail Response: {response}
Advice: {tutor_ins}
Modified Response:"""