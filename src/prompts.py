ANSWER_QUESTION_TEMPLETE = """Examples:
{demo}
{docs}
Following the examples above, answer the question by reasoning step-by-step. Please note I want you to always conclude your responses with the phrase `So, the answer is ...`. Please ensure that this phrase summarizes the key point or answer to the question.
Question: {question}
Answer: {gen_text}"""

ANSWER_USE_DOCUS_TEMPLATE = "Please answer the question based on the above documents and the previous response.\n"

CONFIDENCE_TEMPLATE = """Confucius said, 'To know what you know and to know what you do not know, that is true knowledge.'
I believe you have true knowledge, and I will provide you with a specific context and your response.
{docs}Please provide your score of confidence in your response to demonstrate your familiarity with the relevant knowledge. Please note that the score of confidence is between 0 and 1, and the closer the value is to 1, the better your understanding of this knowledge. Please note that your confidence level is related to the `Your Response`.
Context:
{context}
Your Response:
{response}
Confidence:"""
CONFIDENCE_USE_DOCUS_TEMPLATE = "Please provide your confidence in your response based on the above document.\n"
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
TUTOR_ADVICE_HEADER = "You are a knowledgeable tutor. I hope you can provide advice to help students as a tutor. I will give you a question, the student's previous reasoning content for the question, and their most recent fail responses."
TUTOR_ADVICE_MIDDLE = "Please provide `Advice` in the same format as before. what's more, please carefully analyze the reason of failed responses and offer constructive advice to help them understand the root causes of these errors and improve their response skills."
ADVICE_TEMPLATE = """{header}

(START OF EXAMPLES)
{examples}
(END OF EXAMPLES)

{middle}Question: {question}
Previous reasoning content:{context}
Fail Response: {response}
Advice:"""



# 你是一个能够自我反思和持续改进的高级推理代理。
# 每道题都会为你提供一个问题和之前试验。仔细阅读问题和之前试验，以及提供的建议，并通过思考来改善试验。
# 思考可以对当前情况进行推理，返回答案并完成任务。
REFLECTION_HEADER = """You’re an advanced reasoning agent capable of self-reflection and continuous improvement. Each problem will provide you with a question, previous excellent responses, and the last failed response."""
REFLECTION_MIDDLE = """Please provide `Reflection` in the same format as before, and follow the provided advice to modify the last failed response through reflection."""
REFLECTION_TEMPLATE = """{header}

(START OF EXAMPLES){examples}
(END OF EXAMPLES)

{middle}
Question: {question}
Previous reasoning content:{context}
Last Response:{response}
Advice: {tutor_ins}
Reflection:"""