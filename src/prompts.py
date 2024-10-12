ANSWER_QUESTION_TEMPLETE = """{docs}Please answer the question by thinking step-by-step in fully English.{examples}{use_demo}{use_continue}

### Question: {question}
### Answer: {gen_text}"""

ANSWER_USE_DEMO_TEMPLATE = """ And I hope you can answer the question in the same format as the examples."""
ANSWER_NOT_USE_DEMO_TEMPLATE = """ I hope you can only provide the reasoning and answer. Please note if the Answer is sufficient to resolve the Question, immediately stop reasoning and provide the answer, starting with "So, the answer is """

ANSWER_USE_DOCS_TEMPLATE = """ Please step-by-step through the process of considering what kind of knowledge is needed to answer the question. If such knowledge is present in the document, please utilize it to answer the Question. If not, please disregard the docs."""

CONTINUE_ANSWER_TEMPLATE = """ If more detail is needed, continue reasoning and then provide the answer, aiming to conclude promptly once the Answer is sufficient to resolve the Question. Avoid adding any unrelated phrases, such as "Let me help," "Let's analyze the information," or similar expressions. Additionally, ensure that you trust Answer and do not repeat once the Answer has been provided."""

START_ANSWER_TEMPLATE = """ And Please do not provide anything unrelated to reasoning, such as "Let's me help.", "Let's analyze the information provided", "I"d be delighted to assist!" and so on."""

STEP_REASON_ANSWER_TEMPLATE = """If the document contains information relevant to the current resoning path, please respond based on that information. And If the document does not contain content related to the current resoning path, return "None".
{docs}
### Resoning path: {reasoning}
### Answer:"""

CONFIDENCE_TEMPLATE = """Confucius said, "To know what you know and to know what you do not know, that is true knowledge." I believe you have true knowledge, and I will provide you with "Context" and "Your response" which you generated base on the "Context". {docs}
Please provide your score of confidence in "Your response" to demonstrate your familiarity with the relevant knowledge. Please note that the confidence is between 0 and 1, and the closer the value is to 1, the better your understanding of this knowledge. Please note that your confidence level is related to the "Your Response". Please provide confidence first, and then provide an explanation. {use_docs}

### Context: {context}
### Your Response: {response}
### Confidence:"""

CONFIDENCE_CLASS_TEMPLATE = """Analyse and generate a confidence level for the "Your Response" based on the "Context". Please note provide confidence level first, and then provide an explanation. What's more, if the evidence provided is insufficient, please lower your confidence level.
A. Very Certain
B. Fairly Certain
C. Lightly Certain
D. Not Certain
E. Very Uncertain

{docs}
### Context: {context}
### Your Response: {response}
### Confidence Level:"""

KEYWORDS_TEMPLATE = """Please use 2 to 3 keywords to express the idea behind this sentence.
### Sentence: {sentence}"""

LACK_KNOW_INFO_TEMPLATE = """Your response lacks sufficient relevant knowledge to answer the provided question. Please think deeply and generate a relevant query based on given the Response to find the knowledge. Please do not add any entities unrelated to given the Response. Please only generate a query, do not provide any other entities.
### Question: {question}
### Response: {response}
### Query:"""

HALLUICATION_INFO_TEMPLATE = """Your response contains hallucination information. Please think step-by-step and generate a relevant query based on the Question and Response, but avoid using any Entities terms from the response.
### Response: {response}
### Question: {question}
### Query:"""

ENTITY_TEMPLATE = """Please identify all entities in the given sentence. If there are no entities, respond with 'None'. If multiple entities are present, list them in the following format: 'Entity1, Entity2, Entity3'.

### Sentence: {sentence}
### Entities:"""


ENTITY_REPLEACE_TEMPLATE = """Replace the entity in the following sentence with another entity of the same type (e.g., if the entity is a female movie actress, replace it with another female actress) and return the modified sentence. If there is no entity in the sentence, return 'None'.
### Sentence: {sentence}
### Modified Sentence:"""

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