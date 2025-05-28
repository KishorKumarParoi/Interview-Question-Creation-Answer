prompt_template = """ 
You are an expert at creating questions based on coding resources, materials and documentation. Your goal is to prepare a coder or a programmer for job interview and coding tests. You do this by asking questions below.
---------------
{text}
---------------
Create questions that will prepare the coders or programmers for the Job Interview. Make sure not to looseany important information.
QUESTIONS:
"""

refine_template = """ 
You are an expert at creating questions based on coding resources, materials and documentation. Your goal is to prepare a coder or a programmer for job interview and coding tests.
We have received some practice questions to a certain extent: {existing_answer}. We have the option
to refine the existing questions or add new ones. (only if necessary) with some more context below.
---------------
{text}
---------------
Given the new context, refine the original questions in English. If the context is not helpful, please provide the original questions.
QUESTIONS:           
"""
