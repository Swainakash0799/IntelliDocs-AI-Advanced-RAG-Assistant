from langchain_core.output_parsers import StrOutputParser
from config import llm
from prompts import retriever_prompt, answer_prompt
 
# -------------------------------
# Retriever Agent
# -------------------------------
# Takes the user's raw question and rewrites it into a focused search query.
 
retriever_agent = retriever_prompt | llm | StrOutputParser()
 
# -------------------------------
# Answer Agent
# -------------------------------
# Takes retrieved context + the question and produces the final answer.
 
answer_agent = answer_prompt | llm | StrOutputParser()