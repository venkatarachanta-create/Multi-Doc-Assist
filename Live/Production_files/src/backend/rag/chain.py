# Purpose:
# Learn how the prompt, LLM, and output parser are chained together.
#
# What to focus on:
# - Why the prompt includes both context and question
# - Why the chain stays small and reusable
# - Why output parsing is separated from generation
#
# TODO:
# 1. Explain the role of each pipe in the chain.
# 2. Change the prompt to be more strict and observe the impact.
# 3. Describe how this file supports answer generation.

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.llm import get_llm


def get_conversational_chain():
    # Hint: the chain is the answer-generation assembly line.
    try:
        prompt_template = """
        Question: \n{question}\n
        Answer the above question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n

        Answer:
        """

        llm = get_llm()

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = prompt | llm | StrOutputParser()
        log.info("Created Chain")
        return chain
    except Exception as e:
        log.error("Error creating conversational chain", error=str(e))
        raise e
