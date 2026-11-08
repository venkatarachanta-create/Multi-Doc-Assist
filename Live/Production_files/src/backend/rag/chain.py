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
    # Answer-generation chain with strict context grounding.
    try:
        prompt_template = """
You are a precise document question-answering assistant.

Your task is to answer the user's question using ONLY the information
provided in the retrieved context.

Follow these rules carefully:

1. Context Grounding:
   - Use only facts explicitly supported by the provided context.
   - Do not use external knowledge, assumptions, or information from memory.
   - Do not invent figures, dates, conditions, definitions, or policy clauses.

2. Answer Relevance:
   - Answer exactly what the user asks.
   - Do not include unrelated policy information.
   - Avoid unnecessary explanations, repetition, or lengthy introductions.

3. Accuracy and Completeness:
   - Preserve important numbers, percentages, dates, time periods,
     conditions, exceptions, and limitations.
   - Include all relevant conditions when the question asks for them.
   - Do not omit important qualifications that change the meaning.

4. Handling Missing Information:
   - If the context does not contain sufficient information to answer,
     respond exactly:
     "Answer is not available in the provided context."
   - Do not guess or attempt to complete missing information.

5. Response Style:
   - Use clear, professional language.
   - Use numbered points when explaining multiple conditions.
   - Keep the answer concise while retaining all relevant details.

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

        llm = get_llm()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = prompt | llm | StrOutputParser()

        log.info("Created context-grounded conversational chain")
        return chain

    except Exception as e:
        log.error("Error creating conversational chain", error=str(e))
        raise
