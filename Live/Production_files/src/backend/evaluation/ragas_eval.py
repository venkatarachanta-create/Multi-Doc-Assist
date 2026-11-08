# Purpose:
# This is where "the answer sounds right" becomes an actual number. Every
# function here scores one dimension of quality, using RAGAS metrics or a
# custom LLM-as-judge prompt.
#
# What to focus on (read this once, then look for the SAME pattern seven
# times — that repetition is the whole design):
#   1. Build a SingleTurnSample: (question, generated answer, retrieved
#      context, and sometimes a REFERENCE_ANSWER from config.py).
#   2. Wrap get_llm() in a LangchainLLMWrapper so RAGAS can call it as
#      "the judge."
#   3. Run the async .single_turn_ascore(sample) via asyncio.run(main()).
# Once you can name these three steps from memory, you understand every
# function below — they only differ in which RAGAS metric class they use.
#
# TODO:
# 1. Faithfulness vs. Hallucination: evaluate_hallucination() literally
#    calls Faithfulness() internally and does `1 - score`. Why compute a
#    "new" metric this way instead of duplicating logic?
# 2. evaluate_llm_judge() is the ONE function that doesn't use RAGAS at
#    all — it's a hand-written prompt asking the LLM to grade the answer
#    directly. What can a custom judge check that a RAGAS metric can't?
#    What's the tradeoff (cost, consistency, reproducibility)?
# 3. Every function wraps its body in try/except and logs the error before
#    returning it. Notice they `return e`, not `raise e`, unlike
#    ingestion_service.py and retrieval_service.py. Why the difference —
#    what would happen to a whole answer if one of seven eval metrics
#    threw an exception instead of failing quietly?
# 4. Pick one metric (e.g. evaluate_faithfulness) and change its threshold
#    or prompt. Run the same question through twice — does the score move
#    the way you expected?

from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample

from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
    Faithfulness,
    LLMContextRecall,
    FactualCorrectness
)

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.llm import get_llm

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------
# Helper: Initialize RAGAS evaluator LLM
# ---------------------------------------------------------

def get_evaluator_llm():

    llm = get_llm()

    return LangchainLLMWrapper(llm)


# ---------------------------------------------------------
# 1. Context Precision
# ---------------------------------------------------------

async def evaluate_context_precision(
    query,
    response,
    retrieved_context
):

    log.info("Evaluating Context Precision")

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_context
    )

    evaluator_llm = get_evaluator_llm()

    metric = LLMContextPrecisionWithoutReference(
        llm=evaluator_llm
    )

    return await metric.single_turn_ascore(sample)


# ---------------------------------------------------------
# 2. Response Relevancy
# ---------------------------------------------------------

async def evaluate_response_relevancy(
    query,
    response,
    embeddings,
    retrieved_context
):

    log.info("Evaluating Response Relevancy")

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_context
    )

    evaluator_llm = get_evaluator_llm()

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        embeddings
    )

    metric = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    return await metric.single_turn_ascore(sample)


# ---------------------------------------------------------
# 3. Context Recall
# ---------------------------------------------------------

async def evaluate_context_recall(
    query,
    response,
    retrieved_context,
    reference_answer
):

    log.info("Evaluating Context Recall")

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        reference=reference_answer,
        retrieved_contexts=retrieved_context
    )

    evaluator_llm = get_evaluator_llm()

    metric = LLMContextRecall(
        llm=evaluator_llm
    )

    return await metric.single_turn_ascore(sample)


# ---------------------------------------------------------
# 4. Factual Correctness
# ---------------------------------------------------------

async def evaluate_factual_correctness(
    query,
    response,
    reference_answer
):

    log.info("Evaluating Factual Correctness")

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        reference=reference_answer
    )

    evaluator_llm = get_evaluator_llm()

    metric = FactualCorrectness(
        llm=evaluator_llm
    )

    return await metric.single_turn_ascore(sample)


# ---------------------------------------------------------
# 5. Faithfulness
# ---------------------------------------------------------

async def evaluate_faithfulness(
    query,
    response,
    retrieved_context
):

    log.info("Evaluating Faithfulness")

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_context
    )

    evaluator_llm = get_evaluator_llm()

    metric = Faithfulness(
        llm=evaluator_llm
    )

    return await metric.single_turn_ascore(sample)


# ---------------------------------------------------------
# 6. Custom LLM-as-a-Judge
# ---------------------------------------------------------

async def evaluate_llm_judge(
    query,
    response,
    retrieved_context
):

    log.info("Evaluating using LLM Judge")

    llm = get_llm()

    context_str = "\n".join(
        [f"- {context}" for context in retrieved_context]
    )

    prompt = PromptTemplate(
        input_variables=[
            "query",
            "context_str",
            "response"
        ],
        template="""
You are an expert RAG evaluation judge.

Evaluate the generated answer using the following criteria:

1. Directness:
   Does the answer address the user's question?

2. Grounding:
   Is the answer supported by the retrieved context?

3. Completeness:
   Does it include the relevant information available in the context?

4. Precision:
   Does it avoid irrelevant information?

User Question:
{query}

Retrieved Context:
{context_str}

Generated Answer:
{response}

Return ONLY valid JSON:

{{
    "relevancy_percentage": 0,
    "explanation": "Brief explanation",
    "hallucination_detected": false
}}
"""
    )

    chain = prompt | llm | JsonOutputParser()

    return await chain.ainvoke(
        {
            "query": query,
            "context_str": context_str,
            "response": response
        }
    )