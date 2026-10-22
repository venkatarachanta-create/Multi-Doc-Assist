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
from ragas import SingleTurnSample  # user_input, response, retrieved_contexts, reference
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness, LLMContextRecall, FactualCorrectness
import asyncio
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.llm import get_llm
from src.backend.rag.embeddings import get_embeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.backend.core.config import settings


def evaluate_hallucination(query, response, retrieved_context):
    """
    Measures if the answer contains information NOT present in the context.
    Ragas uses 'Faithfulness' for this. A low Faithfulness score = High Hallucination.
    """
    try:
        log.info('Checking for Hallucinations')
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )
        async def main():
            llm = get_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            # Faithfulness: Is the answer derived solely from the context?
            metric = Faithfulness(llm=evaluator_llm)
            score = await metric.single_turn_ascore(sample)

            # Hallucination Score can be interpreted as 1 - Faithfulness
            hallucination_score = 1 - score
            return {
                "hallucination_detected": hallucination_score > 0.1,  # Threshold
                "hallucination_score": hallucination_score
            }

        return asyncio.run(main())
    except Exception as e:
        log.error("Error during hallucination check", error=str(e))
        return e


def evaluate_factual_correctness(query, response):
    # Hint: this is the only metric that needs settings.REFERENCE_ANSWER —
    # it's checking the answer against a known-correct reference, not just
    # against the retrieved context. What does that mean for how useful
    # this metric is on a real user question with no reference answer?
    try:
        log.info('Evaluating Factual Correctness')
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            reference=settings.REFERENCE_ANSWER,
        )
        async def main():
            llm = get_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            # AnswerCorrectness weights semantic similarity and factual correctness
            correctness = FactualCorrectness(llm=evaluator_llm)
            result = await correctness.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        log.error("Error during evaluating Factual Correctness", error=str(e))
        return e


def evaluate_context_recall(query, response, retrieved_context):
    try:
        log.info('Evaluating Contextual Recall')
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            reference=settings.REFERENCE_ANSWER,
            retrieved_contexts=retrieved_context,
        )
        async def main():
            llm = get_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            recall = LLMContextRecall(llm=evaluator_llm)
            result = await recall.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        log.error("Error during evaluating Contextual Recall", error=str(e))
        return e


def evaluate_faithfulness(query, response, retrieved_context):

    try:
        log.info('Evaluating Faithfulness')
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )
        async def main():
            llm = get_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            faithfulness = Faithfulness(llm=evaluator_llm)
            result = await faithfulness.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        log.error("Error during evaluating faithfulness", error=str(e))
        return e


def evaluate_context_precision(query, response, retrieved_context):

    try:
        log.info('Evaluating Context Precision')
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )
        async def main():
            llm = get_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
            result = await context_precision.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        log.error("Error during evaluating Context Precision", error=str(e))
        return e


def evaluate_response_relevancy(query, response, embeddings, retrieved_context):
    # Hint: the only metric here that ALSO needs an embeddings wrapper, not
    # just an LLM wrapper. Why would relevancy scoring need embeddings when
    # faithfulness/precision/recall don't?
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context
        )

        async def main():
            llm = get_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            embedding_model = embeddings
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
            scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
            result = await scorer.single_turn_ascore(sample)
            return result
        return asyncio.run(main())

    except Exception as e:
        log.error("Error during evaluating response relevancy", error=str(e))
        return e


def evaluate_llm_judge(query, response, retrieved_context):
    # Hint: no SingleTurnSample, no RAGAS metric class — this is a plain
    # prompt | llm | JsonOutputParser chain, same shape as chain.py's
    # get_conversational_chain(). The "evaluator" is just another LLM call.
    llm = get_llm()

    context_str = "\n".join([f"- {c}" for c in retrieved_context])

    prompt = PromptTemplate(
        input_variables=["query", "context_str", "response"],
        template="""
                You are an expert Evaluator. Your task is to judge the RELEVANCY of an Answer based on a User Question and the Provided Context.

                ### Evaluation Criteria:
                1. **Directness**: Does the answer directly address the user's question?
                2. **Grounding**: Is the answer supported by the retrieved context?
                3. **Completeness**: Does it provide all necessary details found in the context to answer the question?
                4. **Precision**: Does it avoid irrelevant fluff?

                ### Input Data:
                **User Question:** {query}
                **Retrieved Context:** {context_str}
                **Generated Answer:** {response}

                ### Output Format:
                You must respond ONLY in the following JSON format:
                {{
                    "relevancy_percentage": <integer between 0 and 100>,
                    "explanation": "<short 2-sentence explanation>",
                    "hallucination_detected": <true/false>
                }}
                """
                    )

    try:
        # chain = prompt | llm | StrOutputParser()
        chain = prompt | llm | JsonOutputParser()

        raw_result = chain.invoke({
            "query": query,
            "context_str": context_str,
            "response": response
        })

        return raw_result

    except Exception as e:
        log.error("Error during evaluating llm-as-a-judge", error=str(e))
        return e
