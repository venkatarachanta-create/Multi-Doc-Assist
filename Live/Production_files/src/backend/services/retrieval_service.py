# Student Guide: retrieval_service.py
#
# Purpose:
# Learn how the full retrieval pipeline is orchestrated.
#
# What to focus on:
# - How one service can support multiple retrieval strategies
# - Why the service deduplicates before answer generation
# - How evaluation is attached to the final answer
#
# TODO for students:
# 1. Explain the difference between each retriever_type branch.
# 2. Describe why deduplication happens before chain invocation.
# 3. List the evaluation metrics returned by this service.

from langchain_community.vectorstores import FAISS
from src.backend.rag.deduplicator import remove_deduplicated_documents
from src.backend.rag.chain import get_conversational_chain
from src.backend.rag.embeddings import get_embeddings
from src.backend.rag.compression import contextual_compression
from src.backend.rag.mmr import mmr
from src.backend.core.config import settings
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.multi_query_retriever import multi_query_retrieval
from tenacity import retry, stop_after_attempt, wait_exponential
from src.backend.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy, evaluate_context_recall, evaluate_factual_correctness, evaluate_faithfulness, evaluate_hallucination, evaluate_llm_judge
from langsmith import traceable


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
@traceable(run_type="chain", name="Retrieve Answer", project_name=settings.LANGCHAIN_PROJECT)
def retrieve_answer(question: str, embedding_model: str, retriever_type: str) -> str:
    # Hint: this function is the top-level answer pipeline.
    # It chooses one retrieval strategy, cleans results, generates an answer, and evaluates quality.
    try:

        log.info("Retrieval started", question=question, embedding_model=embedding_model)

        embeddings = get_embeddings(embedding_model)
        index_path = f"{settings.INDEX_PATH}_{embedding_model}"

        # Method 1: basic FAISS similarity search
        if retriever_type == "FAISS Retriever":
            new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(question)

        # Method 2: contextual compression
        elif retriever_type == "Contextual Compression Retriever":
            docs = contextual_compression(index_path, question, embeddings)

        # Method 3: MMR
        elif retriever_type == "MMR Retriever":
            docs = mmr(index_path, question, embeddings)

        # Method 4: multi-query retrieval
        elif retriever_type == "Multi Query Retriever":
            docs = multi_query_retrieval(index_path, question, embeddings)
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        docs = remove_deduplicated_documents(docs)
        answer_texts = [doc.page_content for doc in docs]

        chain = get_conversational_chain()

        response = chain.invoke(
            {"context": docs, "question": question},
            return_only_outputs=True,
        )

        log.info("Answer generated")

        context_precision = evaluate_context_precision(question, response, answer_texts)
        response_relevancy = evaluate_response_relevancy(question, response, embeddings, answer_texts)
        contextual_recall = evaluate_context_recall(question, response, answer_texts)
        factual_correctness = evaluate_factual_correctness(question, response)
        faithfulness = evaluate_faithfulness(question, response, answer_texts)
        hallucination = evaluate_hallucination(question, response, answer_texts)
        llm_judge = evaluate_llm_judge(question, response, answer_texts)

        # Consolidating Evaluation Metrics
        evaluation_metric = {
            "context_precision": context_precision,
            "response_relevancy": response_relevancy,
            "contextual_recall": contextual_recall,
            "factual_correctness": factual_correctness,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "llm_judge": llm_judge,
        }

        return response, answer_texts, evaluation_metric

    except Exception as e:
        log.error("Error during Retrieval service", error=str(e))
        raise e
