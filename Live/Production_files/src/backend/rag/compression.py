# Purpose:
# Learn how contextual compression filters and reranks retrieved chunks.
#
# What to focus on:
# - The first retriever may return too many chunks
# - A reranker can keep only the chunks most relevant to the query
# - This improves precision and reduces noise
#
# TODO:
# 1. Explain why a reranker is useful after vector search.
# 2. Identify the role of CohereRerank.
# 3. Describe when this strategy is better than basic FAISS retrieval.

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.llm import get_llm
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
from src.backend.core.config import settings


def contextual_compression(index_path, query, embeddings):
    # Hint: this function wraps a base retriever with a compressor.
    # The base retriever finds candidate chunks.
    # The compressor/reranker narrows them down.
    try:
        log.info("Contextual compression invoked", query=query)

        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True).as_retriever()
        log.info("FAISS index loaded")

        compressor = CohereRerank(model="rerank-v3.5", cohere_api_key=settings.COHERE_API_KEY)

        # The reranker scores candidate chunks by query relevance.
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=new_db,
        )
        compressed_docs = compression_retriever.invoke(query)

        log.info("Compression Retriever fetched documents")
        return compressed_docs

    except Exception as e:
        log.error("Error during contextual compression", error=str(e))
        raise e
