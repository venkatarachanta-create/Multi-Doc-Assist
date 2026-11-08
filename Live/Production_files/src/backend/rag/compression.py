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

from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.core.config import settings
from src.backend.rag.retriever_config import get_faiss_retriever


def contextual_compression(
    index_path,
    query,
    embeddings,
    source_id=None
):
    """
    Retrieve candidate documents using FAISS,
    optionally filter by source_id, and rerank
    using Cohere contextual compression.
    """

    try:

        log.info(
            "Contextual compression invoked",
            query=query,
            source_id=source_id
        )

        # 1. Load FAISS index
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        log.info("FAISS index loaded")

        # 2. Create source-aware base retriever
        base_retriever = get_faiss_retriever(
            vector_store=vector_store,
            source_id=source_id,
            k=20
        )

        # 3. Initialize Cohere reranker
        compressor = CohereRerank(
            model="rerank-v3.5",
            cohere_api_key=settings.COHERE_API_KEY,
            top_n=5
        )

        # 4. Apply contextual compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        compressed_docs = compression_retriever.invoke(query)

        log.info(
            "Compression retrieval completed",
            retrieved_documents=len(compressed_docs),
            source_id=source_id
        )

        return compressed_docs

    except Exception as e:

        log.error(
            "Error during contextual compression",
            error=str(e)
        )

        raise
