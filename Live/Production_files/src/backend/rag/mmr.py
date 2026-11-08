# Purpose:
# Learn Maximum Marginal Relevance (MMR), which balances relevance and diversity.
#
# What to focus on:
# - Why retrieval should not repeat the same idea too many times
# - How lambda_mult controls the trade-off
# - Why diversity matters in multi-document answers
#
# TODO:
# 1. Explain what k means.
# 2. Explain what fetch_k means.
# 3. Describe what lambda_mult does in your own words.

from langchain_community.vectorstores import FAISS

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.retriever_config import get_faiss_retriever


def mmr(
    index_path,
    query: str,
    embeddings,
    source_id=None
):
    """
    Retrieve diverse and relevant documents using MMR,
    with optional source-level filtering.
    """

    try:

        log.info(
            "MMR retrieval invoked",
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

        # 2. Configure source-aware MMR retriever
        search_kwargs = {
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }

        if source_id:
            search_kwargs["filter"] = {
                "source_id": source_id
            }

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )

        # 3. Retrieve documents
        docs = retriever.invoke(query)

        log.info(
            "MMR retrieval completed",
            count=len(docs),
            source_id=source_id
        )

        return docs

    except Exception as e:

        log.error(
            "Error during MMR retrieval",
            error=str(e)
        )

        raise
