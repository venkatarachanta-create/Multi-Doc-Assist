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
from src.backend.rag.embeddings import get_embeddings
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.core.config import settings


def mmr(index_path, query: str, embeddings):
    # Hint: MMR first loads the vector store, then asks for diverse but relevant chunks.
    try:
        log.info("MMR invoked", query=query)

        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        log.info("FAISS index loaded")

        docs = new_db.max_marginal_relevance_search(
            query=query,
            k=5,
            fetch_k=20,
            lambda_mult=0.7,
        )

        log.info("MMR documents retrieved", count=len(docs))

        return docs

    except Exception as e:
        log.error("Error during MMR retrieval", error=str(e))
        raise e
