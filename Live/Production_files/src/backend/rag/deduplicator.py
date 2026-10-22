# Purpose:
# Learn how duplicate chunks are removed after retrieval.
#
# What to focus on:
# - Why repeated chunks are bad for the LLM
# - Why a simple dictionary can remove exact duplicates quickly
# - Why deduplication improves token usage and answer quality
#
# TODO:
# 1. Explain why page_content is used as the dictionary key.
# 2. Describe what kind of duplicates this removes.
# 3. Suggest one limitation of this approach.

from src.backend.logger import GLOBAL_LOGGER as log


def remove_deduplicated_documents(documents):
    # Hint: this removes exact duplicates only.
    # It does not detect paraphrases or semantically similar chunks.
    try:
        unique_docs = list({doc.page_content: doc for doc in documents}.values())
        return unique_docs
    except Exception as e:
        log.error("Error during document deduplication", error=str(e))
        raise e
