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
    """
    Remove exact duplicate chunks while preserving
    identical content from different source documents.
    """

    try:
        unique_docs = []
        seen = set()

        for doc in documents:

            source_id = doc.metadata.get("source_id", "unknown")

            content = doc.page_content.strip()

            deduplication_key = (source_id, content)

            if deduplication_key not in seen:

                seen.add(deduplication_key)
                unique_docs.append(doc)

        log.info(
            "Document deduplication completed",
            original_count=len(documents),
            unique_count=len(unique_docs)
        )

        return unique_docs

    except Exception as e:

        log.error(
            "Error during document deduplication",
            error=str(e)
        )

        raise
