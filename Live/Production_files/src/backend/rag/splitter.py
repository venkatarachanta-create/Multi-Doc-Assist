# Purpose:
# Learn how documents are broken into smaller chunks before indexing.
#
# What to focus on:
# - Why large documents must be split
# - Why overlap helps preserve meaning across boundaries
# - Why chunking affects retrieval quality
#
# TODO:
# 1. Explain why chunk_size=800 is a practical production choice.
# 2. Explain why chunk_overlap=50 helps with context continuity.
# 3. Try a smaller chunk size and observe the effect on retrieval.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.backend.logger import GLOBAL_LOGGER as log


def split_documents(documents):
    # Hint: RecursiveCharacterTextSplitter tries to split on natural boundaries first.
    # It prefers paragraphs and sentences before falling back to smaller pieces.
    log.info("Splitting Documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
    )
    split_docs = text_splitter.split_documents(documents)

    return split_docs
