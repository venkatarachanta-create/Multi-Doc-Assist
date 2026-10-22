# Purpose:
# Learn how documents are loaded from PDF, DOCX, and TXT files.
#
# What to focus on:
# - Different file formats need different loaders
# - The loader returns a list of Document objects
# - Each Document keeps text plus metadata such as source and page number
# - load_document() accepts EITHER a directory OR a single file path — that
#   matters for ingestion_service.py's TODO #4 below.
#
# TODO:
# 1. Identify which loader is used for each file type.
# 2. Explain why metadata is important.
# 3. _load_single_file() checks `os.path.getsize(file_path) == 0` before
#    doing anything else. What would happen without that check if a student
#    uploaded a 0-byte file?
# 4. Add support for another file type as an exercise if you want.

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from src.backend.logger import GLOBAL_LOGGER as log


def _load_single_file(file_path):
    # Hint: every guard here returns [] instead of raising. Why is "load
    # nothing, keep going" the right behavior for one bad file in a batch,
    # instead of failing the whole ingestion?
    if os.path.getsize(file_path) == 0:
        log.warning("Skipping empty file", file_path=file_path)
        return []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()
    if file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        return loader.load()
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    log.warning("Skipping unsupported file type", file_path=file_path)
    return []


def load_document(directory_path):
    # Hint: this function does double duty — pass it a folder and it loads
    # every supported file inside; pass it one file's path and it loads
    # just that file. See ingestion_service.py for which one it's called
    # with, and why that choice matters for a growing FAISS index.
    try:
        documents = []
        if os.path.isfile(directory_path):
            return _load_single_file(directory_path)

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            documents.extend(_load_single_file(file_path))

        return documents
    except Exception as e:
        log.error("Error loading documents from directory", error=str(e), directory=directory_path)
        raise e
