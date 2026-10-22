# Student Guide: ingestion_service.py
#
# Purpose:
# Learn how the upload workflow is orchestrated from file save to FAISS indexing.
#
# What to focus on:
# - Why services sit above the rag building blocks
# - How the service coordinates loader -> splitter -> faiss_store
# - Why retry logic and tracing matter in production
#
# TODO for students:
# 1. Explain the ingestion pipeline in order.
# 2. Describe why the file is saved before loading.
# 3. Explain what retry and traceable do.
# 4. Real bug, actually hit while building this project: this used to call
#    `load_document(settings.DATA_DIR)` — the whole uploads folder — instead
#    of `load_document(file_path)` — just the file you just saved. With the
#    whole-folder version, uploading a 2nd document reloads and re-indexes
#    the 1st one too, adding a duplicate copy of it to the FAISS index every
#    single time you ingest something new. Explain why `load_document`
#    accepting a single file path (see loader.py) is what makes the current,
#    correct version possible.

import os
from fastapi import UploadFile
from src.backend.rag.loader import load_document
from src.backend.rag.splitter import split_documents
from src.backend.rag.faiss_store import index_docs
from src.backend.logger import GLOBAL_LOGGER as log
from tenacity import retry, stop_after_attempt, wait_exponential
from src.backend.core.config import settings
from langsmith import traceable


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
@traceable(run_type="chain", name="Ingest Document", project_name=settings.LANGCHAIN_PROJECT)
async def ingest_document(file: UploadFile, embedding_model: str) -> int:
    # Hint: this is the high-level ingestion workflow.
    # It should read like a checklist, not like low-level helper code.
    try:
        os.makedirs(settings.DATA_DIR, exist_ok=True)

        file_path = os.path.join(settings.DATA_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        log.info("File saved", file=file.filename)

        docs = load_document(file_path)
        chunks = split_documents(docs)

        index_docs(chunks, embedding_model)

        log.info("Documents indexed", chunks=len(chunks))
        return len(chunks)

    except Exception as e:
        log.error("Error during Ingestion service", error=str(e))
        raise e
