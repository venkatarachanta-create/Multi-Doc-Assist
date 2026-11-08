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
import hashlib
from pathlib import Path

from fastapi import UploadFile

from src.backend.rag.loader import load_document
from src.backend.rag.splitter import split_documents
from src.backend.rag.faiss_store import index_docs
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.core.config import settings
from langsmith import traceable


def calculate_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash for a document."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


@traceable(
    run_type="chain",
    name="Ingest Document",
    project_name=settings.LANGCHAIN_PROJECT
)
async def ingest_document(
    file: UploadFile,
    embedding_model: str
) -> int:

    try:
        # 1. Validate filename
        filename = Path(file.filename or "").name

        if not filename:
            raise ValueError("Invalid filename")

        allowed_extensions = {".pdf", ".docx", ".txt"}

        if Path(filename).suffix.lower() not in allowed_extensions:
            raise ValueError("Unsupported file type")

        # 2. Create input directory
        os.makedirs(settings.DATA_DIR, exist_ok=True)

        file_path = os.path.join(settings.DATA_DIR, filename)

        # 3. Save uploaded file
        content = await file.read()

        if not content:
            raise ValueError("Uploaded file is empty")

        with open(file_path, "wb") as f:
            f.write(content)

        log.info("File saved", file=filename)

        # 4. Calculate document identity
        file_hash = calculate_file_hash(file_path)

        source_id = hashlib.sha256(
            filename.lower().encode("utf-8")
        ).hexdigest()

        # 5. Load document
        docs = load_document(file_path)

        if not docs:
            raise ValueError("No readable content found")

        # 6. Split into chunks
        chunks = split_documents(docs)

        # 7. Attach document metadata
        for chunk_number, chunk in enumerate(chunks):

            chunk.metadata.update({
                "source": filename,
                "source_id": source_id,
                "file_hash": file_hash,
                "chunk_id": f"{source_id}_{chunk_number}"
            })

        # 8. Index document
        index_docs(chunks, embedding_model)

        log.info(
            "Document ingestion completed",
            file=filename,
            source_id=source_id,
            file_hash=file_hash,
            chunks=len(chunks)
        )

        return len(chunks)

    except Exception as e:

        log.error(
            "Error during ingestion service",
            error=str(e)
        )

        raise
