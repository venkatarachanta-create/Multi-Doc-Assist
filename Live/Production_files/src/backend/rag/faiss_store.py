from pathlib import Path

import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.embeddings import get_embeddings
from src.backend.core.config import settings


# --------------------------------------------------
# Find existing document IDs belonging to a source
# --------------------------------------------------

def get_existing_document_ids(vector_store, source_id: str):
    """
    Find all FAISS document IDs belonging to a source document.
    """

    existing_ids = []

    for doc_id in vector_store.index_to_docstore_id.values():

        document = vector_store.docstore.search(doc_id)

        if (
            not isinstance(document, str)
            and document.metadata.get("source_id") == source_id
        ):
            existing_ids.append(doc_id)

    return existing_ids


# --------------------------------------------------
# Validate existing FAISS index metadata
# --------------------------------------------------

def validate_existing_index(vector_store):
    """
    Ensure the existing FAISS index follows the current
    source-aware metadata structure.

    Prevents legacy documents from being mixed with
    newly indexed documents.
    """

    existing_documents = vector_store.docstore._dict

    legacy_documents = []

    required_fields = [
        "source_id",
        "chunk_id",
        "file_hash"
    ]

    for doc_id, document in existing_documents.items():

        missing_fields = [
            field
            for field in required_fields
            if not document.metadata.get(field)
        ]

        if missing_fields:
            legacy_documents.append({
                "doc_id": doc_id,
                "missing_fields": missing_fields
            })

    if legacy_documents:

        raise ValueError(
            f"Existing FAISS index contains "
            f"{len(legacy_documents)} documents with legacy "
            "or incomplete metadata. Rebuild this index "
            "before adding new documents."
        )

    log.info(
        "Existing FAISS index metadata validation passed",
        total_documents=len(existing_documents)
    )


# --------------------------------------------------
# Validate incoming chunks
# --------------------------------------------------

def validate_chunks(chunks):

    if not chunks:
        raise ValueError("No document chunks provided.")

    required_fields = [
        "source_id",
        "file_hash",
        "chunk_id"
    ]

    source_id = chunks[0].metadata.get("source_id")
    file_hash = chunks[0].metadata.get("file_hash")

    if not source_id or not file_hash:
        raise ValueError(
            "Missing source_id or file_hash in chunk metadata."
        )

    chunk_ids = []

    for index, chunk in enumerate(chunks):

        for field in required_fields:

            if not chunk.metadata.get(field):
                raise ValueError(
                    f"Chunk {index} is missing required metadata: {field}"
                )

        if (
            chunk.metadata["source_id"] != source_id
            or chunk.metadata["file_hash"] != file_hash
        ):
            raise ValueError(
                "Chunks contain inconsistent document metadata."
            )

        chunk_ids.append(chunk.metadata["chunk_id"])

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs detected in incoming documents."
        )

    return source_id, file_hash, chunk_ids


# --------------------------------------------------
# Main indexing function
# --------------------------------------------------

def index_docs(chunks, embedding_model: str):

    if not chunks:
        log.warning(
            "No chunks to index",
            embedding_model=embedding_model
        )
        return

    # Validate incoming document chunks
    source_id, file_hash, chunk_ids = validate_chunks(chunks)

    embeddings = get_embeddings(embedding_model)

    index_path = f"{settings.INDEX_PATH}_{embedding_model}"
    index_directory = Path(index_path)

    # Check whether an existing FAISS index is available
    index_exists = (
        index_directory.is_dir()
        and (index_directory / "index.faiss").is_file()
        and (index_directory / "index.pkl").is_file()
    )

    # --------------------------------------------------
    # Case 1: Load existing FAISS index
    # --------------------------------------------------

    if index_exists:

        log.info(
            "Loading existing FAISS index",
            index_path=index_path
        )

        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Validate metadata before modifying the index
        validate_existing_index(vector_store)

        # Find existing chunks belonging to this document
        existing_ids = get_existing_document_ids(
            vector_store,
            source_id
        )

        # Check whether the uploaded document is unchanged
        if existing_ids:

            existing_document = vector_store.docstore.search(
                existing_ids[0]
            )

            existing_hash = existing_document.metadata.get(
                "file_hash"
            )

            if existing_hash == file_hash:

                log.info(
                    "Document already indexed. Skipping duplicate.",
                    source_id=source_id,
                    file_hash=file_hash
                )

                return

            # Updated document: remove its old chunks
            vector_store.delete(ids=existing_ids)

            log.info(
                "Previous document version deleted",
                source_id=source_id,
                deleted_chunks=len(existing_ids)
            )

    # --------------------------------------------------
    # Case 2: Create a new FAISS index
    # --------------------------------------------------

    else:

        log.info(
            "Creating new FAISS index",
            embedding_model=embedding_model
        )

        index_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        sample_embedding = embeddings.embed_query("hello")

        index = faiss.IndexFlatL2(
            len(sample_embedding)
        )

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    # --------------------------------------------------
    # Add document chunks
    # --------------------------------------------------

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    # --------------------------------------------------
    # Persist FAISS index
    # --------------------------------------------------

    vector_store.save_local(index_path)

    log.info(
        "FAISS indexing completed successfully",
        source_id=source_id,
        file_hash=file_hash,
        added_chunks=len(chunks),
        index_path=index_path
    )