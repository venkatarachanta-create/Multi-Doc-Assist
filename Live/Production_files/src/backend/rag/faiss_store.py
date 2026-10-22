# Purpose:
# Learn how document chunks are stored in a FAISS vector index.
#
# What to focus on:
# - Why FAISS is used for similarity search
# - How the index is created the first time
# - How the existing index is updated with new chunks
#
# TODO:
# 1. Trace the difference between the first save and later updates.
# 2. Explain why the embedding dimension must match the FAISS index.
# 3. Describe why the index path changes with the embedding model.
# 4. What would happen below WITHOUT the empty-chunks guard, if loader.py
#    returned [] for a file (e.g. the empty-file or unsupported-type cases
#    in loader.py's _load_single_file)? Trace what add_texts([]) or
#    add_documents(documents=[]) would do, and why "warn and return early"
#    is safer than letting that happen silently.

from langchain_community.vectorstores import FAISS
from src.backend.logger import GLOBAL_LOGGER as log
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from src.backend.rag.embeddings import get_embeddings
from src.backend.core.config import settings


def index_docs(chunks, embedding_model: str):
    # Hint: this function both creates a new index and updates an existing one.
    #
    # Think about:
    # - What happens when the index already exists?
    # - Why do we use a separate folder per embedding model?
    if not chunks or len(chunks) == 0:
        log.warning("No chunks to index", embedding_model=embedding_model)
        return

    embeddings = get_embeddings(embedding_model)
    index_path = f"{settings.INDEX_PATH}_{embedding_model}"

    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        log.info("Loading existing FAISS index")
        new_texts = [doc.page_content for doc in chunks]
        vector_store.add_texts(new_texts)
        vector_store.save_local(index_path)
        log.info("FAISS index updated and saved", index_path=index_path)

    except Exception:
        # Hint: this branch handles the first-time setup.
        log.info(f"FAISS index not found for {embedding_model}, Creating a new FAISS index")

        sample_embedding = embeddings.embed_query("hello")
        index = faiss.IndexFlatL2(len(sample_embedding))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_store.add_documents(documents=chunks)
        vector_store.save_local(index_path)
        log.info("FAISS index created and saved at ", index_path=index_path)
