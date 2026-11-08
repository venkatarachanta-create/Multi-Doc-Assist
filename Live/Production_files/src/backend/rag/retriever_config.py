from typing import Optional


def get_faiss_retriever(
    vector_store,
    source_id: Optional[str] = None,
    k: int = 5
):
    """
    Create a FAISS retriever with optional source filtering.

    source_id=None:
        Search across all indexed documents.

    source_id=<id>:
        Search only the specified source document.
    """

    search_kwargs = {
        "k": k
    }

    if source_id:

        search_kwargs["filter"] = {
            "source_id": source_id
        }

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )