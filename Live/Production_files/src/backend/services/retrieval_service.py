# Student Guide: retrieval_service.py
#
# Purpose:
# Learn how the full retrieval pipeline is orchestrated.
#
# What to focus on:
# - How one service can support multiple retrieval strategies
# - Why the service deduplicates before answer generation
# - How evaluation is attached to the final answer
#
# TODO for students:
# 1. Explain the difference between each retriever_type branch.
# 2. Describe why deduplication happens before chain invocation.
# 3. List the evaluation metrics returned by this service.

from langchain_community.vectorstores import FAISS

from src.backend.rag.deduplicator import remove_deduplicated_documents
from src.backend.rag.chain import get_conversational_chain
from src.backend.rag.embeddings import get_embeddings
from src.backend.rag.compression import contextual_compression
from src.backend.rag.mmr import mmr
from src.backend.core.config import settings
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.multi_query_retriever import multi_query_retrieval

from langsmith import traceable


@traceable(
    run_type="chain",
    name="Retrieve Answer",
    project_name=settings.LANGCHAIN_PROJECT
)
def retrieve_answer(
    question: str,
    embedding_model: str,
    retriever_type: str,
    source_id: str = None
):
    """
    Online RAG pipeline:

    1. Initialize embeddings.
    2. Load FAISS index.
    3. Retrieve relevant documents.
    4. Optionally filter by source_id.
    5. Remove duplicate documents.
    6. Generate an answer using the LLM.
    7. Return answer and retrieved context.

    source_id=None:
        Search across all indexed documents.

    source_id=<id>:
        Search only the selected source document.

    Detailed evaluation is handled separately.
    """

    try:

        log.info(
            "Retrieval started",
            embedding_model=embedding_model,
            retriever_type=retriever_type,
            source_id=source_id
        )

        # --------------------------------------------------
        # Step 1: Initialize embeddings
        # --------------------------------------------------

        embeddings = get_embeddings(embedding_model)

        index_path = f"{settings.INDEX_PATH}_{embedding_model}"

        # --------------------------------------------------
        # Step 2: Retrieve documents
        # --------------------------------------------------

        if retriever_type == "FAISS Retriever":

            new_db = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            search_kwargs = {
                "k": 5
            }

            if source_id:

                search_kwargs["filter"] = {
                    "source_id": source_id
                }

            docs = new_db.similarity_search(
                query=question,
                **search_kwargs
            )

        elif retriever_type == "Contextual Compression Retriever":

            docs = contextual_compression(
                index_path,
                question,
                embeddings,
                source_id=source_id
            )

        elif retriever_type == "MMR Retriever":

            docs = mmr(
                index_path,
                question,
                embeddings,
                source_id=source_id
            )

        elif retriever_type == "Multi Query Retriever":

            docs = multi_query_retrieval(
                index_path,
                question,
                embeddings,
                source_id=source_id
            )

        else:

            raise ValueError(
                f"Unsupported retriever type: {retriever_type}"
            )

        # --------------------------------------------------
        # Step 3: Remove duplicate documents
        # --------------------------------------------------

        docs = remove_deduplicated_documents(docs)

        # --------------------------------------------------
        # Step 4: Validate source filtering
        # --------------------------------------------------

        if source_id:

            incorrect_sources = [
                doc for doc in docs
                if doc.metadata.get("source_id") != source_id
            ]

            if incorrect_sources:

                raise ValueError(
                    "Source filtering validation failed. "
                    "Retrieved documents contain unexpected sources."
                )

        # --------------------------------------------------
        # Step 5: Extract document content
        # --------------------------------------------------

        answer_texts = [
            doc.page_content for doc in docs
        ]

        # --------------------------------------------------
        # Step 6: Format context
        # --------------------------------------------------

        context = "\n\n".join(answer_texts)

        # --------------------------------------------------
        # Step 7: Initialize conversational chain
        # --------------------------------------------------

        chain = get_conversational_chain()

        # --------------------------------------------------
        # Step 8: Generate answer
        # --------------------------------------------------

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        log.info(
            "Answer generated successfully",
            retrieved_document_count=len(answer_texts),
            source_id=source_id
        )

        # --------------------------------------------------
        # Step 9: Evaluation status
        # --------------------------------------------------

        evaluation_metric = {
            "status": "not_run",
            "mode": "offline"
        }

        return response, answer_texts, evaluation_metric

    except Exception as e:

        log.error(
            "Error during Retrieval service",
            error=str(e)
        )

        raise