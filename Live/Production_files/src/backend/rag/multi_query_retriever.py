# Purpose:
# Learn how query expansion helps retrieve documents that the original question may miss.
#
# What to focus on:
# - One question can be rewritten in multiple ways
# - Each reformulation may retrieve a different useful chunk
# - The results are merged and deduplicated later
#
# TODO:
# 1. Explain why the prompt asks for 3 different versions.
# 2. Describe why include_original=False is used here.
# 3. Compare this method with basic similarity search.

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.rag.llm import get_llm
from src.backend.rag.retriever_config import get_faiss_retriever


def multi_query_retrieval(
    index_path,
    query: str,
    embeddings,
    source_id=None
):
    """
    Generate alternative questions and retrieve relevant
    documents using source-aware FAISS retrieval.
    """

    try:

        log.info(
            "Multi Query Retriever invoked",
            query=query,
            source_id=source_id
        )

        # 1. Initialize LLM
        llm = get_llm()

        # 2. Load FAISS index
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        log.info("FAISS index loaded")

        # 3. Create source-aware base retriever
        base_retriever = get_faiss_retriever(
            vector_store=vector_store,
            source_id=source_id,
            k=3
        )

        # 4. Configure alternative query generation
        custom_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "Generate 3 different versions of this question "
                "to retrieve relevant information from documents.\n"
                "Preserve the original meaning and important terms.\n"
                "Return each alternative question on a separate line.\n\n"
                "Original question: {question}"
            ),
        )

        # 5. Initialize Multi Query Retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=custom_prompt,
            include_original=True
        )

        # 6. Retrieve relevant documents
        relevant_docs = multi_query_retriever.invoke(query)

        log.info(
            "Multi Query retrieval completed",
            retrieved_documents=len(relevant_docs),
            source_id=source_id
        )

        return relevant_docs

    except Exception as e:

        log.error(
            "Error during Multi Query retrieval",
            error=str(e)
        )

        raise
