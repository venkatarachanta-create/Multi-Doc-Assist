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
from src.backend.core.config import settings


def multi_query_retrieval(index_path, query: str, embeddings):
    # Hint: this function uses the LLM to generate alternative queries.
    try:
        log.info("Multi Query Retriever Invoked")

        llm = get_llm()

        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        log.info("FAISS index loaded")

        new_db_retriever = new_db.as_retriever(search_kwargs={"k": 3})

        custom_prompt = PromptTemplate(
            input_variables=["question"],
            template="Generate 3 different versions of this question: {question}",
        )

        multi_query_retriever_custom = MultiQueryRetriever.from_llm(
            retriever=new_db_retriever,
            llm=llm,
            prompt=custom_prompt,
            include_original=False,
        )

        relevant_docs = multi_query_retriever_custom.invoke(query)

        log.info("Multi Query Retriever Docs fetched", docs=relevant_docs)

        return relevant_docs

    except Exception as e:
        log.error("Error during Multi Query Retrieval", error=str(e))
        raise e
