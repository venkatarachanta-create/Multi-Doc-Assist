from langchain_community.vectorstores import FAISS

from src.backend.rag.embeddings import get_embeddings
from src.backend.rag.retriever_config import get_faiss_retriever
from src.backend.core.config import settings


# --------------------------------------------------
# Configuration
# --------------------------------------------------

embedding_model = "OpenAI (text-embedding-3-small)"

index_path = f"{settings.INDEX_PATH}_{embedding_model}"

embeddings = get_embeddings(embedding_model)


# --------------------------------------------------
# Load FAISS Index
# --------------------------------------------------

print("\n========== LOADING FAISS INDEX ==========")

db = FAISS.load_local(
    index_path,
    embeddings,
    allow_dangerous_deserialization=True
)

print("Total vectors:", db.index.ntotal)


# --------------------------------------------------
# Source IDs from Existing Metadata Audit
# --------------------------------------------------

source_ids = {
    "Group Term Life":
        "446c38e681f69a6698374d69c55b5c82396e0bcd59541f1a99083e246a359bb8",

    "Sampoorna Jeevan":
        "57370869c10c166678b354b860140b7a15c03faace121c015a7cf26e379b7853",

    "Sanchay Plus":
        "913d436f75658de97cc616786637ebfb5680d83ce3fe9d38f33661e56562aa6b"
}


# --------------------------------------------------
# Test 1: Source-Specific Retrieval
# --------------------------------------------------

print("\n========== TEST 1: SOURCE FILTERING ==========")

question = "What is the death benefit under Sanchay Plus?"

selected_source_id = source_ids["Sanchay Plus"]

retriever = get_faiss_retriever(
    vector_store=db,
    source_id=selected_source_id,
    k=5
)

filtered_docs = retriever.invoke(question)

print("Question:", question)
print("Expected source: Sanchay Plus")
print("Retrieved chunks:", len(filtered_docs))

for i, doc in enumerate(filtered_docs, start=1):

    print(f"\nRank {i}")

    print("Source:", doc.metadata.get("source"))
    print("Source ID:", doc.metadata.get("source_id"))
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:250])

    assert (
        doc.metadata.get("source_id") == selected_source_id
    ), "Source filtering failed!"


print("\nSOURCE FILTERING TEST PASSED")


# --------------------------------------------------
# Test 2: Global Retrieval
# --------------------------------------------------

print("\n========== TEST 2: GLOBAL RETRIEVAL ==========")

global_retriever = get_faiss_retriever(
    vector_store=db,
    source_id=None,
    k=5
)

global_docs = global_retriever.invoke(question)

assert global_docs, "Global retrieval returned no documents"

print("Retrieved chunks:", len(global_docs))

for i, doc in enumerate(global_docs, start=1):

    print(f"\nRank {i}")

    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:200])


print("\nGLOBAL RETRIEVAL TEST COMPLETED")