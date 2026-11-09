from src.backend.services.retrieval_service import retrieve_answer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "OpenAI (text-embedding-3-small)"

SOURCE_IDS = {
    "Sanchay Plus":
        "913d436f75658de97cc616786637ebfb5680d83ce3fe9d38f33661e56562aa6b"
}

QUESTION = "What is the death benefit under Sanchay Plus?"

RETRIEVERS = [
    "FAISS Retriever",
    "Contextual Compression Retriever",
    "MMR Retriever",
    "Multi Query Retriever"
]


# --------------------------------------------------
# Test Function
# --------------------------------------------------

def test_retriever(retriever_type, source_id=None):

    mode = "SOURCE-SPECIFIC" if source_id else "GLOBAL"

    print("\n" + "=" * 65)
    print(f"TEST: {retriever_type} | MODE: {mode}")
    print("=" * 65)

    answer, answer_texts, evaluation_metric = retrieve_answer(
        question=QUESTION,
        embedding_model=EMBEDDING_MODEL,
        retriever_type=retriever_type,
        source_id=source_id
    )

    print("\nGenerated Answer:")
    print(answer)

    print("\nRetrieved Chunks:", len(answer_texts))

    return answer_texts


# --------------------------------------------------
# Main Execution
# --------------------------------------------------

def main():

    for retriever_type in RETRIEVERS:

        # Test 1: Global retrieval
        global_docs = test_retriever(
            retriever_type=retriever_type
        )

        print("Global retrieval completed.")

        # Test 2: Source-specific retrieval
        expected_source_id = SOURCE_IDS["Sanchay Plus"]

        filtered_docs = test_retriever(
            retriever_type=retriever_type,
            source_id=expected_source_id
        )

        assert filtered_docs, (
            f"No source-specific documents retrieved for {retriever_type}"
        )
        print("Source-specific retrieval completed.")

    print("\nALL RETRIEVER TESTS COMPLETED")


if __name__ == "__main__":
    main()