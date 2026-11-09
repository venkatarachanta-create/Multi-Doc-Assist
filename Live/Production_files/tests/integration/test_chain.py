from src.backend.rag.chain import get_conversational_chain


def main():

    print("\n--- CHAIN SMOKE TEST ---")

    try:
        # Test 1: Chain initialization
        print("\nTest 1: Initializing conversational chain...")

        chain = get_conversational_chain()

        print("PASS: Chain initialized successfully.")

        # Test 2: Answer generation
        print("\nTest 2: Testing answer generation...")

        context = """
        HDFC Life Group Term Life Policy:

        Accident means a sudden, unforeseen and involuntary event
        caused by external, visible and violent means.

        Accidental death covers death resulting directly from an
        accident and occurring within 90 days of the accident.
        """

        question = (
            "Explain the meaning of an accidental event as defined in the policy."
        )

        response = chain.invoke({
            "context": context,
            "question": question
        })

        print("\nGenerated Answer:")
        print(response)

        assert response and response.strip(), (
            "Empty response received."
        )

        print("\nPASS: Answer generated successfully.")

        # Test 3: Missing-context handling
        print("\nTest 3: Testing missing-context handling...")

        missing_context = """
        The policy document contains no information
        about maturity benefits.
        """

        missing_question = "What is the maturity benefit?"

        response = chain.invoke({
            "context": missing_context,
            "question": missing_question
        })

        print("\nGenerated Answer:")
        print(response)

        expected_fallback = (
            "Answer is not available in the provided context."
        )

        assert response.strip() == expected_fallback, (
            "Expected fallback response for missing information."
        )

        print("\nPASS: Missing-context handling works correctly.")

    except Exception as e:

        print("\nFAIL: Chain smoke test failed.")
        print(f"Error: {str(e)}")

        raise


if __name__ == "__main__":
    main()