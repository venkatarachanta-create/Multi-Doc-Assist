# Purpose:
# Learn how text is converted into vectors for semantic search.
#
# What to focus on:
# - Why embeddings are needed in RAG
# - How model selection changes cost and quality
# - Why configuration should come from settings, not hardcoded values
#
# TODO:
# 1. Identify the supported embedding providers.
# 2. Explain why the function returns different classes for different models.
# 3. Replace the sample model names with your own test values if needed.

from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from src.backend.core.config import settings


def get_embeddings(model_type: str):
    # Hint: this function is a model selector.
    # It returns the correct embedding object based on the user's choice.
    #
    # Think about:
    # - Why do we need this conditional logic?
    # - How does the selected model affect the FAISS index?
    # - Which API key is required for each provider?
    if model_type == "OpenAI (text-embedding-3-large)":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
    elif model_type == "Cohere (embed-english-v3.0)":
        embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model="embed-english-v3.0",
        )
    elif model_type == "OpenAI (text-embedding-3-small)":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )
    else:
        # Hint: a production version should handle unsupported models clearly.
        raise ValueError(f"Unsupported embedding model: {model_type}")

    return embeddings
