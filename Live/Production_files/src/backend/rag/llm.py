# Purpose:
# Learn how the answer-generating language model is configured.
#
# What to focus on:
# - Why the LLM is separated from the rest of the logic
# - How settings are used instead of hardcoded secrets
# - How one function can hide model configuration details from the rest of the app
#
# TODO:
# 1. Explain why this function has no business logic.
# 2. Identify where the API key comes from.
# 3. Describe why a single helper function is better than repeating LLM setup everywhere.

from langchain_openai import ChatOpenAI
from src.backend.core.config import settings


def get_llm():
    # Hint: this function creates the answer model used by the chain.
    # The goal is to keep model setup in one place.
    #
    # Ask yourself:
    # - What happens if the model name changes?
    # - Why is this easier to maintain than hardcoding ChatOpenAI in many files?
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    return llm
