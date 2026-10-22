# Purpose:
# Learn why every setting the app needs — API keys, model names, paths — is
# read from one place instead of being scattered across rag/ and services/.
#
# What to focus on:
# - Why pydantic-settings instead of just os.environ.get() everywhere
# - Where these values actually come from at runtime (.env locally, real
#   environment variables in Docker/Azure — see the "Deployment architecture"
#   diagram in the main README)
# - Why every field defaults to an empty string instead of crashing at import
#
# TODO:
# 1. Trace one setting end to end: OPENAI_API_KEY here -> where it's read in
#    llm.py -> where it's actually set (your local .env, or a docker run -e
#    flag, or an Azure App Setting).
# 2. Explain why OPENAI_MODEL has no default value like APP_NAME does. What
#    breaks if it's left unset? (Hint: ask about llm.py's ChatOpenAI call.)
# 3. This class is imported as a single `settings` object at the bottom.
#    Why does the rest of the app import `settings`, not `Settings`?

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration settings.

    Loads configuration from environment variables and .env file.
    Includes settings for model API keys, paths, and LangSmith integration.
    """
    APP_NAME: str = "Multi-Document Assist"
    INDEX_PATH: str = "data/faiss_index"
    DATA_DIR: str = "data/Input_files"

    # Hint: these two have no default — a missing OPENAI_MODEL doesn't crash
    # at startup, it crashes on the *first question* with a confusing OpenAI
    # error. Compare that to a hardcoded model name: worse for flexibility,
    # but you'd never hit this particular failure mode.
    OPENAI_MODEL: str = Field("", env="OPENAI_MODEL")
    OPENAI_EMBEDDING_MODEL: str = Field("", env="OPENAI_EMBEDDING_MODEL")
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")
    COHERE_API_KEY: str = Field("", env="COHERE_API_KEY")

    LANGSMITH_API_KEY: str = Field("", env="LANGSMITH_API_KEY")
    LANGCHAIN_PROJECT: str = Field("", env="LANGCHAIN_PROJECT")

    REFERENCE_QUESTION: str = "what is the revenue increase in FY24?"  # Not used in code
    REFERENCE_ANSWER: str = "Revenue increased to USD 120.0 million in FY2024"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Hint: this is created ONCE, at import time. Every file that does
# `from src.backend.core.config import settings` gets the same instance —
# that's what "one place" means in practice, not just in theory.
settings = Settings()
