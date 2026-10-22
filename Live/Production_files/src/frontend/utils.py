# Purpose:
# The frontend's only connection to the backend. Streamlit never imports
# rag/ or services/ directly — it talks HTTP, exactly like an external
# client would.
#
# What to focus on:
# - API_URL = "http://localhost:8000" — this looks like it should be wrong
#   (localhost?!) but it's correct. FastAPI and Streamlit run in the SAME
#   container (see the "Runtime architecture" diagram in the main README),
#   so "localhost" here means "this container," not "the user's laptop."
# - Every function returns None on failure instead of raising. Look at what
#   app.py does when it gets None back.
#
# TODO:
# 1. What would break if this were deployed as two SEPARATE containers
#    (frontend and backend each with their own IP)? What would API_URL
#    need to become?
# 2. retrieve_answer() returns a 3-tuple by unpacking two separate
#    response.json().get(...) calls. What happens to this function if the
#    backend request fails — does it still return a 3-tuple? Check what
#    app.py assumes about its return value.
# 3. REQUEST_TIMEOUT is 300 seconds. Why so long for what's "just" an HTTP
#    call? (Hint: what does retrieve_answer() on the backend actually do —
#    one API call, or several?)

import requests
import streamlit as st
from src.backend.logger import GLOBAL_LOGGER as log
from src.shared.schemas import QueryRequest, QueryResponse
from typing import Optional


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def upload_document(file, embedding_model: str):
    try:
        log.info("Uploading document: %s with model: %s", file.name, embedding_model)

        response = requests.post(
            f"{API_URL}/upload",
            files={"file": file},
            data={"embedding_model": embedding_model},
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        log.info("Document ingested successfully")
        return response.json().get("chunks", 0)

    except requests.exceptions.Timeout:
        log.error("Upload timed out")
        st.error("⏱️ Upload timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        log.exception("Upload failed")
        st.error(f"❌ Upload failed: {e}")

    return None


def retrieve_answer(question: str, embedding_model: str, retriever_type: str) -> Optional[str]:
    try:
        log.info("Retrieving answer for question: %s with model: %s", question, embedding_model)

        response = requests.post(
            f"{API_URL}/retrieve",
            json={"question": question, "embedding_model": embedding_model, "retriever_type": retriever_type},
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        return response.json().get("answer"), response.json().get("answer_texts"), response.json().get("evaluation_metric")

    except requests.exceptions.Timeout:
        log.error("Retrieval timed out")
        st.error("⏱️ Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        log.exception("Retrieval failed")
        st.error(f"❌ Failed to retrieve answer: {e}")

    return None
