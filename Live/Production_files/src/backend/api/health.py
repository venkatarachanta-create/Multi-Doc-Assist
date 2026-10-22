# Purpose:
# The smallest file in the project — and one of the most important at
# deploy time.
#
# What to focus on:
# - This endpoint has no RAG logic at all. It just proves the process is
#   alive and able to respond to HTTP requests.
# - entrypoint.sh polls this exact endpoint in a loop before starting
#   Streamlit — search for "/health" there.
#
# 1. Why does startup order matter here? What would happen if Streamlit
#    started before FastAPI could answer /health?
# 2. This always returns {"status": "ok"} — it never checks whether the
#    FAISS index exists, or whether the OpenAI API key actually works.
#    Is that a bug? What would you check if you were designing a stricter
#    health check, and what would break if you did?

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}
