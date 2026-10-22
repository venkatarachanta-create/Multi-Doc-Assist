# Purpose:
# This is where the whole backend gets wired together. Nothing here does
# RAG work — it configures the FastAPI app and plugs in the pieces built
# everywhere else.
#
# What to focus on:
# - The order of operations at the top: env vars are set BEFORE the api
#   package is imported. That's not accidental — see the comment inline.
# - The logging middleware wraps every request, not just /upload and
#   /retrieve. That's how the log lines you see in Azure's Log Stream get
#   their method/path/status_code/duration fields.
# - app.include_router(...) is the actual moment routes.py's endpoints
#   become reachable at all.
#
# TODO:
# 1. Why must the os.environ[...] lines run before
#    `from src.backend.api import routes, health`? What would go wrong if
#    you moved the import to the top of the file?
# 2. This file never mentions port 8000 or 8080. Where is the port actually
#    decided? (Hint: it's not in this file at all — check entrypoint.sh.)
# 3. allow_origins=["*"] in CORSMiddleware is permissive. Why is that an
#    acceptable choice for this project, and when would it not be?

import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.backend.core.config import settings
from langsmith import traceable

# Setup Environment Variables (MUST BE BEFORE OTHER IMPORTS)
# Hint: routes.py imports services, which import rag/chain.py's get_llm(),
# which reads settings.LANGSMITH_API_KEY. LangSmith needs these as real
# process env vars (not just pydantic settings) to start tracing — so they
# have to be set before anything downstream gets imported and runs.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

from src.backend.api import routes, health
from src.backend.logger import GLOBAL_LOGGER as log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Hint: this runs for every single request, before and after it's
    # handled — that's what "middleware" means. call_next(request) is where
    # the actual endpoint (upload/retrieve/health) runs.
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    log.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=f"{process_time:.4f}s"
    )

    return response


app.include_router(routes.router)
app.include_router(health.router)


@traceable()
@app.get("/")
def root():
    return {"message": "Multi-Document-Assist API"}
