# Purpose:
# The contract between frontend and backend. Both sides import these same
# classes — that's the whole point of putting them in src/shared/ instead
# of duplicating field names in routes.py and app.py separately.
#
# What to focus on:
# - QueryRequest is what the frontend sends; QueryResponse is what the
#   backend sends back. FastAPI uses these for automatic request validation
#   AND for generating the (internal-only) /docs page.
# - answer_texts is typed as `list`, not `List[str]`, and evaluation_metric
#   as `dict`, not a typed model. That's looser than it could be.
#
# TODO:
# 1. If a request arrives missing `retriever_type`, what does FastAPI do
#    before your code ever runs? Try removing a required field with the
#    live app and see what error comes back.
# 2. Tighten evaluation_metric from `dict` to a proper Pydantic model with
#    the exact keys retrieval_service.py returns (context_precision,
#    response_relevancy, contextual_recall, factual_correctness,
#    faithfulness, hallucination, llm_judge). What would you gain?

from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str
    embedding_model: str
    retriever_type: str


class QueryResponse(BaseModel):
    answer: str
    answer_texts: list
    evaluation_metric: dict
