# Purpose:
# See how thin the API layer actually is once services/ exists. This file
# should never contain RAG logic — only HTTP plumbing: read the request,
# call a service, shape the response.
#
# - Compare this file's line count to ingestion_service.py or
#   retrieval_service.py. The API layer is deliberately boring.
# - /upload takes raw multipart form fields; /retrieve takes a typed
#   Pydantic model (QueryRequest). Why the difference?
#
# TODO:
# 1. Trace a single click of "Ask Questions" in the Streamlit UI all the way
#    to this file: frontend/app.py -> frontend/utils.py -> here -> which
#    service function gets called?
# 2. QueryResponse is defined in src/shared/schemas.py, not here. Why keep
#    request/response shapes in a shared file instead of routes.py?
# 3. Neither endpoint has a try/except here. Where does error handling
#    actually happen for these two calls? (Hint: check the service files —
#    what does @retry do when every attempt fails?)

from fastapi import APIRouter, UploadFile, Form
from src.backend.logger import GLOBAL_LOGGER as log
from src.shared.schemas import QueryRequest, QueryResponse
from src.backend.services.ingestion_service import ingest_document
from src.backend.services.retrieval_service import retrieve_answer


router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile, embedding_model: str = Form(...)):

    log.info('Upload File API Trigerred', embedding_model=embedding_model)
    chunks = await ingest_document(file, embedding_model)
    log.info('Upload File completed')

    return {"chunks": chunks}


@router.post("/retrieve", response_model=QueryResponse)
async def retrieve(request: QueryRequest):

    log.info('Retrieve API Trigerred', embedding_model=request.embedding_model)
    answer, answer_texts, evaluation_metric = retrieve_answer(request.question, request.embedding_model, request.retriever_type)
    log.info('Answer Retrieved')

    return QueryResponse(answer=answer, answer_texts=answer_texts, evaluation_metric=evaluation_metric)
