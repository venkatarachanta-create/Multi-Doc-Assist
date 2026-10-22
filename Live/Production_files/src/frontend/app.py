# Purpose:
# The only file in the whole project a non-technical user ever sees. It
# renders the UI, collects input, and calls utils.py — nothing more.
#
# What to focus on:
# - st.session_state.ingested gates the whole "Ask Questions" section. Look
#   at how it flips from False to True and where.
# - This file has ZERO RAG logic. If you're tempted to add retrieval code
#   here, that's a sign it belongs in rag/ or services/ instead.
#
# TODO:
# 1. Real bug, actually hit while building this project: utils.py's
#    retrieve_answer() returns `None` (not a 3-tuple) when the backend call
#    fails. Line 107 below does
#        answer, retrieved_context, evaluation_metric = retrieve_answer(...)
#    What happens when retrieve_answer() returns bare None instead of a
#    3-tuple? Try it: stop the backend, ask a question, read the traceback.
#    How would you guard against this?
# 2. embedding_model and retriever_type are read fresh from the sidebar on
#    every rerun (Streamlit reruns the whole script on each interaction).
#    What would happen if a user changed the embedding model AFTER
#    ingesting a document, then asked a question? Does the app warn them?
# 3. Trace one full round trip: click "Ingest Document" -> which function
#    here gets called -> which function in utils.py -> which FastAPI route
#    -> which service -> which rag/ files, in order.

import streamlit as st
import requests
from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from src.frontend.utils import upload_document, retrieve_answer
from typing import Optional


st.set_page_config(
    page_title="Multi-Document Assist"
)

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📄 Multi-Document Assist")
st.caption("Upload documents and ask intelligent questions using RAG")

# Initialize session state
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    embedding_model = st.selectbox(
        "Choose Embedding Model",
        ("OpenAI (text-embedding-3-large)", "OpenAI (text-embedding-3-small)", "Cohere (embed-english-v3.0)"),
        index=2,
        help="Select the embedding model to use for ingestion and retrieval."
    )
    st.info(f"Selected Model: **{embedding_model}**")

    retriever_type = st.selectbox(
        "Choose Retriever Type",
        ("FAISS Retriever", "Contextual Compression Retriever", "MMR Retriever", "Multi Query Retriever"),
        index=0,
        help="Select the retriever type to use for retrieval."
    )
    st.info(f"Selected Retriever Type: **{retriever_type}**")

# ---------------- Upload Section ----------------
with st.container():
    st.subheader("📤 Upload & Ingest Document")

    uploaded_files = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT",
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        ingest_clicked = st.button(
            "🚀 Ingest Document",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if ingest_clicked and uploaded_files:
        with st.spinner(f"Ingesting document using {embedding_model}..."):
            total_chunks = 0
            ingested_files = 0
            skipped_files = []

            for uploaded_file in uploaded_files:
                chunks = upload_document(uploaded_file, embedding_model)
                if chunks is None:
                    skipped_files.append(uploaded_file.name)
                    continue

                total_chunks += chunks
                ingested_files += 1

        if ingested_files:
            st.success(f"✅ Ingested {ingested_files} file(s) into {total_chunks} chunk(s)")
            st.session_state.ingested = True

        if skipped_files:
            st.warning(f"Skipped files with no content or upload issues: {', '.join(skipped_files)}")

# Divider
st.divider()

# ---------------- Query Section ----------------
with st.container():
    st.subheader("💬 Ask Questions")

    if not st.session_state.ingested:
        st.info("📌 Please upload and ingest a document first.")
    else:
        question = st.text_input(
            "Enter your question",
            placeholder="What is this document about?",
        )

        ask_clicked = st.button(
            "🔍 Get Answer",
            use_container_width=True,
        )

        if ask_clicked:
            with st.spinner(f"Searching for the best answer using {embedding_model}..."):
                answer, retrieved_context, evaluation_metric = retrieve_answer(question, embedding_model, retriever_type)

            if answer:
                st.markdown("### 📌 Answer")
                st.write(answer)
                log.info("Answer displayed successfully")

                with st.spinner("Hold on! Generating Evaluation metrics"):
                    try:
                        st.markdown("---")
                        st.subheader("📊 RAGA's Evaluation Score")
                        with st.expander("📌 Core RAGAS Metrics", expanded=False):
                            st.info(f"Context Precision: {evaluation_metric['context_precision']}")
                            st.info(f"Response Relevancy: {evaluation_metric['response_relevancy']}")
                            st.info(f"Contextual Recall: {evaluation_metric['contextual_recall']}")
                            st.info(f"Factual Correctness: {evaluation_metric['factual_correctness']}")
                            st.info(f"Faithfulness: {evaluation_metric['faithfulness']}")

                            hallucination = evaluation_metric['hallucination']
                            st.info(
                                f"""
                                **Hallucination Summary**
                                - Is the Model Hallucinated: {hallucination['hallucination_detected']}
                                - Hallucination Score: {hallucination['hallucination_score']}
                                """
                            )

                            llm_judge = evaluation_metric["llm_judge"]
                            st.info(
                                f"""
                                **LLM Judge Summary**
                                - Relevancy: {llm_judge['relevancy_percentage']}%
                                - Hallucination Detected: {llm_judge['hallucination_detected']}
                                - Justification: {llm_judge['explanation']}
                                """
                            )

                    except Exception as e:
                        st.warning(f"RAGAS Evaluation failed: {e}")
