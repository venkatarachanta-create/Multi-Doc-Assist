import asyncio
import json

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from langchain_community.vectorstores import FAISS

from src.backend.logger import GLOBAL_LOGGER as log
from src.backend.services.retrieval_service import retrieve_answer
from src.backend.rag.embeddings import get_embeddings
from src.backend.core.config import settings

from src.backend.evaluation.ragas_eval import (
    evaluate_context_precision,
    evaluate_response_relevancy,
    evaluate_context_recall,
    evaluate_factual_correctness,
    evaluate_faithfulness,
    evaluate_llm_judge
)


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

EVALUATION_DIR = BASE_DIR / "data" / "evaluation"

DATASET_PATH = (
    EVALUATION_DIR / "evaluation_dataset.json"
)

OPENAI_MODEL = "OpenAI (text-embedding-3-small)"

COHERE_MODEL = "Cohere (embed-english-v3.0)"

# Fixed evaluation embedding model.
# Both experiments use this same model for RAGAS
# Response Relevancy evaluation.

EVALUATION_EMBEDDING_MODEL = OPENAI_MODEL

BENCHMARK_MODELS = [
    OPENAI_MODEL,
    COHERE_MODEL
]

RESULT_PATHS = {
    OPENAI_MODEL: EVALUATION_DIR / "openai_results.json",
    COHERE_MODEL: EVALUATION_DIR / "cohere_results.json"
}

COMPARISON_PATH = (
    EVALUATION_DIR / "comparative_benchmark.json"
)

RETRIEVER_TYPE = "FAISS Retriever"


# ---------------------------------------------------------
# 2. Resolve FAISS index path
# ---------------------------------------------------------

def get_index_path(embedding_model):

    index_path = Path(
        f"{settings.INDEX_PATH}_{embedding_model}"
    )

    if not index_path.is_absolute():

        index_path = BASE_DIR / index_path

    index_file = index_path / "index.faiss"

    metadata_file = index_path / "index.pkl"

    if not index_file.is_file() or not metadata_file.is_file():

        raise FileNotFoundError(
            f"FAISS index not found or incomplete: {index_path}"
        )

    return index_path


# ---------------------------------------------------------
# 3. Build source ID mapping
# ---------------------------------------------------------

def build_source_mapping(
    embedding_model,
    expected_sources
):
    """
    Load the selected FAISS index and map original
    document filenames to their corresponding source IDs.

    Each expected source must map to exactly one source ID.
    """

    log.info(
        "Building source mapping",
        embedding_model=embedding_model
    )

    retrieval_embeddings = get_embeddings(
        embedding_model
    )

    index_path = get_index_path(
        embedding_model
    )

    vector_store = FAISS.load_local(
        str(index_path),
        retrieval_embeddings,
        allow_dangerous_deserialization=True
    )

    source_mapping = defaultdict(set)

    documents = vector_store.docstore._dict.values()

    for document in documents:

        metadata = document.metadata

        source_id = metadata.get("source_id")

        source_path = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source_document")
        )

        if not source_id or not source_path:
            continue

        filename = Path(
            str(source_path)
        ).name.casefold()

        source_mapping[filename].add(source_id)

    final_mapping = {}

    for source in expected_sources:

        filename = Path(source).name.casefold()

        matching_ids = source_mapping.get(
            filename,
            set()
        )

        if len(matching_ids) != 1:

            raise ValueError(
                f"Expected exactly one source_id for "
                f"{source}, but found {len(matching_ids)}. "
                f"Check metadata in {index_path}."
            )

        final_mapping[source] = next(
            iter(matching_ids)
        )

    log.info(
        "Source mapping completed",
        embedding_model=embedding_model,
        mapped_sources=len(final_mapping)
    )

    return final_mapping


# ---------------------------------------------------------
# 4. Execute individual metric safely
# ---------------------------------------------------------

async def run_metric(
    metric_name,
    metric_function,
    *args
):

    try:

        result = await metric_function(*args)

        return {
            "status": "success",
            "score": result
        }

    except Exception as e:

        log.error(
            "Evaluation metric failed",
            metric=metric_name,
            error=str(e)
        )

        return {
            "status": "failed",
            "error": str(e)
        }


# ---------------------------------------------------------
# 5. Evaluate one question
# ---------------------------------------------------------

async def evaluate_single_question(
    test_case,
    embedding_model,
    source_id,
    evaluation_embeddings
):

    question_id = test_case["id"]

    question = test_case["question"]

    reference_answer = test_case.get(
        "reference_answer"
    )

    log.info(
        "Starting benchmark question",
        question_id=question_id,
        embedding_model=embedding_model
    )

    # -----------------------------------------------------
    # Step 1: Production RAG retrieval
    # -----------------------------------------------------

    response, retrieved_context, _ = await asyncio.to_thread(
        retrieve_answer,
        question,
        embedding_model,
        RETRIEVER_TYPE,
        source_id
    )

    # -----------------------------------------------------
    # Step 2: Initialize metrics
    # -----------------------------------------------------

    metrics = {}

    # -----------------------------------------------------
    # Step 3: Context Precision
    # -----------------------------------------------------

    metrics["context_precision"] = await run_metric(
        "context_precision",
        evaluate_context_precision,
        question,
        response,
        retrieved_context
    )

    # -----------------------------------------------------
    # Step 4: Response Relevancy
    #
    # IMPORTANT:
    # Both experiments use the same evaluation embeddings.
    # Retrieval embeddings remain model-specific.
    # -----------------------------------------------------

    metrics["response_relevancy"] = await run_metric(
        "response_relevancy",
        evaluate_response_relevancy,
        question,
        response,
        evaluation_embeddings,
        retrieved_context
    )

    # -----------------------------------------------------
    # Step 5: Faithfulness
    # -----------------------------------------------------

    metrics["faithfulness"] = await run_metric(
        "faithfulness",
        evaluate_faithfulness,
        question,
        response,
        retrieved_context
    )

    # -----------------------------------------------------
    # Step 6: Derive Hallucination
    # -----------------------------------------------------

    faithfulness_result = metrics["faithfulness"]

    if faithfulness_result["status"] == "success":

        faithfulness_score = float(
            faithfulness_result["score"]
        )

        hallucination_score = (
            1 - faithfulness_score
        )

        metrics["hallucination"] = {
            "status": "success",
            "score": hallucination_score,
            "hallucination_detected": (
                hallucination_score > 0.1
            )
        }

    else:

        metrics["hallucination"] = {
            "status": "skipped",
            "reason": "Faithfulness evaluation failed"
        }

    # -----------------------------------------------------
    # Step 7: Reference-based evaluation
    # -----------------------------------------------------

    if reference_answer:

        metrics["context_recall"] = await run_metric(
            "context_recall",
            evaluate_context_recall,
            question,
            response,
            retrieved_context,
            reference_answer
        )

        metrics["factual_correctness"] = await run_metric(
            "factual_correctness",
            evaluate_factual_correctness,
            question,
            response,
            reference_answer
        )

    else:

        metrics["context_recall"] = {
            "status": "skipped",
            "reason": "Reference answer not provided"
        }

        metrics["factual_correctness"] = {
            "status": "skipped",
            "reason": "Reference answer not provided"
        }

    # -----------------------------------------------------
    # Step 8: Custom LLM Judge
    # -----------------------------------------------------

    metrics["llm_judge"] = await run_metric(
        "llm_judge",
        evaluate_llm_judge,
        question,
        response,
        retrieved_context
    )

    # -----------------------------------------------------
    # Step 9: Consolidate results
    # -----------------------------------------------------

    result = {

        "id": question_id,

        "source_document": test_case.get(
            "source_document"
        ),

        "source_id": source_id,

        "embedding_model": embedding_model,

        "evaluation_embedding_model": (
            EVALUATION_EMBEDDING_MODEL
        ),

        "retriever_type": RETRIEVER_TYPE,

        "question": question,

        "reference_answer": reference_answer,

        "generated_answer": response,

        "retrieved_context": retrieved_context,

        "metrics": metrics
    }

    log.info(
        "Benchmark question completed",
        question_id=question_id,
        embedding_model=embedding_model
    )

    return result


# ---------------------------------------------------------
# 10. Aggregate metric scores
# ---------------------------------------------------------

def aggregate_metrics(results):

    metric_names = [

        "context_precision",

        "response_relevancy",

        "context_recall",

        "factual_correctness",

        "faithfulness",

        "hallucination"
    ]

    summary = {}

    for metric_name in metric_names:

        scores = []

        for result in results:

            metric = result.get(
                "metrics",
                {}
            ).get(
                metric_name,
                {}
            )

            if metric.get("status") != "success":
                continue

            score = metric.get("score")

            if isinstance(score, (int, float)):

                scores.append(
                    float(score)
                )

        summary[metric_name] = {

            "mean": (
                sum(scores) / len(scores)
                if scores else None
            ),

            "successful_evaluations": len(scores)
        }

    return summary


# ---------------------------------------------------------
# 11. Run one embedding-model experiment
# ---------------------------------------------------------

async def run_model_benchmark(
    dataset,
    embedding_model,
    source_mapping,
    evaluation_embeddings
):

    log.info(
        "Starting embedding benchmark",
        embedding_model=embedding_model,
        total_questions=len(dataset)
    )

    results = []

    for test_case in dataset:

        source_document = test_case["source_document"]

        source_id = source_mapping[source_document]

        try:

            result = await evaluate_single_question(
                test_case,
                embedding_model,
                source_id,
                evaluation_embeddings
            )

            results.append(result)

        except Exception as e:

            log.error(
                "Benchmark question failed",
                question_id=test_case.get("id"),
                embedding_model=embedding_model,
                error=str(e)
            )

            results.append({

                "id": test_case.get("id"),

                "source_document": source_document,

                "embedding_model": embedding_model,

                "status": "failed",

                "error": str(e)
            })

    # -----------------------------------------------------
    # Build experiment report
    # -----------------------------------------------------

    report = {

        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "embedding_model": embedding_model,

        "evaluation_embedding_model": (
            EVALUATION_EMBEDDING_MODEL
        ),

        "retriever_type": RETRIEVER_TYPE,

        "total_questions": len(dataset),

        "completed_questions": sum(
            1 for result in results
            if "metrics" in result
        ),

        "failed_questions": sum(
            1 for result in results
            if result.get("status") == "failed"
        ),

        "aggregate_metrics": aggregate_metrics(
            results
        ),

        "results": results
    }

    # -----------------------------------------------------
    # Save individual model report
    # -----------------------------------------------------

    output_path = RESULT_PATHS[embedding_model]

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    log.info(
        "Embedding benchmark completed",
        embedding_model=embedding_model,
        output_path=str(output_path)
    )

    return report


# ---------------------------------------------------------
# 12. Build comparative report
# ---------------------------------------------------------

def build_comparative_report(
    openai_report,
    cohere_report
):

    metric_names = [

        "context_precision",

        "response_relevancy",

        "context_recall",

        "factual_correctness",

        "faithfulness",

        "hallucination"
    ]

    comparison = {}

    for metric_name in metric_names:

        openai_score = (
            openai_report["aggregate_metrics"]
            [metric_name]["mean"]
        )

        cohere_score = (
            cohere_report["aggregate_metrics"]
            [metric_name]["mean"]
        )

        if (
            openai_score is not None
            and cohere_score is not None
        ):

            delta = (
                cohere_score - openai_score
            )

        else:

            delta = None

        comparison[metric_name] = {

            "openai_mean": openai_score,

            "cohere_mean": cohere_score,

            "cohere_minus_openai": delta
        }

    return {

        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "total_questions_per_model": (
            len(openai_report["results"])
        ),

        "retriever_type": RETRIEVER_TYPE,

        "evaluation_embedding_model": (
            EVALUATION_EMBEDDING_MODEL
        ),

        "models": {

            "openai": OPENAI_MODEL,

            "cohere": COHERE_MODEL
        },

        "comparison": comparison
    }


# ---------------------------------------------------------
# 13. Main benchmark runner
# ---------------------------------------------------------

async def run_evaluation():

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:

        raise ValueError(
            "Evaluation dataset must be a non-empty JSON list"
        )

    question_ids = [
        item["id"]
        for item in dataset
    ]

    if len(question_ids) != len(set(question_ids)):

        raise ValueError(
            "Duplicate question IDs detected"
        )

    expected_sources = sorted({

        item["source_document"]
        for item in dataset

    })

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    log.info(
        "Starting comparative benchmark",
        total_questions=len(dataset),
        total_sources=len(expected_sources)
    )

    # -----------------------------------------------------
    # IMPORTANT: Initialize evaluation embeddings ONCE
    #
    # Both OpenAI and Cohere experiments use the same
    # embeddings for Response Relevancy evaluation.
    # -----------------------------------------------------

    log.info(
        "Initializing fixed evaluation embeddings",
        evaluation_embedding_model=(
            EVALUATION_EMBEDDING_MODEL
        )
    )

    evaluation_embeddings = await asyncio.to_thread(
        get_embeddings,
        EVALUATION_EMBEDDING_MODEL
    )

    # -----------------------------------------------------
    # Run both embedding experiments
    # -----------------------------------------------------

    model_reports = {}

    for embedding_model in BENCHMARK_MODELS:

        # Build model-specific source mapping

        source_mapping = await asyncio.to_thread(
            build_source_mapping,
            embedding_model,
            expected_sources
        )

        # Execute experiment

        report = await run_model_benchmark(
            dataset,
            embedding_model,
            source_mapping,
            evaluation_embeddings
        )

        model_reports[embedding_model] = report

        print("\n--------------------------------------")

        print(
            f"Completed: {embedding_model}"
        )

        print("--------------------------------------")

        print(
            f"Total questions: {report['total_questions']}"
        )

        print(
            f"Completed questions: "
            f"{report['completed_questions']}"
        )

        print(
            f"Failed questions: "
            f"{report['failed_questions']}"
        )

    # -----------------------------------------------------
    # Generate comparative report
    # -----------------------------------------------------

    comparative_report = build_comparative_report(
        model_reports[OPENAI_MODEL],
        model_reports[COHERE_MODEL]
    )

    with open(
        COMPARISON_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            comparative_report,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # -----------------------------------------------------
    # Print final summary
    # -----------------------------------------------------

    print("\n======================================")

    print(
        "15-QUESTION COMPARATIVE BENCHMARK DONE"
    )

    print("======================================")

    print(
        f"\nEvaluation embeddings: "
        f"{EVALUATION_EMBEDDING_MODEL}"
    )

    print(
        f"\nOpenAI report: {RESULT_PATHS[OPENAI_MODEL]}"
    )

    print(
        f"Cohere report: {RESULT_PATHS[COHERE_MODEL]}"
    )

    print(
        f"Comparison: {COMPARISON_PATH}"
    )

    print("\nMetric comparison:")

    for metric_name, scores in (
        comparative_report["comparison"].items()
    ):

        openai_score = scores["openai_mean"]

        cohere_score = scores["cohere_mean"]

        if openai_score is None or cohere_score is None:

            print(
                f"{metric_name:25} Insufficient evaluation data"
            )

            continue

        print(

            f"{metric_name:25} "

            f"OpenAI={openai_score:.4f} | "

            f"Cohere={cohere_score:.4f}"
        )

    log.info(
        "Comparative benchmark completed",
        comparison_path=str(COMPARISON_PATH)
    )


# ---------------------------------------------------------
# 14. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    asyncio.run(
        run_evaluation()
    )