import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import chromadb
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES = ROOT / "evals" / "cases" / "task_retrieval.json"
DEFAULT_CHROMA_DIR = ROOT / "evals" / "data" / "chroma_retrieval"
DEFAULT_COLLECTION = "retrieval_benchmark_tasks"
DEFAULT_OUTPUT = ROOT / "evals" /"reports" /"task_retrieval.json"

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def load_benchmark(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    tasks = data.get("tasks")
    queries = data.get("evaluation_queries")

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Benchmark must contain a non-empty 'tasks' list.")

    if not isinstance(queries, list) or not queries:
        raise ValueError(
            "Benchmark must contain a non-empty 'evaluation_queries' list."
        )

    seen_ids: set[str] = set()

    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        text = str(task.get("text", "")).strip()

        if not task_id or not text:
            raise ValueError("Every task needs a non-empty id and text.")

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID: {task_id}")

        seen_ids.add(task_id)

    for query in queries:
        if not str(query.get("candidate_task", "")).strip():
            raise ValueError("Each query needs a candidate_task.")

        if not isinstance(query.get("expected_retrieved_tasks"), list):
            raise ValueError(
                "Each query needs an expected_retrieved_tasks list."
            )

    return tasks, queries


def create_embeddings(
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch,
        )

        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def prepare_collection(
    chroma_dir: Path,
    collection_name: str,
    tasks: list[dict[str, Any]],
    openai_client: OpenAI,
    embedding_model: str,
    rebuild: bool = False,
):
    """Create and populate a dedicated evaluation collection."""

    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Important: this client uses the explicit evaluation directory.
    # It does not use backend.config or the production retrieval client.
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception as exc:
            message = str(exc).lower()
            if "not found" not in message and "does not exist" not in message:
                raise

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(task["id"]) for task in tasks]
    documents = [str(task["text"]).strip() for task in tasks]

    metadatas = [
        {
            "task_id": str(task["id"]),
            "topic": str(task.get("topic", "unknown")),
            "status": str(task.get("status", "unknown")),
            "record_type": "task",
        }
        for task in tasks
    ]

    embeddings = create_embeddings(
        openai_client,
        embedding_model,
        documents,
    )

    # Upsert allows repeated runs without creating duplicate IDs.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection



def retrieve(
    collection,
    openai_client: OpenAI,
    embedding_model: str,
    query_text: str,
    k: int,
    threshold: float = 0,
) -> list[dict[str, Any]]:
    if not query_text.strip() or k < 1:
        return []

    count = collection.count()
    if count == 0:
        return []

    query_embedding = create_embeddings(
        openai_client,
        embedding_model,
        [query_text.strip()],
    )[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for task_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        similarity = 1.0 - float(distance)

        if similarity < threshold:
            continue

        matches.append(
            {
                "id": str(task_id),
                "text": document,
                "metadata": metadata or {},
                "similarity": similarity,
            }
        )

    return matches




def score_query(
    query_case: dict[str, Any],
    retrieved: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    expected = {
        str(task_id)
        for task_id in query_case["expected_retrieved_tasks"]
    }
    retrieved_ids = [str(task["id"]) for task in retrieved]

    base = {
        "id": query_case.get(
            "id",
            query_case["candidate_task"][:80],
        ),
        "candidate_task": query_case["candidate_task"],
        "expected_retrieved_tasks": sorted(expected),
        "retrieved_ids": retrieved_ids,
        "retrieved_tasks": retrieved,
        "labeled": bool(expected),
    }

    # Empty expected lists are not automatically labeled negatives.
    if not expected:
        return {
            **base,
            f"hit_at_{k}": None,
            f"recall_at_{k}": None,
            "reciprocal_rank": None,
            "precision_at_k": None,
            "false_positive_count": None,
            "false_positive_ids": [],
        }

    relevant_ranks = [
        rank
        for rank, task_id in enumerate(retrieved_ids, start=1)
        if task_id in expected
    ]

    relevant_count = len(relevant_ranks)
    false_positive_ids = [
        task_id for task_id in retrieved_ids
        if task_id not in expected
    ]

    return {
        **base,
        f"hit_at_{k}": int(relevant_count > 0),
        f"recall_at_{k}": relevant_count / len(expected),
        "reciprocal_rank": (
            1.0 / relevant_ranks[0]
            if relevant_ranks else 0.0
        ),
        "precision_at_k": (
            relevant_count / len(retrieved_ids)
            if retrieved_ids else 0.0
        ),
        "false_positive_count": len(false_positive_ids),
        "false_positive_ids": false_positive_ids,
    }



def evaluate(
    queries: list[dict[str, Any]],
    collection,
    openai_client: OpenAI,
    embedding_model: str,
    k: int,
    threshold: float = 0,
) -> dict[str, Any]:
    results = []

    for index, query_case in enumerate(queries, start=1):
        print(
            f"[{index}/{len(queries)}] "
            f"{query_case['candidate_task']}"
        )

        retrieved = retrieve(
            collection=collection,
            openai_client=openai_client,
            embedding_model=embedding_model,
            query_text=query_case["candidate_task"],
            k=k,
            threshold=threshold,
        )

        results.append(score_query(query_case, retrieved, k))

    labeled = [r for r in results if r["labeled"]]

    hit_key = f"hit_at_{k}"
    recall_key = f"recall_at_{k}"

    total_retrieved = sum(
        len(r["retrieved_ids"]) for r in labeled
    )
    total_false_positives = sum(
        r["false_positive_count"] for r in labeled
    )

    summary = {
        "k": k,
        "total_queries": len(results),
        "labeled_queries": len(labeled),
        "unlabeled_queries": len(results) - len(labeled),
        "hit_at_k": (
            mean(r[hit_key] for r in labeled)
            if labeled else None
        ),
        "recall_at_k": (
            mean(r[recall_key] for r in labeled)
            if labeled else None
        ),
        "precision_at_k": (
            mean(r["precision_at_k"] for r in labeled)
            if labeled else None
        ),
        "mrr": (
            mean(r["reciprocal_rank"] for r in labeled)
            if labeled else None
        ),
        "false_positive_count": total_false_positives,
        "false_positive_rate": (
            total_false_positives / total_retrieved
            if total_retrieved else None
        ),
    }

    return {
        "summary": summary,
        "results": results,
    }



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval against an isolated task benchmark."
    )

    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum cosine similarity; candidates below it are dropped.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate the evaluation collection before indexing.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    tasks, queries = load_benchmark(args.cases)

    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print(f"Benchmark: {args.cases}")
    print(f"Tasks: {len(tasks)}")
    print(f"Queries: {len(queries)}")
    print(f"Evaluation DB: {args.chroma_dir}")
    print(f"Collection: {args.collection}")

    collection = prepare_collection(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        tasks=tasks,
        openai_client=openai_client,
        embedding_model=args.embedding_model,
        rebuild=args.rebuild,
    )

    report = evaluate(
        queries=queries,
        collection=collection,
        openai_client=openai_client,
        embedding_model=args.embedding_model,
        k=args.top_k,
        threshold=args.threshold,
    )

    report["metadata"] = {
        "benchmark_file": str(args.cases),
        "chroma_directory": str(args.chroma_dir),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "corpus_task_count": len(tasks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report["summary"]

    print("\nRetrieval evaluation summary")
    print(
        f"  Labeled queries: "
        f"{summary['labeled_queries']}/{summary['total_queries']}"
    )

    if summary["hit_at_k"] is not None:
        print(f"  Hit@{args.top_k}: {summary['hit_at_k']:.3f}")
        print(f"  Recall@{args.top_k}: {summary['recall_at_k']:.3f}")
        print(f"  MRR: {summary['mrr']:.3f}")
    else:
        print("  No labeled queries available to score.")

    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
