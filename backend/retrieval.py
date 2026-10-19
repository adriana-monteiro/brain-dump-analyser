
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import chromadb
from openai import OpenAI
from backend.config import settings


class RetrievalService:
    """Semantic retrieval backed by ChromaDB and OpenAI embeddings."""

    def __init__(
        self,
        persist_directory: str | Path = "./.chroma/tasks",
        collection_name: str = "brain_dump_tasks",
        embedding_model: str | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self.embedding_model = settings.openai_embedding_model

        # Initialize the OpenAI client.
        self.openai_client = openai_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Use configured production defaults unless explicitly overridden.
        persist_directory = (
            persist_directory or settings.chroma_persist_directory
        )
        collection_name = (
            collection_name or settings.chroma_collection_name
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _create_embeddings(
        self, texts: list[str]
    ) -> list[list[float]]:
        if not texts:
            return []

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def upsert(
        self,
        record_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.upsert_batch([(record_id, text, metadata)])

    def upsert_batch(
        self,
        records: Iterable[
            tuple[str, str, dict[str, Any] | None]
        ],
    ) -> None:
        records = list(records)
        if not records:
            return

        ids = [record[0] for record in records]
        texts = [record[1] for record in records]
        metadatas = [record[2] or {} for record in records]

        if any(not text.strip() for text in texts):
            raise ValueError("Record text cannot be empty.")

        embeddings = self._create_embeddings(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            return []

        if n_results < 1:
            return []

        query_embedding = self._create_embeddings([query_text])[0]

        # Chroma requires $and for multiple metadata conditions.
        normalized_where = where
        if where and not any(key.startswith("$") for key in where):
            if len(where) > 1:
                normalized_where = {
                    "$and": [
                        {key: value} for key, value in where.items()
                    ]
                }

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if normalized_where:
            kwargs["where"] = normalized_where

        results = self.collection.query(**kwargs)
        if not results["ids"] or not results["ids"][0]:
            return []

        matches = []
        for record_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "id": record_id,
                    "text": document,
                    "metadata": metadata or {},
                    "similarity": 1.0 - distance,
                }
            )

        return matches

    def update_metadata(
        self,
        record_id: str,
        new_metadata: dict[str, Any],
    ) -> None:
        record = self.collection.get(
            ids=[record_id],
            include=["metadatas"],
        )
        if not record["ids"]:
            raise ValueError(f"Record {record_id!r} not found.")

        metadata = dict(record["metadatas"][0] or {})
        metadata.update(new_metadata)

        self.collection.update(
            ids=[record_id],
            metadatas=[metadata],
        )

    def update_content(
        self,
        record_id: str,
        new_text: str,
        new_metadata: dict[str, Any] | None = None,
    ) -> None:
        record = self.collection.get(
            ids=[record_id],
            include=["metadatas"],
        )
        if not record["ids"]:
            raise ValueError(f"Record {record_id!r} not found.")

        metadata = dict(record["metadatas"][0] or {})
        metadata.update(new_metadata or {})

        self.upsert(record_id, new_text, metadata)

    def delete(self, record_id: str) -> None:
        self.collection.delete(ids=[record_id])

    # Convenience methods for your brain-dump task index.

    def index_task(
        self,
        task_id: int | str,
        text: str,
        topic: str,
        status: str = "open",
    ) -> None:
        self.upsert(
            record_id=str(task_id),
            text=text,
            metadata={
                "task_id": str(task_id),
                "topic": topic,
                "status": status,
                "record_type": "task",
            },
        )

    def search_tasks(
        self,
        text: str,
        n_results: int = 5,
        topic: str | None = None,
        status: str = "open",
    ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {
            "record_type": "task",
            "status": status,
        }
        if topic:
            where["topic"] = topic

        return self.query(
            query_text=text,
            n_results=n_results,
            where=where,
        )
