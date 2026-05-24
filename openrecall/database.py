"""
database.py — ChromaDB-backed persistent store for screen captures.

Replaces the old SQLite + manual numpy cosine-similarity approach with
ChromaDB, which handles embeddings and semantic search natively.
No Docker required — ChromaDB runs as a local embedded database.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from openrecall.config import appdata_folder

logger = logging.getLogger(__name__)

# ChromaDB data lives alongside screenshots
CHROMA_PATH = os.path.join(appdata_folder, "chroma_db")

# Embedding model — same lightweight model as before, now managed by ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    """Lazy-initialise and return the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = _client.get_or_create_collection(
            name="screen_captures",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready at %s (%d entries)", CHROMA_PATH, _collection.count())
    return _collection


def create_db() -> None:
    """Initialise the ChromaDB collection (called at app startup)."""
    try:
        _get_collection()
    except Exception as exc:
        logger.error("Failed to initialise ChromaDB: %s", exc)


# Alias so old code that calls init_db() also works
init_db = create_db


def insert_entry(
    text: str,
    timestamp: int,
    app: str,
    title: str,
    filename: str,
) -> None:
    """
    Store a screen-capture entry.

    ChromaDB automatically computes and stores the embedding from `text`,
    so there is no need to pass a pre-computed embedding vector.
    """
    if not text.strip():
        return
    try:
        collection = _get_collection()
        collection.add(
            documents=[text],
            metadatas=[
                {
                    "app": app,
                    "title": title,
                    "timestamp": timestamp,
                    "filename": filename,
                }
            ],
            ids=[f"{timestamp}_{filename}"],
        )
        logger.debug("Inserted entry timestamp=%d filename=%s", timestamp, filename)
    except Exception as exc:
        logger.error("Error inserting entry: %s", exc)


def search_entries(query: str, n_results: int = 20) -> List[Dict[str, Any]]:
    """
    Semantic search over all captured screens.

    Returns a list of result dicts sorted by relevance (closest first),
    each containing: timestamp, filename, app, title, text, distance.
    """
    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        entries = []
        for i, metadata in enumerate(results["metadatas"][0]):
            entries.append(
                {
                    "timestamp": metadata["timestamp"],
                    "filename": metadata["filename"],
                    "app": metadata.get("app", ""),
                    "title": metadata.get("title", ""),
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return entries

    except Exception as exc:
        logger.error("Search error: %s", exc)
        return []


def get_all_entries() -> List[Dict[str, Any]]:
    """Return all entries sorted by timestamp descending (for the timeline)."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []

        results = collection.get(include=["documents", "metadatas"])
        entries = []
        for i, metadata in enumerate(results["metadatas"]):
            entries.append(
                {
                    "timestamp": metadata["timestamp"],
                    "filename": metadata["filename"],
                    "app": metadata.get("app", ""),
                    "title": metadata.get("title", ""),
                    "text": results["documents"][i],
                }
            )
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return entries

    except Exception as exc:
        logger.error("Error getting all entries: %s", exc)
        return []


def get_timestamps() -> List[int]:
    """Return all timestamps sorted descending (used by the timeline slider)."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
        timestamps = [m["timestamp"] for m in results["metadatas"]]
        return sorted(timestamps, reverse=True)
    except Exception as exc:
        logger.error("Error getting timestamps: %s", exc)
        return []
