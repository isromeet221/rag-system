"""
Central configuration for Part B (env-driven, offline-friendly).
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo layout: .../RAG/partb/this_file.py → repo root is parent of partb/
PARTB_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTB_DIR.parent
PARTA_DIR =REPO_ROOT / "parta"
PORTABLE_DIR = PARTA_DIR / "portable"
RERANKER_DIR = PORTABLE_DIR / "reranker"

# Part A data paths (page viewer)
PARTA_DATA_DIR = PARTA_DIR / "data"
CHECKPOINTS_DIR = PARTA_DATA_DIR / "checkpoints"
METADATA_DIR = PARTA_DATA_DIR / "metadata"
QDRANT_DIR = PARTA_DATA_DIR / "qdrant"


# Mongo (same as Part A defaults)
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB =  "rag_system"


# JWT — MUST match Part A in production
JWT_SECRET =  "ISRO_RAG_SECRET_CHANGE_IN_PROD"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS =8

# LiteLLM OpenAI-compatible proxy (Master)
LITELLM_BASE_URL = "http://127.0.0.1:4000/v1"
LITELLM_API_KEY = ""

# If "1", stream from local Ollama /api/generate instead (dev fallback)
USE_OLLAMA_DIRECT = False
OLLAMA_URL =  "http://127.0.0.1:11434"

QDRANT_URL =  "http://localhost:6333"
COLLECTION_PROPS = "RAG_PROPOSITIons"
COLLECTION_SECTIONS = "RAG_sections"


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "sac@1234"

ENTITY_LABELS = ["Component", "Material", "Specification", "Standard", "Entity"]
NEO4J_ENTITY_LIMIT = 50
PROP_RETRIEVE_LIMIT = 40
SECT_RETRIEVE_LIMIT = 40
PAGE_EXPAND_MAX_CHARS = 9000


# Retrieval constants (aligned with parta/test.py)
GLINER_QUERY_THRESHOLD = float(os.environ.get("RAG_GLINER_QUERY_THRESHOLD", "0.35"))
BOOST_BOTH = float(os.environ.get("RAG_BOOST_BOTH", "0.12"))
LONG_CHUNK_WORDS = int(os.environ.get("RAG_LONG_CHUNK_WORDS", "450"))



# Apply Qdrant/Neo env for processing.* imports (ingest_vectors / ingest_graph)
def apply_parta_service_env() -> None:
    """Ensure parta modules see DB URLs even if only Part B is started."""
    os.environ.setdefault("RAG_QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("RAG_NEO4J_URI", os.environ.get("RAG_NEO4J_URI", "bolt://localhost:7687"))
    os.environ.setdefault("RAG_NEO4J_USER", os.environ.get("RAG_NEO4J_USER", "neo4j"))
    os.environ.setdefault("RAG_NEO4J_PASSWORD", os.environ.get("RAG_NEO4J_PASSWORD", "sac@1234"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


MODE_ORDER = ("fast", "balanced", "deep")

# LiteLLM model_name must match deploy/litellm_config.yaml entries
MODE_CONFIG: dict[str, dict] = {
    "fast": {
        "litellm_model": "gemma3:1b",
        "ollama_model": "gemma3:1b",  # direct fallback
        "qdrant_over_retrieve": int(os.environ.get("RAG_FAST_QDRANT_LIMIT", "40")),
        "final_top_n": int(os.environ.get("RAG_FAST_FINAL_TOP", "8")),
        "context_max_chars": int(os.environ.get("RAG_FAST_CTX_CHARS", "12000")),
        "history_pairs": 3,
        "llm_timeout_s": float(os.environ.get("RAG_FAST_LLM_TIMEOUT", "600")),
    },
    "balanced": {
        "litellm_model": "mistral:7b-instruct",
        "ollama_model": "mistral:7b-instruct-q4_K_M",
        "qdrant_over_retrieve": int(os.environ.get("RAG_BAL_QDRANT_LIMIT", "40")),
        "final_top_n": int(os.environ.get("RAG_BAL_FINAL_TOP", "8")),
        "context_max_chars": int(os.environ.get("RAG_BAL_CTX_CHARS", "14000")),
        "history_pairs": 5,
        "llm_timeout_s": float(os.environ.get("RAG_BAL_LLM_TIMEOUT", "600")),
    },
    "deep": {
        "litellm_model": "llama3.1:8b-instruct",
        "ollama_model": "llama3.1:8b-instruct-q4_K_M",
        "qdrant_over_retrieve": int(os.environ.get("RAG_DEEP_QDRANT_LIMIT", "48")),
        "final_top_n": int(os.environ.get("RAG_DEEP_FINAL_TOP", "10")),
        "context_max_chars": int(os.environ.get("RAG_DEEP_CTX_CHARS", "16000")),
        "history_pairs": 6,
        "llm_timeout_s": float(os.environ.get("RAG_DEEP_LLM_TIMEOUT", "600")),
    },
}