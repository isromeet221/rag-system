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
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB =  "rag_system"


# JWT — MUST match Part A in production
JWT_SECRET = os.environ.get("JWT_SECRET", "ISRO_RAG_SECRET_CHANGE_IN_PROD")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS =8

# LiteLLM OpenAI-compatible proxy (Master)
LITELLM_BASE_URL = "http://127.0.0.1:4000/v1"
# LITELLM_BASE_URL = "https://api.mistral.ai/v1"
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")

# If "1", stream from local Ollama /api/generate instead (dev fallback)
USE_OLLAMA_DIRECT = False
OLLAMA_URL =  "http://127.0.0.1:11434"

# QDRANT_URL =  "http://localhost:6333"
QDRANT_URL           = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY       = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_PROPS     = "RAG_PROPOSITIons"
COLLECTION_SECTIONS = "RAG_sections"


# NEO4J_URI = "bolt://localhost:7687"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

import json
from partb.logger import time_it, async_time_it

import collections

ENTITY_LABELS = ["Component", "Material", "Specification", "Standard", "Entity"]

# Dynamically load from unie_synthetic.json if present
_json_path = PARTB_DIR / "unie_synthetic.json"
if _json_path.exists():
    try:
        with open(_json_path, 'r', encoding='utf-8') as f:
            _data = json.load(f)
        # Extract generic entity labels (ignore relation pairs with '<>' and generic 'match')
        _all_labels = [
            item[2] for record in _data 
            for item in record.get('ner', []) 
            if '<>' not in item[2] and item[2].lower() != "match"
        ]
        _counter = collections.Counter(_all_labels)
        # Take the top 30 most frequent entity types
        ENTITY_LABELS = [k for k, v in _counter.most_common(30)]
    except Exception as e:
        print(f"Warning: Failed to load labels from JSON: {e}")
NEO4J_ENTITY_LIMIT = 50
PROP_RETRIEVE_LIMIT = 40
SECT_RETRIEVE_LIMIT = 40
PAGE_EXPAND_MAX_CHARS = 9000


# Retrieval constants (aligned with parta/test.py)
GLINER_QUERY_THRESHOLD = float(os.environ.get("RAG_GLINER_QUERY_THRESHOLD", "0.35"))
BOOST_BOTH = float(os.environ.get("RAG_BOOST_BOTH", "0.12"))
LONG_CHUNK_WORDS = int(os.environ.get("RAG_LONG_CHUNK_WORDS", "450"))

# ── Reranker tuning ──────────────────────────────────────────────────────────
# Jina Reranker v3 has 131K context and processes ALL candidate documents
# listwise in a single forward pass via model.rerank(query, documents).
# Segment-max pooling is only needed for text exceeding 131K tokens.
RERANK_SEGMENT_TOKENS = int(os.environ.get("RAG_RERANK_SEGMENT_TOKENS", "60000"))
RERANK_FULL_CHUNK_WORDS = int(os.environ.get("RAG_RERANK_FULL_CHUNK_WORDS", "80000"))

# Drop candidates whose rerank score is below this floor. Jina Reranker v3
# produces unbounded logit scores (not sigmoid-squashed), so the absolute
# values differ from the previous BGE model. Tune empirically after reading
# the score-distribution logs (rerank_candidates logs min/median/max per
# query). 0.0 = no filtering.
RERANK_MIN_SCORE = float(os.environ.get("RAG_RERANK_MIN_SCORE", "0.0"))

# Score-distribution logging sample rate — log min/median/max of rerank scores
# once per N queries so you can calibrate RERANK_MIN_SCORE / BOOST_BOTH.
RERANK_LOG_DISTRIBUTION_EVERY = int(os.environ.get("RAG_RERANK_LOG_DIST_EVERY", "1"))


# MMR (Maximum Marginal Relevance) diversity
ENABLE_MMR = os.environ.get("RAG_ENABLE_MMR", "0") == "1"
MMR_LAMBDA = float(os.environ.get("RAG_MMR_LAMBDA", "0.7"))
MMR_POOL_MULTIPLIER = int(os.environ.get("RAG_MMR_POOL_MULTIPLIER", "2"))


# Query classification & routing
ENABLE_QUERY_CLASSIFICATION = os.environ.get("RAG_ENABLE_QUERY_CLASSIFICATION", "1") == "1"

# Per-(mode, query_type) overrides applied on top of MODE_CONFIG.
# Each key: mode -> query_type -> {overrides}
#   boost_both_mult:        Multiplier for BOOST_BOTH when a candidate comes from both Qdrant + Neo4j
#   page_expand_range:      Pages before/after the main page to fetch (0 = no expansion)
#   final_top_n_adjust:     Added to the mode's final_top_n (can be negative)
#   context_max_chars_adjust: Added to the mode's context_max_chars (can be negative)
#   cross_book:             If True, ignore user's book filter and search all books
QUERY_TYPE_OVERRIDES: dict[str, dict[str, dict]] = {
    "fast": {
        "spec_lookup": {"boost_both_mult": 2.0, "page_expand_range": 1, "final_top_n_adjust": 0, "context_max_chars_adjust": 0, "cross_book": False},
        "process":     {"boost_both_mult": 1.0, "page_expand_range": 2, "final_top_n_adjust": 0, "context_max_chars_adjust": 0, "cross_book": False},
        "comparison":  {"boost_both_mult": 1.5, "page_expand_range": 1, "final_top_n_adjust": 2, "context_max_chars_adjust": 0, "cross_book": True},
        "overview":    {"boost_both_mult": 1.0, "page_expand_range": 0, "final_top_n_adjust": -3, "context_max_chars_adjust": -2000, "cross_book": False},
    },
    "balanced": {
        "spec_lookup": {"boost_both_mult": 2.0, "page_expand_range": 1, "final_top_n_adjust": 0, "context_max_chars_adjust": 0, "cross_book": False},
        "process":     {"boost_both_mult": 1.0, "page_expand_range": 2, "final_top_n_adjust": 2, "context_max_chars_adjust": 2000, "cross_book": False},
        "comparison":  {"boost_both_mult": 1.5, "page_expand_range": 1, "final_top_n_adjust": 4, "context_max_chars_adjust": 2000, "cross_book": True},
        "overview":    {"boost_both_mult": 1.0, "page_expand_range": 0, "final_top_n_adjust": -2, "context_max_chars_adjust": -4000, "cross_book": False},
    },
    "deep": {
        "spec_lookup": {"boost_both_mult": 2.0, "page_expand_range": 1, "final_top_n_adjust": 0, "context_max_chars_adjust": 0, "cross_book": False},
        "process":     {"boost_both_mult": 1.0, "page_expand_range": 2, "final_top_n_adjust": 2, "context_max_chars_adjust": 2000, "cross_book": False},
        "comparison":  {"boost_both_mult": 1.5, "page_expand_range": 1, "final_top_n_adjust": 4, "context_max_chars_adjust": 2000, "cross_book": True},
        "overview":    {"boost_both_mult": 1.0, "page_expand_range": 0, "final_top_n_adjust": -3, "context_max_chars_adjust": -2000, "cross_book": False},
    },
}

# Defaults when no type-specific override is defined (e.g. "general" type)
QUERY_TYPE_GENERAL = {"boost_both_mult": 1.0, "page_expand_range": 1, "final_top_n_adjust": 0, "context_max_chars_adjust": 0, "cross_book": False}


# Greedy context allocation — select items by value density (score/chars)
# instead of fixed priority order (specs → pages → chunks).
CONTEXT_GREEDY = os.environ.get("RAG_CONTEXT_GREEDY", "1") == "1"


# Apply Qdrant/Neo env for processing.* imports (ingest_vectors / ingest_graph)
@time_it
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
        "litellm_model": "open-mistral-nemo",
        "ollama_model": "gemma3:1b",  # direct fallback
        "qdrant_over_retrieve": int(os.environ.get("RAG_FAST_QDRANT_LIMIT", "40")),
        "final_top_n": int(os.environ.get("RAG_FAST_FINAL_TOP", "8")),
        "context_max_chars": int(os.environ.get("RAG_FAST_CTX_CHARS", "12000")),
        "history_pairs": 3,
        "llm_timeout_s": float(os.environ.get("RAG_FAST_LLM_TIMEOUT", "600")),
    },
    "balanced": {
        "litellm_model": "mistral-small-latest",
        "ollama_model": "mistral:7b-instruct-q4_K_M",
        "qdrant_over_retrieve": int(os.environ.get("RAG_BAL_QDRANT_LIMIT", "40")),
        "final_top_n": int(os.environ.get("RAG_BAL_FINAL_TOP", "8")),
        "context_max_chars": int(os.environ.get("RAG_BAL_CTX_CHARS", "14000")),
        "history_pairs": 5,
        "llm_timeout_s": float(os.environ.get("RAG_BAL_LLM_TIMEOUT", "600")),
    },
    "deep": {
        "litellm_model": "mistral-large-latest",
        "ollama_model": "llama3.1:8b-instruct-q4_K_M",
        "qdrant_over_retrieve": int(os.environ.get("RAG_DEEP_QDRANT_LIMIT", "48")),
        "final_top_n": int(os.environ.get("RAG_DEEP_FINAL_TOP", "10")),
        "context_max_chars": int(os.environ.get("RAG_DEEP_CTX_CHARS", "16000")),
        "history_pairs": 6,
        "llm_timeout_s": float(os.environ.get("RAG_DEEP_LLM_TIMEOUT", "600")),
    },
}