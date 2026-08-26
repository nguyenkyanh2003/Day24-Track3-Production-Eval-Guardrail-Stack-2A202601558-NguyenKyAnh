"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- LLM provider ---
# Prefer Gemini when configured because its OpenAI-compatible endpoint lets the
# whole lab run on the free tier without changing the evaluation interfaces.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_API_KEY = GEMINI_API_KEY or OPENAI_API_KEY
USING_GEMINI = bool(GEMINI_API_KEY)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or (GEMINI_BASE_URL if USING_GEMINI else "")
_configured_model = os.getenv("LLM_MODEL", "")
if USING_GEMINI and _configured_model == "gemini-2.5-flash-lite":
    _configured_model = "gemini-3.5-flash-lite"
LLM_MODEL = _configured_model or ("gemini-3.5-flash-lite" if USING_GEMINI else "gpt-4o-mini")
LLM_MAX_RPM = int(os.getenv("LLM_MAX_RPM", "14" if USING_GEMINI else "0"))

# NeMo's OpenAI engine reads these conventional environment variables.
if USING_GEMINI:
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

# RAGAS uses a small multilingual local embedding model for answer relevancy.
# This avoids an additional paid embeddings endpoint and works well for Vietnamese.
RAGAS_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "4"))
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "6"))
RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "120"))

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3
RERANK_CANDIDATE_K = 8  # CPU-friendly shortlist from the hybrid top-20

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or LLM_MODEL

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
