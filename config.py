"""
config.py
─────────
Central configuration hub for the AWS Exam Generator pipeline.
Reads environment overrides where applicable but ships with
production-ready defaults tuned for a GTX 1070 (8 GB VRAM).
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Path constants
# ─────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
DATA_DIR       = BASE_DIR / "data"
CHROMA_DIR     = BASE_DIR / "chroma_store"
OUTPUT_DIR     = BASE_DIR / "output"
REGISTRY_PATH  = BASE_DIR / "registry.json"

# Create required directories on import
for _d in (DATA_DIR, CHROMA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Ollama / LLM settings  (GTX 1070 8 GB tuning)
# ─────────────────────────────────────────────
@dataclass
class OllamaConfig:
    # Model name as registered in Ollama
    model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODEL", "qwen2.5:7b-instruct"
        )
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    # 4 096 tokens keeps the full KV-cache inside 8 GB VRAM at Q4_K_M.
    # Raise via OLLAMA_NUM_CTX on systems with more VRAM headroom.
    num_ctx: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    )
    # Temperature controls creativity vs. consistency
    temperature: float = 0.4
    # top_p nucleus sampling
    top_p: float = 0.9
    # Keeps model resident in VRAM between calls (-1 = indefinitely)
    keep_alive: int = -1
    # Hard cap on generated tokens per request
    max_tokens: int = 2048
    # Request timeout in seconds
    timeout: int = 180


# ─────────────────────────────────────────────
# Embedding / RAG settings
# ─────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    # Lightweight model: ~23 MB, runs on CPU, leaves VRAM free for LLM
    model_name: str = "all-MiniLM-L6-v2"
    # Token chunk size for PDF splitting
    chunk_size: int = 500
    # Overlap keeps context across chunk boundaries
    chunk_overlap: int = 50
    # Number of RAG chunks returned per query.
    # 4 chunks × 500 tokens = 2 000 input tokens, leaving ~2 000 tokens for
    # the generated response within the 4 096-token context window.
    # Raise via RAG_TOP_K alongside OLLAMA_NUM_CTX on higher-VRAM systems.
    top_k_results: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "4"))
    )


# ─────────────────────────────────────────────
# Question-generation policy
# ─────────────────────────────────────────────
@dataclass
class GenerationConfig:
    # 75 % single-answer, 25 % multiple-selection
    single_answer_ratio: float = 0.75
    # Maximum AI reviewer retry iterations per question
    max_revision_attempts: int = 2
    # Maximum generation attempts for a single question slot
    max_generation_attempts: int = 2
    # Batch size – keep at 1 to prevent OOM on GTX 1070
    batch_size: int = 1
    # Number of distractor options for single-answer questions
    single_options_count: int = 4
    # Number of distractor options for multiple-select questions
    multi_options_count: int = 5
    # Extra generate+review rounds attempted to backfill questions lost to
    # generation failures or reviewer rejection, so the final approved
    # count matches the originally requested count where possible.
    max_topup_rounds: int = 5
    # Stop backfilling early if this many consecutive top-up rounds
    # approve zero new questions (signals a systemic quality problem
    # that more rounds won't fix).
    max_zero_progress_rounds: int = 2


# ─────────────────────────────────────────────
# Convenience accessor (singleton-ish pattern)
# ─────────────────────────────────────────────
OLLAMA_CONFIG    = OllamaConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
GENERATION_CONFIG = GenerationConfig()
