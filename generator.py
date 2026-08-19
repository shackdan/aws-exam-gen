"""
generator.py
────────────
AI Question Generator Agent

Responsibilities:
  1. Query the ChromaDB vector store for domain-relevant context chunks
     (RAG retrieval).
  2. Build structured prompts and call the local Ollama LLM to draft
     ExamQuestion objects.
  3. Enforce the 75 % single-answer / 25 % multiple-selection split.
  4. Act as the revision_callback for reviewer.py — receiving a
     ValidationCritique and re-prompting the LLM to correct flaws.
  5. Return fully populated ExamQuestion instances ready for the
     reviewer pipeline.

Hardware note:
  All LLM calls are synchronous and single-threaded to prevent OOM
  crashes on the GTX 1070 (8 GB VRAM).  The embedding model
  (all-MiniLM-L6-v2) runs on CPU so it never competes for VRAM.
"""

from __future__ import annotations

import re
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from config import (
    BASE_DIR,
    CHROMA_DIR,
    EMBEDDING_CONFIG,
    GENERATION_CONFIG,
    OLLAMA_CONFIG,
    REGISTRY_PATH,
)
from schemas import (
    ExamQuestion,
    GenerationRequest,
    GenerationResult,
    QuestionType,
    ValidationCritique,
    ValidationStatus,
)
from utils import (
    build_output_filename,
    export_to_csv,
    export_to_json,
    export_to_moodle_xml,
    extract_json_from_text,
    get_cert_metadata,
    hash_question_text,
    load_prior_question_hashes,
    load_registry,
    make_progress_bar,
    print_domain_alignment_table,
    print_generation_summary,
    print_question_table,
    select_domain_by_quota,
)

# ─────────────────────────────────────────────
# Module logger
# ─────────────────────────────────────────────
log = logging.getLogger("aws_exam_gen.generator")


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """\
You are an expert AWS Solutions Architect and certified AWS exam item author \
with deep hands-on experience across all AWS service categories. Your task is \
to write high-quality, scenario-based multiple-choice questions (MCQs) for \
official AWS certification practice exams.

Every question you write MUST:
  - Be grounded in a realistic AWS customer workload scenario.
  - Test understanding of trade-offs (cost vs. performance, HA vs. simplicity, \
security vs. convenience) rather than pure fact recall.
  - Use accurate AWS service names, API actions, limits, and pricing models as \
documented in current AWS FAQs and official documentation.
  - Follow the AWS exam item-writing guidelines: one clearly best answer for \
single-answer questions, plausible but definitively wrong distractors.
  - Mirror the difficulty level and style of questions found in the official \
AWS exam guide for the specified certification tier.

You MUST respond ONLY with a valid JSON object. No prose, no markdown fences, \
no commentary outside the JSON structure.
"""

SINGLE_ANSWER_PROMPT_TEMPLATE = """\
Generate ONE single-answer multiple-choice question for the AWS {cert_name} \
({cert_code}) exam.

Target domain  : {domain}
Difficulty tier: {difficulty}
{avoid_note}
─── REFERENCE CONTEXT (from official AWS documentation) ──────────────────────
{context}
──────────────────────────────────────────────────────────────────────────────

Requirements:
  - Exactly 4 answer options labelled A, B, C, D.
  - Exactly 1 correct answer.
  - The question stem must describe a realistic customer scenario (≥ 2 sentences).
  - Distractors must be plausible AWS services or configurations that a \
partially-prepared candidate might choose.
  - The explanation must state WHY the correct answer is best AND why each \
distractor is inferior, citing specific AWS service behaviour or limits.
  - Use uppercase letters only for correct_answers.
  - Do NOT include prose, markdown fences, code blocks, or commentary outside the JSON.

Return ONLY a JSON object with this exact schema:
{{
  "cert_code"      : "{cert_code}",
  "domain"         : "{domain}",
  "question_type"  : "single",
  "question_text"  : "<scenario description — minimum 2 sentences>",
  "options"        : [
    "A. <option text>",
    "B. <option text>",
    "C. <option text>",
    "D. <option text>"
  ],
  "correct_answers": ["<single letter A–D>"],
  "explanation"    : "<detailed justification citing AWS documentation>",
  "aws_services"   : ["<ServiceName>", ...]
}}
"""

MULTIPLE_SELECTION_PROMPT_TEMPLATE = """\
Generate ONE multiple-selection question for the AWS {cert_name} \
({cert_code}) exam.

Target domain  : {domain}
Difficulty tier: {difficulty}
Correct answers: {num_correct} (the question stem MUST say "Select {count_word}")
{avoid_note}
─── REFERENCE CONTEXT (from official AWS documentation) ──────────────────────
{context}
──────────────────────────────────────────────────────────────────────────────

Requirements:
  - Exactly 5 answer options labelled A, B, C, D, E.
  - Exactly {num_correct} correct answers.
  - The question stem MUST end with "(Select {count_word})" or \
"(Select {count_word} answers)".
  - The scenario must involve a genuine trade-off or multi-step solution where \
selecting fewer or more than {num_correct} answers would be incorrect.
  - All distractors must be real AWS services or configurations that are \
related to the domain but suboptimal or inapplicable to this scenario.
  - The explanation must cover every option individually.
  - Use uppercase letters only for correct_answers.
  - Do NOT include prose, markdown fences, code blocks, or commentary outside the JSON.

Return ONLY a JSON object with this exact schema:
{{
  "cert_code"      : "{cert_code}",
  "domain"         : "{domain}",
  "question_type"  : "multiple",
  "question_text"  : "<scenario description ending with (Select {count_word})>",
  "options"        : [
    "A. <option text>",
    "B. <option text>",
    "C. <option text>",
    "D. <option text>",
    "E. <option text>"
  ],
  "correct_answers": [<list of {num_correct} letter strings, e.g. "A", "C">],
  "explanation"    : "<per-option justification citing AWS documentation>",
  "aws_services"   : ["<ServiceName>", ...]
}}
"""


# ─────────────────────────────────────────────
# ChromaDB / RAG retrieval
# ─────────────────────────────────────────────

def _get_chroma_collection(cert_code: str) -> Any:
    """
    Return the ChromaDB collection for a given certification.
    Raises RuntimeError if the collection has not been ingested yet.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path    = str(CHROMA_DIR),
            settings= Settings(anonymized_telemetry=False),
        )
        collection_name = f"aws_{cert_code.lower().replace('-', '_')}"

        # List existing collections to check existence
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' not found. "
                f"Run 'python main.py ingest --cert {cert_code}' first."
            )

        return client.get_collection(collection_name)

    except ImportError:
        raise RuntimeError(
            "chromadb is not installed. Run: pip install chromadb"
        )


def _load_embedding_model() -> Any:
    """
    Load the sentence-transformer embedding model onto CPU.
    Returns the SentenceTransformer instance.
    Keeps VRAM free for the Ollama LLM.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            EMBEDDING_CONFIG.model_name,
            device="cpu",    # Explicit CPU — never competes with Ollama VRAM
        )
        log.debug(
            f"Embedding model '{EMBEDDING_CONFIG.model_name}' loaded on CPU."
        )
        return model
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )


def retrieve_context(
    domain:                str,
    collection:            Any,
    embedding_model:       Any,
    n_results:             int              = EMBEDDING_CONFIG.top_k_results,
    exclude_chunk_ids:     Optional[Set[str]] = None,
    max_chunks_per_source: int              = 2,
) -> Tuple[str, Optional[str], List[str]]:
    """
    Retrieve the most relevant document chunks from ChromaDB for a
    given exam domain using semantic similarity search.

    Every question generated for a domain embeds the same domain
    string, so a plain top-k nearest-neighbor query would return the
    same chunks for every question in that domain — and, because a
    handful of source documents (e.g. the ~1,000-page Well-Architected
    Framework PDF) can account for 50-70%+ of a collection's chunks,
    those oversized documents would dominate every retrieval regardless
    of how relevant a smaller, more specific document is. To counter
    both effects this over-fetches a larger candidate pool and then
    greedily selects up to `n_results` chunks while: (1) skipping any
    id in `exclude_chunk_ids` (chunks already used earlier in the same
    domain's generation run, passed in by the caller), and (2) capping
    how many chunks can come from any single `source_file`.

    Parameters
    ──────────
    domain
        The exam domain string used as the semantic search query.
    collection
        ChromaDB collection containing embedded document chunks.
    embedding_model
        Loaded SentenceTransformer model (CPU) for query embedding.
    n_results
        Number of top chunks to retrieve and concatenate.
    exclude_chunk_ids
        Chunk ids to skip — typically chunks already used for earlier
        questions in the same domain, so repeated calls rotate through
        the corpus instead of returning the same context every time.
    max_chunks_per_source
        Maximum number of chunks allowed from any single source_file
        per call, so one oversized document can't crowd out the rest
        of the corpus.

    Returns
    ───────
    A single string of concatenated context passages, or an empty
    string if retrieval fails or the collection is empty.
    """
    if collection is None:
        log.warning("retrieve_context called with no ChromaDB collection.")
        return "", None, []

    try:
        # Embed the domain string as the semantic search query.
        # Some SentenceTransformer models do not accept extra kwargs.
        try:
            query_embedding = embedding_model.encode(domain)
        except TypeError:
            query_embedding = embedding_model.encode(domain)

        # Convert returned embeddings into a plain Python list if needed.
        if not isinstance(query_embedding, list):
            try:
                query_embedding = query_embedding.tolist()
            except AttributeError:
                query_embedding = list(query_embedding)

        # Clamp n_results to collection size to avoid ChromaDB errors
        # when the collection has fewer documents than requested
        collection_size = collection.count()
        if collection_size == 0:
            log.warning(
                f"ChromaDB collection is empty — "
                "no context available for generation."
            )
            return "", None, []

        clamped_n = min(n_results, collection_size)

        # If the exclude set has grown so large that fewer chunks remain
        # than we need, rotate: drop the exclusions for this call rather
        # than starving retrieval down to near-empty context. This keeps
        # small collections (a few hundred chunks) usable across a full
        # domain's worth of questions instead of degrading over time.
        if exclude_chunk_ids and (collection_size - len(exclude_chunk_ids)) < clamped_n:
            exclude_chunk_ids = None

        # Over-fetch a much larger candidate pool than n_results so the
        # per-source cap and exclusion list have real alternatives to
        # select from. This matters more than it looks: a domain name is
        # a short, generic query, and an oversized, self-similar source
        # document (e.g. a 1,000-page framework PDF) can occupy the
        # entire top few hundred nearest neighbors by itself — a shallow
        # over-fetch (e.g. top-20) would just hand the cap logic more of
        # the same one document instead of a genuine alternative.
        candidate_n = min(collection_size, max(clamped_n * 40, 200))

        results = collection.query(
            query_embeddings = [query_embedding],
            n_results        = candidate_n,
            include          = ["documents", "metadatas", "distances"],
        )

        # results["documents"] is a list of lists (one per query vector)
        documents      = results.get("documents", [[]])[0]
        metadatas      = results.get("metadatas", [[]])[0]
        candidate_ids  = results.get("ids", [[]])[0] if "ids" in results else []

        if not documents:
            log.warning(
                f"ChromaDB returned no documents for domain '{domain}'."
            )
            return "", None, []

        # ── Greedy diversified selection ───────────
        # Rank order from ChromaDB is preserved; a chunk is skipped only
        # if it was already used for this domain or its source has hit
        # the per-call cap. A backfill pass relaxes the source cap (but
        # keeps honoring exclusions) if the cap left us short.
        selected_idx:     List[int]  = []
        per_source_count: Counter    = Counter()
        exclude_chunk_ids = exclude_chunk_ids or set()

        for idx in range(len(documents)):
            if len(selected_idx) >= clamped_n:
                break
            cid = candidate_ids[idx] if idx < len(candidate_ids) else None
            if cid is not None and cid in exclude_chunk_ids:
                continue
            source = metadatas[idx].get("source_file", "unknown")
            if per_source_count[source] >= max_chunks_per_source:
                continue
            selected_idx.append(idx)
            per_source_count[source] += 1

        if len(selected_idx) < clamped_n:
            for idx in range(len(documents)):
                if len(selected_idx) >= clamped_n:
                    break
                if idx in selected_idx:
                    continue
                cid = candidate_ids[idx] if idx < len(candidate_ids) else None
                if cid is not None and cid in exclude_chunk_ids:
                    continue
                selected_idx.append(idx)

        # Concatenate retrieved chunks with source attribution
        source_url: Optional[str] = None
        context_parts: List[str] = []
        chunk_ids: List[str] = []
        for rank, idx in enumerate(selected_idx):
            doc, meta = documents[idx], metadatas[idx]
            source  = meta.get("source_file", "unknown")
            page    = meta.get("page_number", "?")
            chunk_url = meta.get("source_url")
            if source_url is None and chunk_url:
                source_url = chunk_url
            attribution = f"[Source {rank + 1}: {source}, p.{page}]"
            if chunk_url:
                attribution = f"[Source {rank + 1}: {source}, p.{page}, URL: {chunk_url}]"
            context_parts.append(f"{attribution}\n{doc}")
            if idx < len(candidate_ids):
                chunk_ids.append(candidate_ids[idx])

        context = "\n\n".join(context_parts)

        log.debug(
            f"Retrieved {len(selected_idx)} chunk(s) for domain '{domain}' "
            f"from {len(per_source_count)} distinct source(s) "
            f"(collection size: {collection_size}, excluded: {len(exclude_chunk_ids)})."
        )

        return context, source_url, chunk_ids

    except Exception as retrieval_err:
        log.error(
            f"ChromaDB retrieval failed for domain '{domain}': "
            f"{retrieval_err}"
        )
        return "", None, []



# ─────────────────────────────────────────────
# Question type selection
# ─────────────────────────────────────────────

def _select_question_type(
    index: int,
    total: int,
    current_single_count: int,
    current_multi_count: int,
) -> QuestionType:
    """
    Enforce the 75 % single / 25 % multiple split across the batch.

    Uses a deterministic allocation strategy:
      - Calculate how many of each type are needed in total.
      - Select the type that is most "behind" its target quota.
      - Fall back to weighted random when both are on track.

    This ensures the split holds even for small batches (e.g. 4 questions
    → 3 single, 1 multiple).
    """
    target_single = round(total * GENERATION_CONFIG.single_answer_ratio)
    target_multi  = total - target_single

    remaining_total  = total  - index
    remaining_single = target_single - current_single_count
    remaining_multi  = target_multi  - current_multi_count

    # Force single if multi quota is already met
    if remaining_multi <= 0:
        return QuestionType.SINGLE_ANSWER

    # Force multiple if single quota is already met
    if remaining_single <= 0:
        return QuestionType.MULTIPLE_SELECTION

    # Both still needed — choose the one most behind its proportional target
    single_deficit = remaining_single / max(target_single, 1)
    multi_deficit  = remaining_multi  / max(target_multi,  1)

    if single_deficit > multi_deficit:
        return QuestionType.SINGLE_ANSWER
    elif multi_deficit > single_deficit:
        return QuestionType.MULTIPLE_SELECTION
    else:
        # Tie: weighted random to avoid deterministic patterns
        return random.choices(
            [QuestionType.SINGLE_ANSWER, QuestionType.MULTIPLE_SELECTION],
            weights=[
                GENERATION_CONFIG.single_answer_ratio,
                1 - GENERATION_CONFIG.single_answer_ratio,
            ],
            k=1,
        )[0]


def _select_num_correct_for_multi() -> Tuple[int, str]:
    """
    Choose how many correct answers a multiple-selection question has.
    Weighted toward 2 (most common in real AWS exams), with occasional 3.

    Returns
    ───────
    (num_correct, count_word)
        e.g. (2, 'TWO') or (3, 'THREE')
    """
    num_correct = random.choices([2, 3], weights=[0.80, 0.20], k=1)[0]
    count_words = {2: "TWO", 3: "THREE", 4: "FOUR"}
    return num_correct, count_words[num_correct]


# ─────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────

def build_generation_prompt(
    cert_code: str,
    cert_name: str,
    domain: str,
    difficulty: str,
    question_type: QuestionType,
    context: str,
    num_correct: int = 2,
    count_word: str = "TWO",
    avoid_services: Optional[List[str]] = None,
) -> str:
    """
    Assemble the user-turn prompt for the LLM generation call.

    Inserts a truncated context block to stay within the configured
    OLLAMA_CONFIG.num_ctx limit (4096 tokens on the GTX 1070 dev box,
    override via the OLLAMA_NUM_CTX env var on higher-VRAM systems).

    Context budget:
      System + user prompt ≈  750 tokens (template scaffolding)
      Response             ≈ OLLAMA_CONFIG.max_tokens (generation cap)
      Context block        ≈ remainder of num_ctx, at ~4 chars/token

    avoid_services
        AWS services that have already shown up disproportionately
        often among this domain's previously generated questions
        (see generate_batch). Surfaced to the LLM as a soft steer, not
        a hard rule — the retrieved context is still the source of
        truth, but this discourages defaulting to whichever service
        the retrieved chunks keep repeating.
    """
    # Truncate context to avoid exceeding num_ctx
    reserved_tokens   = 750 + OLLAMA_CONFIG.max_tokens
    context_budget    = max(OLLAMA_CONFIG.num_ctx - reserved_tokens, 500)
    max_context_chars = context_budget * 4  # ~4 chars/token
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n… [context truncated]"

    avoid_note = ""
    if avoid_services:
        avoid_note = (
            "Already emphasized in this domain — do not center the question "
            f"on these unless clearly the best fit: {', '.join(avoid_services)}\n"
        )

    if question_type == QuestionType.SINGLE_ANSWER:
        return SINGLE_ANSWER_PROMPT_TEMPLATE.format(
            cert_name  = cert_name,
            cert_code  = cert_code,
            domain     = domain,
            difficulty = difficulty,
            avoid_note = avoid_note,
            context    = context or "(No context retrieved — use general AWS knowledge.)",
        )
    else:
        return MULTIPLE_SELECTION_PROMPT_TEMPLATE.format(
            cert_name  = cert_name,
            cert_code  = cert_code,
            domain     = domain,
            difficulty = difficulty,
            avoid_note = avoid_note,
            context    = context or "(No context retrieved — use general AWS knowledge.)",
            num_correct= num_correct,
            count_word = count_word,
        )


# ─────────────────────────────────────────────
# Ollama LLM call
# ─────────────────────────────────────────────

@retry(
    retry   = retry_if_exception_type((ConnectionError, TimeoutError)),
    stop    = stop_after_attempt(3),
    wait    = wait_fixed(3),
    reraise = True,
)
def _call_ollama_generator(
    user_prompt: str,
    ollama_client: Any,
) -> str:
    """
    Send a generation prompt to Ollama and return the raw response string.

    Uses a slightly higher temperature than the reviewer (0.4) to
    encourage creative, varied question scenarios while remaining
    factually grounded.

    Retries up to 3 times on transient network / timeout errors.
    """
    response = ollama_client.chat(
        model    = OLLAMA_CONFIG.model,
        messages = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        options  = {
            "temperature": OLLAMA_CONFIG.temperature,
            "top_p"      : OLLAMA_CONFIG.top_p,
            "num_ctx"    : OLLAMA_CONFIG.num_ctx,
            "num_predict": OLLAMA_CONFIG.max_tokens,
        },
        format    = "json",
        keep_alive= OLLAMA_CONFIG.keep_alive,
    )
    return response["message"]["content"]


# ─────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────

def _parse_question_response(
    raw_text: str,
    cert_code: str,
    domain: str,
    question_type: QuestionType,
    chunk_ids: List[str],
    source_url: Optional[str] = None,
) -> Optional[ExamQuestion]:
    """
    Parse the LLM's raw response text into a validated ExamQuestion.

    Applies several normalisation steps before Pydantic validation:
      - Extracts JSON from surrounding prose or markdown fences.
      - Normalises option labels to 'X. text' format.
      - Upper-cases correct_answer letters.
      - Injects identity fields (cert_code, domain, question_type)
        in case the LLM omitted or misspelled them.
      - Attaches RAG source chunk IDs for provenance.

    Returns None if parsing or Pydantic validation fails entirely.
    """
    data = extract_json_from_text(raw_text)
    if data is None:
        log.warning(
            f"[{cert_code}] Failed to extract JSON from generation response."
        )
        return None

    try:
        # ── Normalise identity fields ─────────────
        data["cert_code"]     = cert_code
        data["domain"]        = domain
        data["question_type"] = question_type.value

        # ── Normalise correct_answers ─────────────
        raw_correct = data.get("correct_answers", [])
        if isinstance(raw_correct, str):
            # LLM sometimes returns a single string instead of a list
            raw_correct = [raw_correct]
        data["correct_answers"] = [
            a.strip().upper() for a in raw_correct if a
        ]

        # ── Normalise explanation ─────────────────
        # LLM sometimes returns a list of per-option strings instead of
        # a single explanation paragraph.
        raw_explanation = data.get("explanation", "")
        if isinstance(raw_explanation, list):
            data["explanation"] = "\n\n".join(str(item) for item in raw_explanation)
        elif not isinstance(raw_explanation, str):
            data["explanation"] = str(raw_explanation)

        # ── Normalise options ─────────────────────
        raw_options = data.get("options", [])
        normalised_options = _normalise_options(raw_options)
        data["options"] = normalised_options

        # ── Attach RAG provenance ─────────────────
        data["rag_source_chunks"] = chunk_ids
        data["source_url"]        = source_url

        # ── Default aws_services if absent ────────
        if "aws_services" not in data or not data["aws_services"]:
            data["aws_services"] = []

        # ── Build Pydantic model ──────────────────
        question = ExamQuestion(**data)
        return question

    except Exception as parse_err:
        log.warning(
            f"[{cert_code}] ExamQuestion construction failed: {parse_err}\n"
            f"  Raw data keys: {list(data.keys()) if data else 'N/A'}"
        )
        return None


def _validate_generated_question(question: ExamQuestion) -> List[str]:
    """Return a list of structural validation errors for generated questions."""
    errors: List[str] = []

    if question.question_type == QuestionType.SINGLE_ANSWER:
        if len(question.options) != 4:
            errors.append(
                f"Single-answer question must have exactly 4 options, "
                f"but got {len(question.options)}."
            )
        if len(question.correct_answers) != 1:
            errors.append(
                f"Single-answer question must have exactly 1 correct answer, "
                f"but got {len(question.correct_answers)}."
            )
    else:
        if len(question.options) != 5:
            errors.append(
                f"Multiple-selection question must have exactly 5 options, "
                f"but got {len(question.options)}."
            )
        if len(question.correct_answers) < 2:
            errors.append(
                f"Multiple-selection question must have at least 2 correct answers, "
                f"but got {len(question.correct_answers)}."
            )
        if len(question.correct_answers) > 4:
            errors.append(
                f"Multiple-selection question must have no more than 4 correct answers, "
                f"but got {len(question.correct_answers)}."
            )

    option_labels = []
    for option in question.options:
        if "." not in option:
            errors.append(
                f"Option '{option}' is missing a label in the required 'X. text' format."
            )
            continue
        label = option.split(".")[0].strip().upper()
        if label not in ["A", "B", "C", "D", "E"]:
            errors.append(
                f"Option label '{label}' is invalid; it must be A, B, C, D or E."
            )
        option_labels.append(label)

    invalid_answers = [
        ans for ans in question.correct_answers
        if ans.upper() not in option_labels
    ]
    if invalid_answers:
        errors.append(
            f"Correct answers {invalid_answers} do not match option labels {option_labels}."
        )

    if len(question.question_text.strip()) < 50:
        errors.append(
            "Question text is too short; it should be a realistic scenario of at least 50 characters."
        )

    if len(question.explanation.strip()) < 120:
        errors.append(
            "Explanation is too short; it should justify the correct answer and dismiss each distractor."
        )

    if question.question_type == QuestionType.MULTIPLE_SELECTION:
        count_word_map = {2: "TWO", 3: "THREE", 4: "FOUR"}
        expected = count_word_map.get(len(question.correct_answers), str(len(question.correct_answers)))
        if expected and f"select {expected}".lower() not in question.question_text.lower():
            errors.append(
                "Multiple-selection stem must include 'Select {expected}' or "
                "'(Select {expected})'."
            )

    return errors


def _normalise_options(raw_options: List[Any]) -> List[str]:
    """
    Ensure each option follows the 'X. <text>' format.

    Handles common LLM deviations:
      - Options returned as plain strings without labels.
      - Options using ')' separator instead of '.'.
      - Options with extra whitespace or newlines.
      - Dict objects (some models return {'label': 'A', 'text': '...'}).
    """
    expected_labels = ["A", "B", "C", "D", "E"]
    normalised: List[str] = []

    for idx, opt in enumerate(raw_options):
        label = expected_labels[idx] if idx < len(expected_labels) else str(idx)

        # Handle dict format from some model outputs
        if isinstance(opt, dict):
            text = opt.get("text") or opt.get("content") or str(opt)
            opt  = text

        opt = str(opt).strip()

        # Already in correct format: "A. text" or "A) text"
        if len(opt) >= 2 and opt[0].upper() in expected_labels and opt[1] in (".", ")"):
            # Standardise separator to '.'
            normalised.append(f"{opt[0].upper()}. {opt[2:].strip()}")
            continue

        # Option has no label prefix — inject one
        normalised.append(f"{label}. {opt}")

    return normalised


# ─────────────────────────────────────────────
# Single question generation
# ─────────────────────────────────────────────

def generate_single_question(
    cert_code:         str,
    cert_name:         str,
    domain:            str,
    difficulty:        str,
    question_type:     QuestionType,
    collection:        Any,
    embedding_model:   Any,
    ollama_client:     Any,
    num_correct:       int = 1,
    count_word:        str = "ONE",
    exclude_chunk_ids: Optional[Set[str]] = None,
    avoid_services:    Optional[List[str]] = None,
    seen_hashes:       Optional[Set[str]] = None,
) -> Optional[ExamQuestion]:
    """
    Generate a single ExamQuestion via RAG + Ollama prompt.

    exclude_chunk_ids
        Chunk ids already used for earlier questions in this domain
        (see generate_batch), so retrieval rotates through the corpus
        instead of returning the same top-k chunks for every question
        in the domain. The chunks actually used end up on the returned
        question's `rag_source_chunks` field for the caller to fold in.
    avoid_services
        AWS services already over-represented in this domain so far
        (see generate_batch) — passed through to build_generation_prompt
        as a soft steer against the LLM defaulting to the same service.
    seen_hashes
        Normalized question_text hashes to reject as duplicates —
        prior-run exports plus everything accepted so far this run
        (see generate_batch / utils.load_prior_question_hashes). A
        match is treated the same as a structural validation failure:
        retried up to max_generation_attempts, then the slot fails.
        On success this set gains the new question's hash in place.

    Returns None on any failure so the caller can skip the slot
    gracefully without crashing the batch.
    """
    try:
        for attempt in range(1, GENERATION_CONFIG.max_generation_attempts + 1):
            # ── Step 1: Retrieve context ───────────
            context, source_url, chunk_ids = retrieve_context(
                domain            = domain,
                collection        = collection,
                embedding_model   = embedding_model,
                n_results         = EMBEDDING_CONFIG.top_k_results,
                exclude_chunk_ids = exclude_chunk_ids,
            )

            if not context:
                log.warning(
                    f"No context retrieved for domain '{domain}'. "
                    "Attempting generation without RAG context."
                )
                context = (
                    f"AWS certification domain: {domain}. "
                    f"Certification: {cert_name}."
                )

            # ── Step 2: Build prompt ───────────────
            try:
                prompt = build_generation_prompt(
                    cert_code      = cert_code,
                    cert_name      = cert_name,
                    domain         = domain,
                    difficulty     = difficulty,
                    question_type  = question_type,
                    context        = context,
                    num_correct    = num_correct,
                    count_word     = count_word,
                    avoid_services = avoid_services,
                )
            except Exception as prompt_err:
                log.error(f"Prompt build failed for domain '{domain}': {prompt_err}")
                return None

            # ── Step 3: Call Ollama ────────────────
            try:
                raw_text = _call_ollama_generator(prompt, ollama_client)
            except Exception as ollama_err:
                log.error(f"Ollama call failed for domain '{domain}': {ollama_err}")
                continue

            if not raw_text or not raw_text.strip():
                log.warning(
                    f"Ollama returned empty content for domain '{domain}'."
                )
                continue

            log.debug(
                f"Raw LLM response for domain '{domain}' "
                f"({len(raw_text)} chars):\n{raw_text[:300]}…"
            )

            # ── Step 5: Parse response ─────────────
            try:
                question = _parse_question_response(
                    raw_text      = raw_text,
                    cert_code     = cert_code,
                    domain        = domain,
                    question_type = question_type,
                    chunk_ids     = chunk_ids,
                    source_url    = source_url,
                )
            except Exception as parse_err:
                log.error(
                    f"Response parsing failed for domain '{domain}': {parse_err}"
                )
                continue

            if question is None:
                log.warning(
                    f"Response parsing returned None for domain '{domain}'."
                )
                continue

            validation_errors = _validate_generated_question(question)
            if validation_errors:
                log.warning(
                    f"Generated question failed structural validation for domain '{domain}': "
                    f"{validation_errors}"
                )
                if attempt < GENERATION_CONFIG.max_generation_attempts:
                    log.info(
                        f"Retrying generation for domain '{domain}' "
                        f"(attempt {attempt + 1}/{GENERATION_CONFIG.max_generation_attempts})."
                    )
                    continue
                log.warning(
                    f"Giving up after {attempt} generation attempts for domain '{domain}'."
                )
                return None

            # ── Step 5b: Cross-run + within-run duplicate check ──
            if seen_hashes is not None:
                text_hash = hash_question_text(question.question_text)
                if text_hash in seen_hashes:
                    log.warning(
                        f"Generated question duplicates a prior/earlier question for "
                        f"domain '{domain}' (question_text hash already seen): "
                        f"{question.question_text[:80]}…"
                    )
                    if attempt < GENERATION_CONFIG.max_generation_attempts:
                        log.info(
                            f"Retrying generation for domain '{domain}' "
                            f"(attempt {attempt + 1}/{GENERATION_CONFIG.max_generation_attempts})."
                        )
                        continue
                    log.warning(
                        f"Giving up after {attempt} generation attempts for domain "
                        f"'{domain}' — kept producing a duplicate question."
                    )
                    return None
                seen_hashes.add(text_hash)

            question.difficulty = difficulty
            question.validation_status = ValidationStatus.PENDING
            question.revision_count = 0
            question.source_context = context[:500]
            if source_url:
                question.source_url = source_url

            log.debug(
                f"Successfully generated question for domain '{domain}': "
                f"{question.question_text[:80]}…"
            )

            return question

        log.warning(
            f"No valid question generated for domain '{domain}' after "
            f"{GENERATION_CONFIG.max_generation_attempts} attempts."
        )
        return None

    except Exception as outer_err:
        log.error(
            f"Unhandled exception in generate_single_question "
            f"for domain '{domain}': {outer_err}",
            exc_info=True,
        )
        return None



# ─────────────────────────────────────────────
# Revision callback (called by reviewer.py)
# ─────────────────────────────────────────────

def build_revision_callback(
    ollama_client: Any,
    collection: Any,
    embedding_model: Any,
) -> Any:
    """
    Factory function that returns a revision_callback closure.

    The closure is passed to reviewer.review_question() as the
    mechanism for requesting corrected questions when validation fails.

    The callback:
      1. Receives the failed ExamQuestion and its ValidationCritique.
      2. Imports build_revision_prompt from reviewer to construct
         the correction instruction.
      3. Calls the LLM with that prompt.
      4. Parses the response into a revised ExamQuestion.
      5. Returns the revised question (or raises on total failure).

    Using a factory keeps the generator and reviewer decoupled while
    sharing the same Ollama client and VRAM context.
    """
    def revision_callback(
        question: ExamQuestion,
        critique: ValidationCritique,
    ) -> ExamQuestion:
        # Import here to avoid circular imports at module load time
        from reviewer import build_revision_prompt

        log.info(
            f"[{question.question_id[:8]}] Applying revision "
            f"(attempt {question.revision_count + 1}) — "
            f"score was {critique.overall_score:.1f}/10."
        )

        # Build the revision instruction prompt
        revision_prompt = build_revision_prompt(question, critique)

        # Call LLM with the correction instructions
        try:
            raw_response = _call_ollama_generator(revision_prompt, ollama_client)
        except Exception as llm_err:
            raise RuntimeError(
                f"LLM revision call failed for question "
                f"[{question.question_id[:8]}]: {llm_err}"
            ) from llm_err

        # Parse the revised question
        revision_data = extract_json_from_text(raw_response)
        if revision_data is None:
            raise ValueError(
                f"LLM returned unparseable JSON during revision for "
                f"question [{question.question_id[:8]}]."
            )

        # Apply the revision using the reviewer's merge helper
        from reviewer import _apply_revision_to_question
        revised_question = _apply_revision_to_question(question, revision_data)

        log.debug(
            f"[{revised_question.question_id[:8]}] Revision applied — "
            f"revision_count now {revised_question.revision_count}."
        )

        return revised_question

    return revision_callback


# ─────────────────────────────────────────────
# Batch generation orchestrator
# ─────────────────────────────────────────────

def generate_batch(
    request: GenerationRequest,
    ollama_client: Any,
    collection: Any,
    embedding_model: Any,
    cert_metadata: Dict[str, Any],
    slots:              Optional[int] = None,
    start_index:        int           = 0,
    single_count_start: int           = 0,
    multi_count_start:  int           = 0,
    domain_count_start: Optional[Dict[str, int]] = None,
    seen_hashes:        Optional[Set[str]] = None,
) -> List[ExamQuestion]:
    """
    Generate a batch of ExamQuestion objects according to the request.

    Enforces:
      - Domain weighting from data/exam-domains.txt (the exam guide's
        Content Domain breakdown; see utils.get_cert_metadata)
      - 75 % single / 25 % multiple split via _select_question_type()
      - Sequential execution (GTX 1070 single-thread constraint)
      - Configurable domain filtering via request.domain_filter
      - Cross-run + within-run question uniqueness (see seen_hashes)

    Parameters
    ──────────
    request
        A GenerationRequest specifying cert_code, count, etc.
    ollama_client
        Initialised Ollama client.
    collection
        ChromaDB collection for the cert_code.
    embedding_model
        Loaded SentenceTransformer (CPU).
    cert_metadata
        Certification metadata dict from registry.json, with "domains" /
        "domain_weights" overlaid from data/exam-domains.txt.
    slots
        Number of questions to generate on *this* call. Defaults to
        `request.count`. Used by the top-up loop in
        `run_generation_pipeline` to backfill only the shortfall after
        earlier rounds, without losing the 75/25 type-ratio target
        (which is always computed against `request.count` as a whole).
    start_index, single_count_start, multi_count_start, domain_count_start
        Running progress from prior rounds, so `_select_question_type` and
        `select_domain_by_quota` keep allocating against the overall
        request total instead of restarting their ratios from zero.
    seen_hashes
        Normalized-text hashes (see utils.hash_question_text) to reject
        as duplicates — typically seeded from every prior approved
        export for this cert_code (utils.load_prior_question_hashes)
        and mutated in place as this run accepts new questions, so it
        also catches duplicates within the same run and across top-up
        rounds. Pass the same set object into successive calls (as
        run_generation_pipeline's top-up loop does) to keep it growing
        across rounds; omit it to disable duplicate checking entirely.

    Returns
    ───────
    List of ExamQuestion objects (validation_status = Pending).
    """
    cert_code      = request.cert_code
    cert_name      = cert_metadata["name"]
    difficulty     = request.difficulty or cert_metadata["tier"]
    domain_weights = cert_metadata["domain_weights"]
    total          = request.count
    slots_to_fill  = slots if slots is not None else total

    questions:          List[ExamQuestion] = []
    single_count: int = single_count_start
    multi_count:  int = multi_count_start
    domain_counts: Dict[str, int] = dict(domain_count_start or {})
    failed_slots: int = 0

    # Chunk ids already used for each domain in this batch, so
    # retrieve_context() rotates through the corpus instead of handing
    # every question in a domain the same top-k chunks (see its
    # docstring for why that matters).
    domain_used_chunks: Dict[str, Set[str]] = {}

    # AWS services already covered per domain in this batch (from the
    # LLM's self-reported aws_services field), so overused services can
    # be flagged to the LLM as "already emphasized" — the prompt-side
    # complement to domain_used_chunks above.
    domain_service_counts: Dict[str, Counter] = {}

    # Mutated in place (not reassigned) so the caller's set — seeded
    # from prior runs and shared across top-up rounds — keeps growing
    # across every call rather than resetting each time.
    if seen_hashes is None:
        seen_hashes = set()

    log.info(
        f"Starting batch generation: {slots_to_fill} questions for {cert_code} "
        f"({cert_name}) — difficulty: {difficulty}."
    )

    with make_progress_bar() as progress:
        task = progress.add_task(
            f"[cyan]Generating {cert_code} questions…",
            total=slots_to_fill,
        )

        for offset in range(slots_to_fill):
            idx = start_index + offset

            # ── Select question type ───────────────
            q_type = _select_question_type(
                index                = idx,
                total                = total,
                current_single_count = single_count,
                current_multi_count  = multi_count,
            )

            # ── Select domain ──────────────────────
            domain = select_domain_by_quota(
                domain_weights = domain_weights,
                domain_counts  = domain_counts,
                total          = total,
                domain_filter  = request.domain_filter,
            )

            # ── Select num_correct for multi ───────
            num_correct, count_word = (
                _select_num_correct_for_multi()
                if q_type == QuestionType.MULTIPLE_SELECTION
                else (1, "ONE")
            )

            log.debug(
                f"  [{offset + 1}/{slots_to_fill}] type={q_type.value} | "
                f"domain='{domain}' | correct={num_correct}"
            )

            # ── Services to steer away from ────────
            # Once a domain has enough questions to judge from, flag any
            # AWS service that's already showing up in a disproportionate
            # share of that domain's questions.
            domain_so_far   = domain_counts.get(domain, 0)
            avoid_services: Optional[List[str]] = None
            if domain_so_far >= GENERATION_CONFIG.service_repeat_min_sample:
                threshold = max(
                    2, round(domain_so_far * GENERATION_CONFIG.service_repeat_threshold_pct)
                )
                overused = [
                    svc for svc, count in
                    domain_service_counts.get(domain, Counter()).most_common()
                    if count >= threshold
                ]
                avoid_services = overused[:5] or None

            # ── Generate question ──────────────────
            question = generate_single_question(
                cert_code         = cert_code,
                cert_name         = cert_name,
                domain            = domain,
                difficulty        = difficulty,
                question_type     = q_type,
                collection        = collection,
                embedding_model   = embedding_model,
                ollama_client     = ollama_client,
                num_correct       = num_correct,
                count_word        = count_word,
                exclude_chunk_ids = domain_used_chunks.get(domain),
                avoid_services    = avoid_services,
                seen_hashes       = seen_hashes,
            )

            # ── Handle generation failure ──────────
            if question is None:
                failed_slots += 1
                log.warning(
                    f"  [{offset + 1}/{slots_to_fill}] Generation failed — "
                    f"slot will be skipped (total failed: {failed_slots})."
                )
                progress.advance(task)
                continue

            # ── Update type / domain counters ──────
            if question.question_type == QuestionType.SINGLE_ANSWER:
                single_count += 1
            else:
                multi_count += 1
            domain_counts[question.domain] = domain_counts.get(question.domain, 0) + 1
            domain_used_chunks.setdefault(question.domain, set()).update(
                question.rag_source_chunks
            )
            domain_service_counts.setdefault(question.domain, Counter()).update(
                question.aws_services
            )

            questions.append(question)

            progress.advance(task)
            progress.update(
                task,
                description=(
                    f"[cyan]Generating {cert_code} questions… "
                    f"[{len(questions)}/{slots_to_fill} generated, "
                    f"{single_count} single, {multi_count} multi]"
                ),
            )

    log.info(
        f"Batch generation complete: {len(questions)} questions produced, "
        f"{failed_slots} slot(s) failed. "
        f"Split: {single_count} single / {multi_count} multiple."
    )

    return questions


# ─────────────────────────────────────────────
# Full pipeline runner
# ─────────────────────────────────────────────

def run_generation_pipeline(
    request: GenerationRequest,
) -> GenerationResult:
    """
    Top-level entry point that orchestrates the complete generation +
    validation pipeline for a single GenerationRequest.

    Steps:
      1.  Load the certification registry and validate the cert_code.
      2.  Initialise the Ollama client (model stays resident in VRAM).
      3.  Load the sentence-transformer embedding model (CPU).
      4.  Retrieve the ChromaDB collection for the certification.
      5.  Generate the raw question batch via generate_batch().
      6.  Wire the revision_callback and run the reviewer pipeline
          via reviewer.review_batch().
      6b. Top up: if generation failures or reviewer rejections left
          the approved count short of request.count, run further
          generate + review rounds (capped by
          GENERATION_CONFIG.max_topup_rounds) until the target is
          met or the round budget is exhausted.
      7.  Export approved questions to the requested output format.
      8.  Return a GenerationResult summary to the CLI.

    All heavy objects (Ollama client, embedding model, ChromaDB
    collection) are initialised once and shared across the entire
    pipeline to minimise VRAM / memory churn on the GTX 1070.

    Parameters
    ──────────
    request
        Fully populated GenerationRequest from the CLI.

    Returns
    ───────
    GenerationResult with counts, output file path, and all questions.
    """
    pipeline_start = time.monotonic()

    # ── Step 1: Load registry ──────────────────
    log.info("Loading certification registry…")
    try:
        registry      = load_registry(REGISTRY_PATH)
        cert_metadata = get_cert_metadata(request.cert_code, registry)
    except (FileNotFoundError, ValueError) as reg_err:
        raise RuntimeError(
            f"Registry error: {reg_err}"
        ) from reg_err

    log.info(
        f"Certification: {cert_metadata['name']} "
        f"({request.cert_code}) | Tier: {cert_metadata['tier']}"
    )

    # ── Step 2: Initialise Ollama client ───────
    log.info(
        f"Initialising Ollama client → model: {OLLAMA_CONFIG.model} | "
        f"num_ctx: {OLLAMA_CONFIG.num_ctx} | "
        f"keep_alive: {OLLAMA_CONFIG.keep_alive}"
    )
    try:
        ollama_client = ollama.Client(
            host    = OLLAMA_CONFIG.base_url,
            timeout = OLLAMA_CONFIG.timeout,
        )
        # Warm-up ping to confirm Ollama is reachable before heavy work
        ollama_client.list()
        log.info("Ollama client connected successfully.")
    except Exception as ollama_err:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_CONFIG.base_url}. "
            f"Ensure Ollama is running. Error: {ollama_err}"
        ) from ollama_err

    # ── Step 3: Load embedding model ──────────
    log.info(
        f"Loading embedding model '{EMBEDDING_CONFIG.model_name}' on CPU…"
    )
    try:
        embedding_model = _load_embedding_model()
    except RuntimeError as emb_err:
        raise RuntimeError(
            f"Embedding model load failed: {emb_err}"
        ) from emb_err

    # ── Step 4: Load ChromaDB collection ──────
    log.info(
        f"Connecting to ChromaDB collection for {request.cert_code}…"
    )
    try:
        collection = _get_chroma_collection(request.cert_code)
        log.info(
            f"ChromaDB collection loaded: "
            f"{collection.count()} chunks available."
        )
    except RuntimeError as chroma_err:
        raise RuntimeError(
            f"ChromaDB error: {chroma_err}"
        ) from chroma_err

    # ── Step 4b: Load prior-run question hashes ─
    # Cross-run duplicate detection: every previously exported question
    # for this cert_code (output/{cert_code}_*.json — export_to_json
    # only ever writes approved questions) is hashed up front. The same
    # set is threaded through generate_batch() below and every top-up
    # round, and grows as new questions are accepted, so duplicates are
    # rejected against prior runs AND everything generated so far in
    # this run.
    from config import OUTPUT_DIR
    seen_question_hashes = load_prior_question_hashes(
        cert_code  = request.cert_code,
        output_dir = OUTPUT_DIR,
    )

    # ── Step 5: Generate raw question batch ───
    log.info(
        f"Generating {request.count} question(s) for {request.cert_code}…"
    )
    raw_questions = generate_batch(
        request         = request,
        ollama_client   = ollama_client,
        collection      = collection,
        embedding_model = embedding_model,
        cert_metadata   = cert_metadata,
        seen_hashes     = seen_question_hashes,
    )

    if not raw_questions:
        log.error(
            "Generation produced zero valid questions. "
            "Check Ollama model availability and ChromaDB ingestion."
        )
        return GenerationResult(
            cert_code        = request.cert_code,
            requested        = request.count,
            approved         = 0,
            rejected         = 0,
            questions        = [],
            output_file      = None,
            duration_seconds = time.monotonic() - pipeline_start,
            metadata         = {"error": "No questions generated."},
        )

    log.info(
        f"Raw generation complete: {len(raw_questions)} question(s) "
        f"ready for review."
    )

    # ── Step 6: Wire revision callback & run reviewer ──
    log.info("Starting AI reviewer validation loop…")

    revision_cb = build_revision_callback(
        ollama_client   = ollama_client,
        collection      = collection,
        embedding_model = embedding_model,
    )

    # Import here to maintain clean separation of concerns and
    # avoid any circular import at module load time
    from reviewer import review_batch

    reviewed_questions, critiques = review_batch(
        questions         = raw_questions,
        ollama_client     = ollama_client,
        revision_callback = revision_cb,
    )

    # ── Tally results ──────────────────────────
    approved_questions = [
        q for q in reviewed_questions
        if q.validation_status == ValidationStatus.APPROVED
    ]
    rejected_questions = [
        q for q in reviewed_questions
        if q.validation_status == ValidationStatus.REJECTED
    ]

    log.info(
        f"Review complete — Approved: {len(approved_questions)} | "
        f"Rejected: {len(rejected_questions)}"
    )

    # ── Step 6b: Top-up rounds ─────────────────
    # Generation failures and reviewer rejections both shrink the final
    # approved count below request.count. Keep generating + reviewing
    # additional slots — tracking single/multi counts, domain counts, and
    # the index across rounds so the 75/25 ratio and the content-outline
    # domain weights both still hold — until the target is met or we
    # give up (round cap, or repeated zero-progress rounds).
    generated_total = len(raw_questions)
    single_count = sum(
        1 for q in raw_questions if q.question_type == QuestionType.SINGLE_ANSWER
    )
    multi_count = len(raw_questions) - single_count
    domain_counts: Dict[str, int] = dict(Counter(q.domain for q in raw_questions))

    topup_round = 0
    zero_progress_rounds = 0

    while (
        len(approved_questions) < request.count
        and topup_round < GENERATION_CONFIG.max_topup_rounds
    ):
        topup_round += 1
        shortfall = request.count - len(approved_questions)
        log.info(
            f"Top-up round {topup_round}/{GENERATION_CONFIG.max_topup_rounds}: "
            f"{len(approved_questions)}/{request.count} approved so far — "
            f"generating {shortfall} more to backfill."
        )

        topup_raw = generate_batch(
            request            = request,
            ollama_client      = ollama_client,
            collection         = collection,
            embedding_model    = embedding_model,
            cert_metadata      = cert_metadata,
            slots              = shortfall,
            start_index        = generated_total,
            single_count_start = single_count,
            multi_count_start  = multi_count,
            domain_count_start = domain_counts,
            seen_hashes        = seen_question_hashes,
        )

        if not topup_raw:
            log.warning(
                f"Top-up round {topup_round} produced zero raw questions — "
                "stopping backfill."
            )
            break

        generated_total += len(topup_raw)
        single_count += sum(
            1 for q in topup_raw if q.question_type == QuestionType.SINGLE_ANSWER
        )
        multi_count = generated_total - single_count
        domain_counts = dict(Counter(domain_counts) + Counter(q.domain for q in topup_raw))

        topup_reviewed, topup_critiques = review_batch(
            questions         = topup_raw,
            ollama_client     = ollama_client,
            revision_callback = revision_cb,
        )
        reviewed_questions.extend(topup_reviewed)
        critiques.extend(topup_critiques)

        newly_approved = [
            q for q in topup_reviewed
            if q.validation_status == ValidationStatus.APPROVED
        ]
        approved_questions.extend(newly_approved)
        rejected_questions.extend(
            q for q in topup_reviewed
            if q.validation_status == ValidationStatus.REJECTED
        )

        log.info(
            f"Top-up round {topup_round} complete — "
            f"{len(newly_approved)} newly approved "
            f"({len(approved_questions)}/{request.count} total)."
        )

        if newly_approved:
            zero_progress_rounds = 0
        else:
            zero_progress_rounds += 1
            if zero_progress_rounds >= GENERATION_CONFIG.max_zero_progress_rounds:
                log.warning(
                    f"{zero_progress_rounds} consecutive top-up round(s) approved "
                    "nothing — stopping backfill early."
                )
                break

    approved_count = len(approved_questions)
    rejected_count = len(rejected_questions)

    if approved_count < request.count:
        log.warning(
            f"Final approved count ({approved_count}) is still short of the "
            f"requested {request.count} after {topup_round} top-up round(s)."
        )

    # ── Step 7: Compile content-outline alignment stats ──
    # Compare each domain's share of *approved* questions against its
    # target weight from data/exam-domains.txt (the exam guide's content
    # outline) so alignment can be audited from both the console and the
    # export file, rather than only being inferable by hand.
    domain_weights = cert_metadata["domain_weights"]
    domain_stats: Dict[str, Dict[str, Any]] = {
        domain: {"approved": 0, "rejected": 0, "target_pct": weight * 100}
        for domain, weight in domain_weights.items()
    }
    for q in reviewed_questions:
        entry = domain_stats.setdefault(
            q.domain, {"approved": 0, "rejected": 0, "target_pct": 0.0}
        )
        if q.validation_status == ValidationStatus.APPROVED:
            entry["approved"] += 1
        else:
            entry["rejected"] += 1

    for entry in domain_stats.values():
        actual_pct = (
            round(entry["approved"] / approved_count * 100, 1)
            if approved_count > 0
            else 0.0
        )
        entry["actual_pct"] = actual_pct
        entry["delta_pct"]  = round(actual_pct - entry["target_pct"], 1)

    # ── Step 8: Export output ──────────────────
    output_file: Optional[str] = None

    # Determine output path
    if request.output_path:
        output_path = Path(request.output_path)
    else:
        from config import OUTPUT_DIR
        output_path = build_output_filename(
            cert_code  = request.cert_code,
            count      = approved_count,
            fmt        = request.output_format.value,
            output_dir = OUTPUT_DIR,
        )

    # Export only approved questions (rejected kept in result for audit)
    questions_to_export = (
        approved_questions if approved_questions else reviewed_questions
    )

    try:
        if request.output_format.value == "json":
            export_to_json(
                questions      = questions_to_export,
                output_path    = output_path,
                cert_code      = request.cert_code,
                extra_metadata = {"domain_stats": domain_stats},
            )
        elif request.output_format.value == "csv":
            export_to_csv(
                questions   = questions_to_export,
                output_path = output_path,
            )
        elif request.output_format.value == "moodle_xml":
            export_to_moodle_xml(
                questions   = questions_to_export,
                output_path = output_path,
            )

        output_file = str(output_path)
        log.info(f"Export complete → {output_file}")

    except Exception as export_err:
        log.error(f"Export failed: {export_err}")
        output_file = None

    # ── Step 9: Build and return result ───────
    duration = time.monotonic() - pipeline_start

    # Compile revision statistics
    total_revisions = sum(q.revision_count for q in reviewed_questions)
    avg_revisions   = (
        round(total_revisions / len(reviewed_questions), 2)
        if reviewed_questions
        else 0.0
    )

    # Compile type distribution statistics
    final_single = sum(
        1 for q in reviewed_questions
        if q.question_type == QuestionType.SINGLE_ANSWER
    )
    final_multi = sum(
        1 for q in reviewed_questions
        if q.question_type == QuestionType.MULTIPLE_SELECTION
    )

    result = GenerationResult(
        cert_code        = request.cert_code,
        requested        = request.count,
        approved         = approved_count,
        rejected         = rejected_count,
        questions        = reviewed_questions,
        output_file      = output_file,
        duration_seconds = round(duration, 2),
        metadata         = {
            "cert_name"        : cert_metadata["name"],
            "tier"             : cert_metadata["tier"],
            "model"            : OLLAMA_CONFIG.model,
            "embedding_model"  : EMBEDDING_CONFIG.model_name,
            "domain_stats"     : domain_stats,
            "type_distribution": {
                "single"  : final_single,
                "multiple": final_multi,
            },
            "revision_stats"   : {
                "total_revisions"  : total_revisions,
                "avg_per_question" : avg_revisions,
            },
            "output_format"    : request.output_format.value,
        },
    )

    return result


# ─────────────────────────────────────────────
# Convenience: display helpers for CLI
# ─────────────────────────────────────────────

def display_results(result: GenerationResult) -> None:
    """
    Print the question table and generation summary panel to the console.
    Called by main.py after run_generation_pipeline() returns.
    Separates display logic from pipeline logic for testability.
    """
    if result.questions:
        print_question_table(result.questions)

    print_generation_summary(result)

    domain_stats = result.metadata.get("domain_stats")
    if domain_stats:
        print_domain_alignment_table(domain_stats)
