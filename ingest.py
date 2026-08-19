"""
ingest.py
─────────
Local RAG & Document Ingestion Engine

Responsibilities:
  1. Scan a local directory for PDF files belonging to a specific
     AWS certification (exam guides, whitepapers, FAQs, blueprints).
  2. Extract and clean raw text from each PDF using pdfplumber
     (with pypdf as a fallback).
  3. Split extracted text into overlapping chunks tuned to the
     sentence-transformer embedding model's input window.
  4. Embed each chunk locally via sentence-transformers (CPU-only)
     so VRAM remains fully available for the Ollama LLM.
  5. Upsert all embeddings into a ChromaDB persistent collection
     scoped to the certification code.
  6. Produce a detailed ingestion report for CLI display.

Design constraints:
  - Embedding runs on CPU to never compete with Ollama for VRAM.
  - ChromaDB uses local persistent storage (no network dependency).
  - All operations are synchronous and single-threaded to keep
    memory usage predictable on the GTX 1070 workstation.
  - Previously ingested chunks are detected via SHA-256 content
    hashing and skipped (idempotent re-ingestion).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_CONFIG,
    REGISTRY_PATH,
)
from utils import (
    get_cert_metadata,
    load_registry,
    logger,
    make_progress_bar,
)

# ─────────────────────────────────────────────
# Module logger
# ─────────────────────────────────────────────
log = logging.getLogger("aws_exam_gen.ingest")
console = Console()


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single text chunk ready for embedding and ChromaDB upsert.

    Attributes
    ──────────
    chunk_id
        Deterministic SHA-256 identifier derived from source_file
        + page_number + chunk_index.  Enables idempotent upserts.
    text
        Cleaned, whitespace-normalised chunk text.
    source_file
        Filename (not full path) of the originating PDF.
    page_number
        1-based page number where this chunk begins.
    chunk_index
        0-based position of this chunk within the document.
    cert_code
        AWS certification code this document belongs to.
    domain_hint
        Optional domain keyword extracted from the filename or
        document metadata to improve RAG retrieval relevance.
    token_count
        Approximate token count (len // 4) for budget tracking.
    """
    chunk_id:    str
    text:        str
    source_file: str
    page_number: int
    chunk_index: int
    cert_code:   str
    source_url:  Optional[str] = None
    domain_hint: str        = ""
    token_count: int        = 0

    def to_chroma_document(self) -> str:
        """Return the raw text used as the ChromaDB document body."""
        return self.text

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Return metadata dict stored alongside the embedding."""
        return {
            "source_file" : self.source_file,
            "page_number" : self.page_number,
            "chunk_index" : self.chunk_index,
            "cert_code"   : self.cert_code,
            "source_url"  : self.source_url,
            "domain_hint" : self.domain_hint,
            "token_count" : self.token_count,
        }


@dataclass
class IngestionReport:
    """
    Summary statistics returned after a complete ingestion run.
    Displayed as a Rich table by the CLI.
    """
    cert_code:         str
    source_directory:  str
    pdfs_found:        int              = 0
    pdfs_processed:    int              = 0
    pdfs_failed:       List[str]        = field(default_factory=list)
    total_pages:       int              = 0
    total_chunks:      int              = 0
    chunks_upserted:   int              = 0
    chunks_skipped:    int              = 0
    duration_seconds:  float            = 0.0
    collection_name:   str              = ""
    collection_total:  int              = 0
    per_file_stats:    List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────

def _extract_text_pdfplumber(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract page text from a PDF using pdfplumber.

    Returns a list of (page_number, page_text) tuples (1-based pages).
    pdfplumber is preferred over pypdf for its superior table and
    multi-column layout handling common in AWS whitepapers.

    Raises ImportError if pdfplumber is not installed.
    Raises RuntimeError on unrecoverable PDF parse errors.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        )

    pages: List[Tuple[int, str]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    pages.append((page_num, text))
                except Exception as page_err:
                    log.debug(
                        f"pdfplumber: skipping page {page_num} in "
                        f"'{pdf_path.name}': {page_err}"
                    )
                    pages.append((page_num, ""))
    except Exception as pdf_err:
        raise RuntimeError(
            f"pdfplumber failed to open '{pdf_path.name}': {pdf_err}"
        ) from pdf_err

    return pages


def _extract_text_pypdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Fallback PDF text extractor using pypdf.

    Used automatically when pdfplumber raises an error or is not
    available.  pypdf handles some encrypted / compressed PDFs that
    pdfplumber cannot open.

    Returns a list of (page_number, page_text) tuples.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is not installed. Run: pip install pypdf"
        )

    pages: List[Tuple[int, str]] = []

    try:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                pages.append((page_num, text))
            except Exception as page_err:
                log.debug(
                    f"pypdf: skipping page {page_num} in "
                    f"'{pdf_path.name}': {page_err}"
                )
                pages.append((page_num, ""))
    except Exception as pdf_err:
        raise RuntimeError(
            f"pypdf failed to open '{pdf_path.name}': {pdf_err}"
        ) from pdf_err

    return pages


def _extract_text_html(html_path: Path) -> List[Tuple[int, str]]:
    """
    Extract text from an HTML file using BeautifulSoup.

    Strips boilerplate (nav, header, footer, scripts) then extracts
    clean text from the main content area.  The entire document is
    returned as a single logical "page" (page_number=1) so the
    existing split_text_into_chunks pipeline handles segmentation.

    Returns a list with at most one (1, text) tuple, or an empty
    list if no usable content is found.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is not installed. "
            "Run: pip install beautifulsoup4"
        )

    try:
        raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as read_err:
        raise RuntimeError(
            f"Cannot read '{html_path.name}': {read_err}"
        ) from read_err

    soup = BeautifulSoup(raw_html, "lxml")

    # Remove non-content elements
    for tag in soup(
        ["script", "style", "nav", "header", "footer",
         "noscript", "aside", "iframe", "meta", "link"]
    ):
        tag.decompose()

    # Prefer semantic content containers; fall back to <body>
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.body
    )
    if main is None:
        return []

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < 80:
        log.debug(f"'{html_path.name}': extracted text too short — skipping.")
        return []

    log.debug(
        f"HTML extractor: '{html_path.name}' → "
        f"{len(text)} chars extracted."
    )
    return [(1, text)]


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract (page_number, text) tuples from a PDF or HTML file.

    For HTML files, delegates to _extract_text_html which treats the
    whole document as a single logical page (page_number = 1).
    For PDF files, attempts pdfplumber first; falls back to pypdf.
    Returns an empty list if all extractors fail (triggers skip).

    Parameters
    ──────────
    pdf_path
        Absolute path to the PDF or HTML file.

    Returns
    ───────
    List of (1-based page number, page text) tuples.
    """
    if pdf_path.suffix.lower() == ".html":
        log.debug(f"HTML file detected — using HTML extractor for '{pdf_path.name}'.")
        try:
            return _extract_text_html(pdf_path)
        except Exception as html_err:
            log.error(
                f"HTML extraction failed for '{pdf_path.name}': {html_err}"
            )
            return []

    log.debug(f"Extracting text from '{pdf_path.name}'…")

    # Primary: pdfplumber
    try:
        pages = _extract_text_pdfplumber(pdf_path)
        non_empty = sum(1 for _, t in pages if t.strip())
        log.debug(
            f"pdfplumber extracted {len(pages)} pages "
            f"({non_empty} non-empty) from '{pdf_path.name}'."
        )
        return pages
    except Exception as primary_err:
        log.warning(
            f"pdfplumber failed for '{pdf_path.name}' "
            f"({primary_err}). Trying pypdf fallback…"
        )

    # Fallback: pypdf
    try:
        pages = _extract_text_pypdf(pdf_path)
        non_empty = sum(1 for _, t in pages if t.strip())
        log.debug(
            f"pypdf extracted {len(pages)} pages "
            f"({non_empty} non-empty) from '{pdf_path.name}'."
        )
        return pages
    except Exception as fallback_err:
        log.error(
            f"Both extractors failed for '{pdf_path.name}': "
            f"{fallback_err}"
        )
        return []


# ─────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────

# Common PDF artefacts in AWS documentation
_HEADER_FOOTER_PATTERNS = [
    r"Amazon Web Services\s*[-–]\s*.*",   # Page headers
    r"AWS\s+\w+\s+Guide\s*\|\s*\w+",     # Document title repetitions
    r"©\s*\d{4}\s*Amazon",               # Copyright lines
    r"Page\s+\d+\s+of\s+\d+",           # Page number footers
    r"https?://\S+",                      # Bare URLs (kept in explanation but not chunks)
    r"\f",                                # Form-feed characters
]
_HEADER_FOOTER_RE = re.compile(
    "|".join(_HEADER_FOOTER_PATTERNS),
    re.IGNORECASE,
)

_EXCESSIVE_WHITESPACE_RE = re.compile(r"\s{3,}")
_BULLET_NORMALISE_RE     = re.compile(r"^[\•\▪\◦\-\*]\s+", re.MULTILINE)


def clean_page_text(raw_text: str) -> str:
    """
    Normalise raw PDF text for embedding quality.

    Operations (in order):
      1. Remove page headers, footers, copyright lines, and form feeds.
      2. Normalise bullet-point characters to a standard dash.
      3. Collapse runs of 3+ whitespace characters to a single space.
      4. Strip leading/trailing whitespace.
      5. Drop lines that are purely numeric (page numbers left over).
      6. Remove lines shorter than 4 characters (artefacts).

    Returns the cleaned text string.
    """
    if not raw_text:
        return ""

    # Step 1: Remove header/footer patterns
    text = _HEADER_FOOTER_RE.sub(" ", raw_text)

    # Step 2: Normalise bullets
    text = _BULLET_NORMALISE_RE.sub("- ", text)

    # Step 3: Collapse excessive whitespace (but preserve paragraph breaks)
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        line = _EXCESSIVE_WHITESPACE_RE.sub(" ", line).strip()

        # Step 5: Drop pure-numeric lines (stray page numbers)
        if line.isdigit():
            continue

        # Step 6: Drop very short artefact lines
        if len(line) < 4:
            continue

        cleaned_lines.append(line)

    # Re-join with single newlines, then collapse multiple blank lines
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────

def _approximate_token_count(text: str) -> int:
    """Estimate token count at ~4 characters per token."""
    return max(1, len(text) // 4)


def _char_budget(token_count: int) -> int:
    """Convert a token count to an approximate character budget."""
    return token_count * 4


def split_text_into_chunks(
    text: str,
    chunk_size_tokens: int  = EMBEDDING_CONFIG.chunk_size,
    overlap_tokens: int     = EMBEDDING_CONFIG.chunk_overlap,
) -> List[str]:
    """
    Split a long text string into overlapping token-budget chunks.

    Strategy:
      - Target chunk size: `chunk_size_tokens` (~500 tokens → ~2000 chars).
      - Overlap: `overlap_tokens` (~50 tokens → ~200 chars) carried
        forward from the end of each chunk to preserve context across
        chunk boundaries.
      - Splits preferentially on paragraph boundaries (\n\n), then
        sentence boundaries (. ! ?), then word boundaries as a last
        resort to avoid cutting mid-word.

    Parameters
    ──────────
    text
        Cleaned document text to split.
    chunk_size_tokens
        Target size of each chunk in tokens.
    overlap_tokens
        Number of tokens of overlap between consecutive chunks.

    Returns
    ───────
    List of non-empty text chunk strings.
    """
    if not text.strip():
        return []

    chunk_chars   = _char_budget(chunk_size_tokens)
    overlap_chars = _char_budget(overlap_tokens)

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_chars

        if end >= len(text):
            # Final chunk — take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # ── Prefer paragraph boundary ──────────
        para_break = text.rfind("\n\n", start, end)
        if para_break != -1 and para_break > start + (chunk_chars // 2):
            end = para_break

        else:
            # ── Prefer sentence boundary ───────
            for terminator in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                sent_break = text.rfind(terminator, start, end)
                if sent_break != -1 and sent_break > start + (chunk_chars // 2):
                    end = sent_break + len(terminator)
                    break
            else:
                # ── Fall back to word boundary ─
                word_break = text.rfind(" ", start, end)
                if word_break != -1 and word_break > start:
                    end = word_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = max(start + 1, end - overlap_chars)

    return chunks


# ─────────────────────────────────────────────
# Chunk ID generation
# ─────────────────────────────────────────────

def _generate_chunk_id(
    cert_code:   str,
    source_file: str,
    page_number: int,
    chunk_index: int,
    text:        str,
) -> str:
    """
    Generate a deterministic SHA-256 chunk identifier.

    Including the text content in the hash ensures that if a document
    is updated, existing chunks with changed content get new IDs and
    are re-upserted rather than silently stale.

    Format: sha256(cert_code|source_file|page|chunk_idx|text[:200])
    Truncated to 40 hex characters for ChromaDB ID compatibility.
    """
    content = f"{cert_code}|{source_file}|{page_number}|{chunk_index}|{text[:200]}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:40]


# ─────────────────────────────────────────────
# Domain hint extraction
# ─────────────────────────────────────────────

# Mapping of filename keywords to domain hint strings
_DOMAIN_KEYWORD_MAP: Dict[str, str] = {
    # Architecture / framework
    "architect"     : "Architecture Best Practices",
    "wellarch"      : "Architecture Best Practices",
    "resilient"     : "Resilient Architectures",
    "highavail"     : "High Availability",
    # Compute
    "compute"       : "Compute",
    "ec2"           : "Compute",
    "ecs"           : "Compute",
    "eks"           : "Compute",
    "fargate"       : "Compute",
    "lambda"        : "Serverless",
    "serverless"    : "Serverless",
    "autoscal"      : "Resilient Architectures",
    # Storage
    "storage"       : "Storage",
    "s3"            : "Storage",
    "efs"           : "Storage",
    "ebs"           : "Storage",
    # Database
    "database"      : "Database",
    "rds"           : "Database",
    "aurora"        : "Database",
    "dynamodb"      : "Database",
    "elasticache"   : "Database",
    "redshift"      : "Database",
    # Networking / CDN
    "network"       : "Networking and Content Delivery",
    "vpc"           : "Networking and Content Delivery",
    "elb"           : "Networking and Content Delivery",
    "cloudfront"    : "Networking and Content Delivery",
    "route53"       : "Networking and Content Delivery",
    "apigateway"    : "Networking and Content Delivery",
    # Application integration
    "sqs"           : "Application Integration",
    "sns"           : "Application Integration",
    "eventbridge"   : "Application Integration",
    "stepfunctions" : "Application Integration",
    # Security / IAM
    "security"      : "Security and Compliance",
    "iam"           : "Identity and Access Management",
    "kms"           : "Security and Compliance",
    # Monitoring / management
    "monitoring"    : "Monitoring and Logging",
    "cloudwatch"    : "Monitoring and Logging",
    "cloudformation": "Deployment and Automation",
    "devops"        : "Deployment and Automation",
    "cicd"          : "Deployment and Automation",
    # Cost / billing
    "cost"          : "Cost Optimization",
    "pricing"       : "Cost Optimization",
    "billing"       : "Billing and Pricing",
    # ML / AI / analytics
    "ml"            : "Machine Learning",
    "sagemaker"     : "Machine Learning",
    "bedrock"       : "Generative AI",
    "rekognit"      : "Machine Learning",
    "comprehend"    : "Machine Learning",
    "textract"      : "Machine Learning",
    "translat"      : "Machine Learning",
    "transcrib"     : "Machine Learning",
    "polly"         : "Machine Learning",
    "personaliz"    : "Machine Learning",
    "forecast"      : "Machine Learning",
    "analytics"     : "Analytics",
    "kinesis"       : "Analytics",
    "glue"          : "Analytics",
    "emr"           : "Analytics",
    "athena"        : "Analytics",
    "opensearch"    : "Analytics",
    "quicksight"    : "Analytics",
    "msk"           : "Analytics",
    "lakeformation" : "Analytics",
    # Developer / CI-CD tools
    "cognito"       : "Security and Compliance",
    "codedeploy"    : "Deployment and Automation",
    "codepipeline"  : "Deployment and Automation",
    "codebuild"     : "Deployment and Automation",
    "codecommit"    : "Deployment and Automation",
    "beanstalk"     : "Compute",
    "xray"          : "Monitoring and Logging",
    "secretsmanager": "Security and Compliance",
    # Security / governance
    "cloudtrail"    : "Security and Compliance",
    "guardduty"     : "Security and Compliance",
    "securityhub"   : "Security and Compliance",
    "inspector"     : "Security and Compliance",
    "macie"         : "Security and Compliance",
    "waf"           : "Security and Compliance",
    "shield"        : "Security and Compliance",
    "firewall"      : "Security and Compliance",
    "detective"     : "Security and Compliance",
    "certificatemanager": "Security and Compliance",
    "config"        : "Management and Governance",
    "systemsmanager": "Management and Governance",
    "organizations" : "Management and Governance",
    "controltower"  : "Management and Governance",
    "servicecatalog": "Management and Governance",
    "backup"        : "Storage",
    # Networking extras
    "directconnect" : "Networking and Content Delivery",
    "transitgateway": "Networking and Content Delivery",
    "vpn"           : "Networking and Content Delivery",
    "globalaccelerator": "Networking and Content Delivery",
    # Migration
    "migration"     : "Migration and Transfer",
}


def _extract_domain_hint(filename: str) -> str:
    """
    Infer a domain hint from a PDF filename using keyword matching.

    Normalises the filename to lowercase and strips hyphens/underscores
    before matching against _DOMAIN_KEYWORD_MAP.

    Returns an empty string if no keyword matches (graceful degradation).
    """
    normalised = filename.lower().replace("-", "").replace("_", "")
    for keyword, domain in _DOMAIN_KEYWORD_MAP.items():
        if keyword in normalised:
            return domain
    return ""


# ─────────────────────────────────────────────
# Document → chunks pipeline
# ─────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    cert_code: str,
) -> Tuple[List[DocumentChunk], int, int]:
    """
    Full processing pipeline for a single PDF file.

    Steps:
      1. Extract page text (pdfplumber → pypdf fallback).
      2. Clean each page's raw text.
      3. Concatenate all page texts into a single document string.
      4. Split into overlapping DocumentChunk objects.
      5. Generate deterministic chunk IDs.

    Parameters
    ──────────
    pdf_path
        Absolute path to the PDF.
    cert_code
        Certification this PDF belongs to (e.g. 'SAA-C03').

    Returns
    ───────
    (chunks, page_count, empty_page_count)
      chunks           : List of DocumentChunk objects ready for embedding.
      page_count       : Total pages in the PDF.
      empty_page_count : Pages that yielded no usable text after cleaning.
    """
    log.info(f"Processing: '{pdf_path.name}'")

    domain_hint = _extract_domain_hint(pdf_path.stem)
    if domain_hint:
        log.debug(f"Domain hint inferred: '{domain_hint}' from '{pdf_path.name}'")

    # ── Extract pages ──────────────────────────
    raw_pages = extract_pdf_pages(pdf_path)
    if not raw_pages:
        log.warning(f"No pages extracted from '{pdf_path.name}'. Skipping.")
        return [], 0, 0

    page_count       = len(raw_pages)
    empty_page_count = 0

    # ── Clean and concatenate pages ────────────
    # Build a per-page cleaned text list, tagging page boundaries
    # so chunk metadata can record approximate page numbers.
    cleaned_page_texts: List[Tuple[int, str]] = []
    for page_num, raw_text in raw_pages:
        cleaned = clean_page_text(raw_text)
        if not cleaned:
            empty_page_count += 1
            continue
        cleaned_page_texts.append((page_num, cleaned))

    if not cleaned_page_texts:
        log.warning(
            f"All {page_count} pages in '{pdf_path.name}' were empty "
            "after cleaning. Skipping."
        )
        return [], page_count, empty_page_count

    # ── Chunk each page independently ─────────
    # Chunking per-page (rather than the full concatenated document)
    # preserves page_number metadata accuracy in DocumentChunk and
    # prevents chunks from spanning across page boundaries in
    # ways that break semantic coherence.
    all_chunks: List[DocumentChunk] = []
    global_chunk_index = 0

    for page_num, page_text in cleaned_page_texts:
        page_chunk_texts = split_text_into_chunks(page_text)

        for local_idx, chunk_text in enumerate(page_chunk_texts):
            token_count = _approximate_token_count(chunk_text)
            chunk_id    = _generate_chunk_id(
                cert_code   = cert_code,
                source_file = pdf_path.name,
                page_number = page_num,
                chunk_index = global_chunk_index,
                text        = chunk_text,
            )

            chunk = DocumentChunk(
                chunk_id    = chunk_id,
                text        = chunk_text,
                source_file = pdf_path.name,
                page_number = page_num,
                chunk_index = global_chunk_index,
                cert_code   = cert_code,
                source_url  = _lookup_source_url(cert_code, pdf_path.name),
                domain_hint = domain_hint,
                token_count = token_count,
            )
            all_chunks.append(chunk)
            global_chunk_index += 1

    log.debug(
        f"'{pdf_path.name}': {page_count} pages → "
        f"{len(all_chunks)} chunks "
        f"({empty_page_count} empty page(s) skipped)."
    )

    return all_chunks, page_count, empty_page_count


# ─────────────────────────────────────────────
# ChromaDB operations
# ─────────────────────────────────────────────

def _lookup_source_url(cert_code: str, source_file: str) -> Optional[str]:
    """Return the public URL for a locally ingested source PDF, if known."""
    try:
        from download_docs import DOCUMENT_CATALOGUE
    except ImportError:
        return None

    for doc in DOCUMENT_CATALOGUE.get(cert_code, []):
        if doc.get("filename") == source_file:
            return doc.get("url")
    return None


def _get_or_create_collection(cert_code: str) -> Any:
    """
    Retrieve or create a ChromaDB persistent collection for a
    specific certification code.

    Collection naming convention:
        aws_{cert_code_lowercase_no_hyphens}
        e.g. SAA-C03 → aws_saa_c03

    Returns the ChromaDB Collection object.
    Raises RuntimeError if ChromaDB cannot be initialised.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise RuntimeError(
            "chromadb is not installed. Run: pip install chromadb"
        )

    try:
        client = chromadb.PersistentClient(
            path     = str(CHROMA_DIR),
            settings = Settings(anonymized_telemetry=False),
        )
        collection_name = (
            f"aws_{cert_code.lower().replace('-', '_')}"
        )
        collection = client.get_or_create_collection(
            name     = collection_name,
            metadata = {
                "cert_code"  : cert_code,
                "description": (
                    f"AWS {cert_code} exam documentation embeddings"
                ),
            },
            # Default ef_search (100) is tuned for small collections. Once
            # a handful of oversized source PDFs (e.g. the ~1,000-page
            # Well-Architected Framework) dominate a corpus, a shallow
            # HNSW search can converge entirely inside their neighborhood
            # and never surface chunks from smaller, more specific
            # documents — starving generator.py's retrieve_context() of
            # anything to diversify across. A deeper search costs a bit
            # of query latency in exchange for materially better recall.
            configuration = {"hnsw": {"ef_search": 400}},
        )
        # Existing collections keep whatever ef_search they were created
        # with — bump it in place so already-ingested certs benefit too
        # without a full re-ingest.
        current_ef = (
            collection.configuration_json.get("hnsw", {}).get("ef_search")
            if collection.configuration_json else None
        )
        if current_ef is not None and current_ef < 400:
            collection.modify(configuration={"hnsw": {"ef_search": 400}})
            log.debug(
                f"Raised '{collection_name}' ef_search {current_ef} → 400."
            )
        log.debug(
            f"ChromaDB collection '{collection_name}' ready "
            f"(existing docs: {collection.count()})."
        )
        return collection

    except Exception as chroma_err:
        raise RuntimeError(
            f"ChromaDB initialisation failed: {chroma_err}"
        ) from chroma_err


def _batch_upsert_chunks(
    collection:      Any,
    chunks:          List[DocumentChunk],
    embeddings:      List[List[float]],
    batch_size:      int = 100,
) -> Tuple[int, int]:
    """
    Upsert chunks and their embeddings into ChromaDB in batches.

    Uses upsert (not add) so re-running ingestion on the same
    documents is idempotent — existing chunks are updated only
    if their content hash (and therefore chunk_id) changed.

    Parameters
    ──────────
    collection
        ChromaDB Collection to upsert into.
    chunks
        List of DocumentChunk objects.
    embeddings
        Parallel list of embedding vectors (same length as chunks).
    batch_size
        Number of chunks per ChromaDB upsert call.
        Keeping this ≤ 100 prevents memory spikes during large ingestions.

    Returns
    ───────
    (upserted_count, error_count)
    """
    upserted = 0
    errors   = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks     = chunks[batch_start : batch_start + batch_size]
        batch_embeddings = embeddings[batch_start : batch_start + batch_size]

        try:
            collection.upsert(
                ids        = [c.chunk_id for c in batch_chunks],
                documents  = [c.to_chroma_document() for c in batch_chunks],
                metadatas  = [c.to_chroma_metadata() for c in batch_chunks],
                embeddings = batch_embeddings,
            )
            upserted += len(batch_chunks)
            log.debug(
                f"Upserted batch {batch_start // batch_size + 1}: "
                f"{len(batch_chunks)} chunks."
            )
        except Exception as upsert_err:
            errors += len(batch_chunks)
            log.error(
                f"ChromaDB upsert failed for batch starting at "
                f"index {batch_start}: {upsert_err}"
            )

    return upserted, errors


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────

def _load_embedding_model() -> Any:
    """
    Load the sentence-transformer embedding model onto CPU.

    Using device='cpu' is mandatory — the GTX 1070's 8 GB VRAM
    must remain fully available for the Ollama LLM.
    The all-MiniLM-L6-v2 model (~23 MB) runs efficiently on CPU
    and produces 384-dimensional embeddings suitable for ChromaDB.

    Returns
    ───────
    A loaded SentenceTransformer model instance.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    log.info(
        f"Loading embedding model '{EMBEDDING_CONFIG.model_name}' "
        "on CPU (VRAM preserved for Ollama)…"
    )
    model = SentenceTransformer(
        EMBEDDING_CONFIG.model_name,
        device="cpu",
    )
    log.info("Embedding model loaded successfully.")
    return model


def embed_chunks(
    chunks: List[DocumentChunk],
    embedding_model: Any,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Generate embedding vectors for a list of DocumentChunk objects.

    Processes chunks in batches to keep CPU memory usage bounded
    during large ingestion runs (some AWS whitepapers are 200+ pages).

    Parameters
    ──────────
    chunks
        List of DocumentChunk objects to embed.
    embedding_model
        Loaded SentenceTransformer instance.
    batch_size
        Number of chunks to embed per model call.
        64 is a good balance for CPU-side all-MiniLM-L6-v2.

    Returns
    ───────
    List of float vectors parallel to the input chunks list.
    """
    if not chunks:
        return []

    texts      = [c.text for c in chunks]
    embeddings: List[List[float]] = []

    log.info(
        f"Embedding {len(chunks)} chunks in batches of {batch_size}…"
    )

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        try:
            batch_embeddings = embedding_model.encode(
                batch_texts,
                show_progress_bar    = False,
                normalize_embeddings = True,   # Cosine similarity optimised
            )
            embeddings.extend(batch_embeddings.tolist())
            log.debug(
                f"Embedded batch {batch_start // batch_size + 1}: "
                f"{len(batch_texts)} chunks."
            )
        except Exception as embed_err:
            log.error(
                f"Embedding failed for batch at index {batch_start}: "
                f"{embed_err}. Inserting zero-vectors as placeholders."
            )
            # Insert zero-vectors so chunk indices stay aligned
            dim = 384  # all-MiniLM-L6-v2 output dimension
            embeddings.extend([[0.0] * dim] * len(batch_texts))

    return embeddings


# ─────────────────────────────────────────────
# Directory scanning
# ─────────────────────────────────────────────

def scan_pdf_directory(source_path: Path) -> List[Path]:
    """
    Recursively scan a directory for PDF files.

    Skips hidden files (starting with '.') and files smaller than
    1 KB (likely placeholder or corrupt files).

    Parameters
    ──────────
    source_path
        Directory to scan (absolute or relative Path).

    Returns
    ───────
    Sorted list of PDF Path objects found under source_path.
    Raises FileNotFoundError if source_path does not exist.
    """
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source directory '{source_path}' does not exist. "
            "Create the directory and place your PDF files inside it."
        )

    if not source_path.is_dir():
        raise NotADirectoryError(
            f"'{source_path}' is not a directory."
        )

    pdf_files: List[Path] = []

    all_source_files = sorted(
        list(source_path.rglob("*.pdf")) + list(source_path.rglob("*.html"))
    )

    for pdf_path in all_source_files:
        # Skip hidden files
        if pdf_path.name.startswith("."):
            log.debug(f"Skipping hidden file: '{pdf_path.name}'")
            continue

        # Skip suspiciously small files
        try:
            size_kb = pdf_path.stat().st_size / 1024
            if size_kb < 1.0:
                log.warning(
                    f"Skipping '{pdf_path.name}' — file too small "
                    f"({size_kb:.1f} KB). Possibly corrupt or empty."
                )
                continue
        except OSError:
            log.warning(f"Cannot stat '{pdf_path.name}'. Skipping.")
            continue

        pdf_files.append(pdf_path)

    log.info(
        f"Found {len(pdf_files)} PDF file(s) in '{source_path}'."
    )

    return pdf_files


# ─────────────────────────────────────────────
# CLI display helpers
# ─────────────────────────────
# ─────────────────────────────────────────────
# CLI display helpers
# ─────────────────────────────────────────────

def _print_ingestion_report(report: IngestionReport) -> None:
    """
    Render a Rich table and summary panel for the ingestion report.
    Called at the end of run_ingestion_pipeline() for CLI feedback.
    """
    # ── Per-file statistics table ──────────────
    table = Table(
        title       = f"Ingestion Results — {report.cert_code}",
        show_header = True,
        header_style= "bold magenta",
        show_lines  = True,
    )

    table.add_column("File",         style="cyan",   width=40)
    table.add_column("Pages",        style="white",  width=8,  justify="right")
    table.add_column("Chunks",       style="green",  width=8,  justify="right")
    table.add_column("Upserted",     style="green",  width=10, justify="right")
    table.add_column("Skipped",      style="yellow", width=10, justify="right")
    table.add_column("Domain Hint",  style="blue",   width=30)
    table.add_column("Status",       style="white",  width=10)

    for stat in report.per_file_stats:
        status_str = (
            "[green]OK[/green]"
            if stat.get("status") == "ok"
            else "[red]FAILED[/red]"
        )
        table.add_row(
            stat.get("filename",    "")[:38],
            str(stat.get("pages",   0)),
            str(stat.get("chunks",  0)),
            str(stat.get("upserted",0)),
            str(stat.get("skipped", 0)),
            stat.get("domain_hint", "")[:28],
            status_str,
        )

    console.print(table)

    # ── Summary panel ──────────────────────────
    failed_str = (
        ", ".join(report.pdfs_failed)
        if report.pdfs_failed
        else "None"
    )

    summary = (
        f"[bold]Certification   :[/bold]  {report.cert_code}\n"
        f"[bold]Source directory:[/bold]  {report.source_directory}\n"
        f"[bold]PDFs found      :[/bold]  {report.pdfs_found}\n"
        f"[bold]PDFs processed  :[/bold]  [green]{report.pdfs_processed}[/green]\n"
        f"[bold]PDFs failed     :[/bold]  [red]{len(report.pdfs_failed)}[/red]"
        f"  ({failed_str})\n"
        f"[bold]Total pages     :[/bold]  {report.total_pages}\n"
        f"[bold]Total chunks    :[/bold]  {report.total_chunks}\n"
        f"[bold]Chunks upserted :[/bold]  [green]{report.chunks_upserted}[/green]\n"
        f"[bold]Chunks skipped  :[/bold]  [yellow]{report.chunks_skipped}[/yellow]\n"
        f"[bold]Collection total:[/bold]  {report.collection_total} "
        f"(all docs in '{report.collection_name}')\n"
        f"[bold]Duration        :[/bold]  {report.duration_seconds:.1f}s"
    )

    console.print(
        Panel(
            summary,
            title        = "[bold blue]Ingestion Complete[/bold blue]",
            border_style = "blue",
            expand       = False,
        )
    )


# ─────────────────────────────────────────────
# Existing chunk ID lookup
# ─────────────────────────────────────────────

def _get_existing_chunk_ids(collection: Any) -> set:
    """
    Retrieve the set of all chunk IDs already stored in a ChromaDB
    collection.

    Used to implement skip logic: chunks whose IDs already exist in
    ChromaDB are not re-embedded or re-upserted, saving significant
    CPU time on repeat ingestion runs.

    Returns an empty set if the collection is empty or the query fails
    (graceful degradation — all chunks will be upserted).
    """
    try:
        existing = collection.get(include=[])
        ids = set(existing.get("ids", []))
        log.debug(
            f"Found {len(ids)} existing chunk ID(s) in collection."
        )
        return ids
    except Exception as fetch_err:
        log.warning(
            f"Could not fetch existing chunk IDs from ChromaDB: "
            f"{fetch_err}. All chunks will be upserted."
        )
        return set()


def _filter_new_chunks(
    chunks: List[DocumentChunk],
    existing_ids: set,
) -> Tuple[List[DocumentChunk], int]:
    """
    Partition chunks into new (not yet in ChromaDB) and skipped
    (already present with the same content hash).

    Parameters
    ──────────
    chunks
        All chunks produced by process_pdf() for a document.
    existing_ids
        Set of chunk IDs already stored in the ChromaDB collection.

    Returns
    ───────
    (new_chunks, skipped_count)
        new_chunks    : Chunks that need embedding and upsert.
        skipped_count : Number of chunks already in ChromaDB.
    """
    new_chunks: List[DocumentChunk] = []
    skipped = 0

    for chunk in chunks:
        if chunk.chunk_id in existing_ids:
            skipped += 1
        else:
            new_chunks.append(chunk)

    return new_chunks, skipped


# ─────────────────────────────────────────────
# Top-level ingestion pipeline
# ─────────────────────────────────────────────

def run_ingestion_pipeline(
    cert_code:   str,
    source_path: Path,
) -> IngestionReport:
    """
    Top-level entry point for the document ingestion pipeline.

    Orchestrates the complete flow:
      1. Validate cert_code against the registry.
      2. Scan source_path for PDF files.
      3. Load the sentence-transformer embedding model (CPU).
      4. Get or create the ChromaDB collection for cert_code.
      5. Fetch existing chunk IDs for skip detection.
      6. For each PDF:
           a. Extract and clean page text.
           b. Split into overlapping DocumentChunk objects.
           c. Filter out already-ingested chunks.
           d. Embed new chunks via sentence-transformers.
           e. Upsert embeddings into ChromaDB.
           f. Record per-file statistics.
      7. Print the ingestion report to the console.
      8. Return the IngestionReport for programmatic use.

    Parameters
    ──────────
    cert_code
        AWS certification code (e.g. 'SAA-C03').
        Must exist in registry.json.
    source_path
        Directory containing PDF files to ingest.
        Scanned recursively for *.pdf files.

    Returns
    ───────
    IngestionReport dataclass with full statistics.

    Raises
    ──────
    FileNotFoundError
        If source_path does not exist.
    RuntimeError
        If the registry, ChromaDB, or embedding model cannot
        be initialised.
    ValueError
        If cert_code is not found in the registry.
    """
    pipeline_start = time.monotonic()

    log.info(
        f"Starting ingestion pipeline for '{cert_code}' "
        f"from '{source_path}'…"
    )

    # ── Step 1: Validate cert_code ─────────────
    log.info("Validating certification code against registry…")
    try:
        registry      = load_registry(REGISTRY_PATH)
        cert_metadata = get_cert_metadata(cert_code, registry)
        log.info(
            f"Certification validated: {cert_metadata['name']} "
            f"({cert_metadata['tier']} tier)."
        )
    except (FileNotFoundError, ValueError) as reg_err:
        raise RuntimeError(
            f"Registry validation failed: {reg_err}"
        ) from reg_err

    # ── Step 2: Scan for PDFs ──────────────────
    log.info(f"Scanning '{source_path}' for PDF files…")
    try:
        pdf_files = scan_pdf_directory(source_path)
    except (FileNotFoundError, NotADirectoryError) as scan_err:
        raise RuntimeError(
            f"Directory scan failed: {scan_err}"
        ) from scan_err

    if not pdf_files:
        log.warning(
            f"No PDF files found in '{source_path}'. "
            "Ingestion complete with nothing to process."
        )
        return IngestionReport(
            cert_code        = cert_code,
            source_directory = str(source_path),
            pdfs_found       = 0,
        )

    # ── Step 3: Load embedding model ──────────
    try:
        embedding_model = _load_embedding_model()
    except RuntimeError as emb_err:
        raise RuntimeError(
            f"Embedding model load failed: {emb_err}"
        ) from emb_err

    # ── Step 4: Get or create ChromaDB collection ──
    log.info(
        f"Initialising ChromaDB collection for '{cert_code}'…"
    )
    try:
        collection      = _get_or_create_collection(cert_code)
        collection_name = (
            f"aws_{cert_code.lower().replace('-', '_')}"
        )
    except RuntimeError as chroma_err:
        raise RuntimeError(
            f"ChromaDB initialisation failed: {chroma_err}"
        ) from chroma_err

    # ── Step 5: Fetch existing chunk IDs ──────
    existing_ids = _get_existing_chunk_ids(collection)

    # ── Initialise report ─────────────────────
    report = IngestionReport(
        cert_code        = cert_code,
        source_directory = str(source_path),
        pdfs_found       = len(pdf_files),
        collection_name  = collection_name,
    )

    # ── Step 6: Process each PDF ───────────────
    log.info(
        f"Processing {len(pdf_files)} PDF file(s)…"
    )

    with make_progress_bar() as progress:
        pdf_task = progress.add_task(
            f"[cyan]Ingesting {cert_code} documents…",
            total=len(pdf_files),
        )

        for pdf_path in pdf_files:
            file_stat: Dict[str, Any] = {
                "filename"   : pdf_path.name,
                "pages"      : 0,
                "chunks"     : 0,
                "upserted"   : 0,
                "skipped"    : 0,
                "domain_hint": "",
                "status"     : "ok",
            }

            try:
                # ── 6a: Extract and chunk ──────
                chunks, page_count, empty_pages = process_pdf(
                    pdf_path  = pdf_path,
                    cert_code = cert_code,
                )

                file_stat["pages"] = page_count
                report.total_pages += page_count

                if not chunks:
                    log.warning(
                        f"'{pdf_path.name}' produced no chunks. "
                        "Skipping embedding step."
                    )
                    file_stat["status"] = "ok"
                    report.per_file_stats.append(file_stat)
                    report.pdfs_processed += 1
                    progress.advance(pdf_task)
                    continue

                file_stat["chunks"]      = len(chunks)
                file_stat["domain_hint"] = chunks[0].domain_hint if chunks else ""
                report.total_chunks     += len(chunks)

                # ── 6b: Filter already-ingested chunks ──
                new_chunks, skipped_count = _filter_new_chunks(
                    chunks       = chunks,
                    existing_ids = existing_ids,
                )

                file_stat["skipped"]  = skipped_count
                report.chunks_skipped += skipped_count

                if skipped_count > 0:
                    log.info(
                        f"'{pdf_path.name}': {skipped_count} chunk(s) "
                        "already in ChromaDB — skipped."
                    )

                if not new_chunks:
                    log.info(
                        f"'{pdf_path.name}': all chunks already ingested. "
                        "Nothing to upsert."
                    )
                    report.pdfs_processed += 1
                    report.per_file_stats.append(file_stat)
                    progress.advance(pdf_task)
                    continue

                # ── 6c: Embed new chunks ───────
                log.info(
                    f"'{pdf_path.name}': embedding "
                    f"{len(new_chunks)} new chunk(s)…"
                )
                embeddings = embed_chunks(
                    chunks          = new_chunks,
                    embedding_model = embedding_model,
                )

                # ── 6d: Upsert into ChromaDB ───
                upserted_count, error_count = _batch_upsert_chunks(
                    collection = collection,
                    chunks     = new_chunks,
                    embeddings = embeddings,
                )

                file_stat["upserted"]  = upserted_count
                report.chunks_upserted += upserted_count

                # Add newly upserted IDs to the existing set so
                # duplicate detection works within the same run
                # (in case the same chunk appears in multiple PDFs)
                for chunk in new_chunks:
                    existing_ids.add(chunk.chunk_id)

                if error_count > 0:
                    log.warning(
                        f"'{pdf_path.name}': {error_count} chunk(s) "
                        "failed to upsert into ChromaDB."
                    )

                report.pdfs_processed += 1

            except Exception as pdf_err:
                log.error(
                    f"Failed to process '{pdf_path.name}': {pdf_err}"
                )
                file_stat["status"] = "failed"
                report.pdfs_failed.append(pdf_path.name)

            finally:
                report.per_file_stats.append(file_stat)
                progress.advance(pdf_task)
                progress.update(
                    pdf_task,
                    description=(
                        f"[cyan]Ingesting {cert_code} documents… "
                        f"[{report.pdfs_processed}/{report.pdfs_found} done, "
                        f"{report.chunks_upserted} chunks upserted]"
                    ),
                )

    # ── Step 7: Final collection count ────────
    try:
        report.collection_total = collection.count()
    except Exception:
        report.collection_total = -1

    report.duration_seconds = round(
        time.monotonic() - pipeline_start, 2
    )

    # ── Step 8: Print report ───────────────────
    _print_ingestion_report(report)

    log.info(
        f"Ingestion pipeline complete for '{cert_code}' in "
        f"{report.duration_seconds:.1f}s."
    )

    return report


# ─────────────────────────────────────────────
# Collection management utilities
# ─────────────────────────────────────────────

def get_collection_info(cert_code: str) -> Dict[str, Any]:
    """
    Return summary information about an existing ChromaDB collection.

    Useful for the CLI 'status' command to show what has already
    been ingested without running a full pipeline.

    Returns
    ───────
    Dict with keys: collection_name, cert_code, total_chunks,
    source_files, domain_hints.

    Raises RuntimeError if the collection does not exist.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path     = str(CHROMA_DIR),
            settings = Settings(anonymized_telemetry=False),
        )
        collection_name = (
            f"aws_{cert_code.lower().replace('-', '_')}"
        )
        existing = [c.name for c in client.list_collections()]

        if collection_name not in existing:
            raise RuntimeError(
                f"Collection '{collection_name}' does not exist. "
                f"Run 'python main.py ingest --cert {cert_code}' first."
            )

        collection = client.get_collection(collection_name)
        total      = collection.count()

        # Sample metadata to extract source file and domain info
        source_files: set  = set()
        domain_hints: set  = set()

        if total > 0:
            sample_size = min(total, 500)
            try:
                sample = collection.get(
                    limit   = sample_size,
                    include = ["metadatas"],
                )
                for meta in sample.get("metadatas", []):
                    if meta.get("source_file"):
                        source_files.add(meta["source_file"])
                    if meta.get("domain_hint"):
                        domain_hints.add(meta["domain_hint"])
            except Exception as sample_err:
                log.debug(
                    f"Could not sample collection metadata: {sample_err}"
                )

        return {
            "collection_name": collection_name,
            "cert_code"      : cert_code,
            "total_chunks"   : total,
            "source_files"   : sorted(source_files),
            "domain_hints"   : sorted(domain_hints),
        }

    except RuntimeError:
        raise
    except Exception as info_err:
        raise RuntimeError(
            f"Could not retrieve collection info: {info_err}"
        ) from info_err


def delete_collection(cert_code: str) -> bool:
    """
    Delete the ChromaDB collection for a certification code.

    Used by the CLI 'reset' command to wipe and re-ingest a
    certification's document store from scratch.

    Returns True if deleted successfully, False if not found.
    Raises RuntimeError on unexpected ChromaDB errors.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path     = str(CHROMA_DIR),
            settings = Settings(anonymized_telemetry=False),
        )
        collection_name = (
            f"aws_{cert_code.lower().replace('-', '_')}"
        )
        existing = [c.name for c in client.list_collections()]

        if collection_name not in existing:
            log.warning(
                f"Collection '{collection_name}' does not exist. "
                "Nothing to delete."
            )
            return False

        client.delete_collection(collection_name)
        log.info(
            f"Collection '{collection_name}' deleted successfully."
        )
        return True

    except Exception as del_err:
        raise RuntimeError(
            f"Failed to delete collection for '{cert_code}': {del_err}"
        ) from del_err
