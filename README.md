# AWS Exam Gen

A local, offline pipeline for generating AWS certification practice exams. It downloads official AWS source documentation (exam guides, whitepapers, service FAQs), ingests it into a local ChromaDB vector store, and uses a local Ollama LLM with a RAG (retrieval-augmented generation) + AI-reviewer loop to draft, validate, and export multiple-choice questions.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- An Ollama model pulled that matches `OLLAMA_MODEL` in [config.py](config.py) (defaults to `qwen2.5:7b-instruct`)

```powershell
ollama pull qwen2.5:7b-instruct
ollama serve
```

## Setup

**PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Bash**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tuning for more VRAM

The defaults in [config.py](config.py) are sized for the dev box's Nvidia GTX 1070 (8 GB VRAM): a 4096-token context window and 4 RAG chunks retrieved per query. If you have a card with more VRAM to spare, raise these via environment variables instead of editing the code:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Ollama model to use — a larger model (e.g. `qwen2.5:14b-instruct`) can improve question quality |
| `OLLAMA_NUM_CTX` | `4096` | Context window size (tokens) passed to Ollama |
| `RAG_TOP_K` | `4` | Number of RAG chunks retrieved per question |

The prompt's context-truncation limit scales automatically with `OLLAMA_NUM_CTX`, so raising it also lets more retrieved context reach the model.

**PowerShell**

```powershell
$env:OLLAMA_MODEL = "qwen2.5:14b-instruct"
$env:OLLAMA_NUM_CTX = "16384"
$env:RAG_TOP_K = "8"
```

**Bash**

```bash
export OLLAMA_MODEL="qwen2.5:14b-instruct"
export OLLAMA_NUM_CTX="16384"
export RAG_TOP_K="8"
```

## Supported certifications

Run `python main.py list` to see all certifications known to [registry.json](registry.json), their tier, domains, and exam question counts.

| Code | Certification |
| --- | --- |
| CLF-C02 | AWS Certified Cloud Practitioner |
| AIF-C01 | AWS Certified AI Practitioner |
| SAA-C03 | AWS Certified Solutions Architect Associate |
| DVA-C02 | AWS Certified Developer Associate |
| SOA-C03 | AWS Certified CloudOps Engineer Associate |
| DEA-C01 | AWS Certified Data Engineer Associate |
| MLA-C01 | AWS Certified Machine Learning Engineer Associate |
| SAP-C02 | AWS Certified Solutions Architect Professional |
| DOP-C02 | AWS Certified DevOps Engineer Professional |
| AIP-C01 | AWS Certified Generative AI Developer Professional |
| SCS-C03 | AWS Certified Security Specialty |
| ANS-C01 | AWS Certified Advanced Networking Specialty |

## 1. Download source documents

`download_docs.py` fetches the official PDFs/HTML FAQs for each certification into `./data/<CERT_CODE>/`.

Download for a single certification:

```powershell
python download_docs.py --cert SAA-C03
```

Download for **every** certification (recommended for a first-time setup):

```powershell
python download_docs.py --cert all
```

Other useful flags: `--dry-run` (preview without writing files), `--force` (re-download existing files), `--required-only` (skip optional documents), `--list` (show the document catalogue per cert).

## 2. Ingest documents into ChromaDB

`main.py ingest` extracts text from the downloaded PDFs/HTML, chunks it, embeds it with `sentence-transformers`, and upserts it into a local ChromaDB collection (`./chroma_store/`).

Ingest a single certification:

```powershell
python main.py ingest --cert SAA-C03
```

Ingest **every** certification in the registry:

**PowerShell**

```powershell
$certs = @(
    "CLF-C02", "AIF-C01", "SAA-C03", "DVA-C02", "SOA-C03",
    "DEA-C01", "MLA-C01", "SAP-C02", "DOP-C02", "AIP-C01", "SCS-C03", "ANS-C01"
)

foreach ($cert in $certs) {
    python main.py ingest --cert $cert
}
```

**Bash**

```bash
certs=(CLF-C02 AIF-C01 SAA-C03 DVA-C02 SOA-C03 DEA-C01 MLA-C01 SAP-C02 DOP-C02 AIP-C01 SCS-C03 ANS-C01)

for cert in "${certs[@]}"; do
    python main.py ingest --cert "$cert"
done
```

Check what was ingested at any time with `python main.py status --cert <CODE>`. Use `python main.py reset --cert <CODE>` to delete a collection and start over.

## 3. Generate questions

`main.py generate` retrieves relevant chunks from ChromaDB, drafts questions with the local LLM, and runs each one through an AI-reviewer loop (up to 2 revisions) before writing only **Approved** questions to `./output/`. Each run is capped at 200 questions (`--count`), so to build a 500-question bank per certification, loop 5 runs of 100 questions each — every run writes its own timestamped file (e.g. `output/SAA-C03_100q_20260814T120000.json`).

Generate 500 questions (5 x 100) for a single certification:

**PowerShell**

```powershell
for ($i = 1; $i -le 5; $i++) {
    python main.py generate --cert SAA-C03 --count 100 --output json
}
```

**Bash**

```bash
for i in $(seq 1 5); do
    python main.py generate --cert SAA-C03 --count 100 --output json
done
```

Generate 500 questions (5 x 100) for **every** certification:

**PowerShell**

```powershell
$certs = @(
    "CLF-C02", "AIF-C01", "SAA-C03", "DVA-C02", "SOA-C03",
    "DEA-C01", "MLA-C01", "SAP-C02", "DOP-C02", "AIP-C01", "SCS-C03", "ANS-C01"
)

foreach ($cert in $certs) {
    for ($i = 1; $i -le 5; $i++) {
        python main.py generate --cert $cert --count 100 --output json
    }
}
```

**Bash**

```bash
certs=(CLF-C02 AIF-C01 SAA-C03 DVA-C02 SOA-C03 DEA-C01 MLA-C01 SAP-C02 DOP-C02 AIP-C01 SCS-C03 ANS-C01)

for cert in "${certs[@]}"; do
    for i in $(seq 1 5); do
        python main.py generate --cert "$cert" --count 100 --output json
    done
done
```

Supported `--output` formats: `json`, `csv`, `moodle_xml`. Other useful flags: `--domain "<name>"` (restrict to one exam domain), `--difficulty <tier>`, `--show-questions` (print results to the console), `--out-file <path>` (explicit output path instead of a timestamped default).

> Since the LLM reviewer rejects some drafts, each 100-question run typically yields somewhat fewer than 100 *approved* questions. Run extra batches per certification if you need to top up to a full 500.

## Command reference

| Command | Purpose |
| --- | --- |
| `python download_docs.py --cert <CODE\|all>` | Download official source PDFs/FAQs |
| `python main.py list` | List supported certifications |
| `python main.py ingest --cert <CODE>` | Ingest a certification's documents into ChromaDB |
| `python main.py status --cert <CODE>` | Show ChromaDB collection info for a certification |
| `python main.py reset --cert <CODE>` | Delete a certification's ChromaDB collection |
| `python main.py generate --cert <CODE> --count <N>` | Generate and validate MCQ questions |

Run any command with `--help` for the full list of options.

## Troubleshooting

- **`Ollama not reachable`** — start Ollama with `ollama serve` and confirm the model from [config.py](config.py) is pulled.
- **`No questions were approved`** — confirm the certification was ingested (`python main.py status --cert <CODE>`), try a smaller `--count`, or check logs with `--log-level DEBUG`.
- **`Source directory not found`** during ingest — run the download step first, or pass `--path` to point at your own PDF directory.

## Contributing

### Project structure

| File | Responsibility |
| --- | --- |
| [main.py](main.py) | CLI entrypoint (`click`) — wires up `ingest`, `generate`, `status`, `reset`, `list` |
| [download_docs.py](download_docs.py) | `DOCUMENT_CATALOGUE` of source URLs per cert + downloader CLI |
| [ingest.py](ingest.py) | PDF/HTML parsing, chunking, embedding, ChromaDB upsert |
| [generator.py](generator.py) | RAG retrieval + LLM question drafting pipeline |
| [reviewer.py](reviewer.py) | AI reviewer persona that approves/rejects/revises drafted questions |
| [schemas.py](schemas.py) | Pydantic v2 models (`ExamQuestion`, `ValidationCritique`, `GenerationRequest`, ...) |
| [config.py](config.py) | Central config (paths, Ollama settings, embedding/generation tuning) |
| [utils.py](utils.py) | Logging, JSON repair, export formatting (CSV/Moodle XML), console output |
| [registry.json](registry.json) | Certification metadata: name, tier, domains, domain weights, exam question counts |

### Adding a new certification

1. Add an entry to `registry.json` with `name`, `tier`, `domains`, `domain_weights` (must sum to ~1.0), `total_questions`, and `passing_score`.
2. Add a matching entry to `DOCUMENT_CATALOGUE` in [download_docs.py](download_docs.py) with the official AWS source URLs (exam guide, whitepapers, FAQs) for that cert.
3. Verify end-to-end: `download_docs.py --cert <CODE>` → `main.py ingest --cert <CODE>` → `main.py generate --cert <CODE> --count 5 --show-questions`.

### Making changes

There is no automated test suite or linter configured yet — verify changes manually by running the affected command(s) end-to-end (see above) with `--log-level DEBUG` and confirming the console output/exported file look correct. Match the existing style: type hints, `from __future__ import annotations`, Pydantic models for data passed between modules, and `rich` for console output.

Keep pull requests focused on one change (e.g. one new certification, one bug fix) and describe what you tested it with (which cert, model, and command).

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Copyright (c) 2026 Dan Newton

This project is licensed under the **Creative Commons Attribution-NonCommercial 
4.0 International License**.

- ✅ You **can** share and adapt this code
- ✅ You **must** give credit to the original author
- ❌ You **cannot** use this for commercial purposes

See the [LICENSE](LICENSE) file for details or visit  
https://creativecommons.org/licenses/by-nc/4.0/
