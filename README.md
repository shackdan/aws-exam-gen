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
    "AIF-C01", "CLF-C02", "SOA-C03", "DEA-C01", "DVA-C02", "MLA-C01", "SAA-C03", "SAP-C02", "DOP-C02", "AIP-C01", "SCS-C03", "ANS-C01"
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

`main.py generate` retrieves relevant chunks from ChromaDB, drafts questions with the local LLM, and runs each one through an AI-reviewer loop (up to 2 revisions) before writing only **Approved** questions to `./output/`. Each run is capped at 200 questions (`--count`), so to build a full bank per certification, loop several runs of 100–200 questions each — every run writes its own timestamped file (e.g. `output/SAA-C03_100q_20260814T120000.json`) and is checked for duplicates against every prior approved export for that cert, so looping more runs adds coverage rather than repeats.

### How many questions is enough?

Matching the real exam's question count (see `total_questions` in [registry.json](registry.json)) only gets you one mock exam — it doesn't tell you whether a user is actually weak on a specific domain or just got unlucky on one attempt. As a rule of thumb, a bank needs to be large enough for (a) several non-repeating mock exams and (b) enough questions per domain that a low score there is a real signal, not noise. That works out to roughly 4–7x the real exam length, more for certs with more domains:

| Code | Certification | Real exam Qs | Domains | Recommended bank |
| --- | --- | --- | --- | --- |
| CLF-C02 | AWS Certified Cloud Practitioner | 65 | 4 | 300 |
| AIF-C01 | AWS Certified AI Practitioner | 85 | 5 | 350 |
| SAA-C03 | AWS Certified Solutions Architect Associate | 65 | 4 | 350 |
| DVA-C02 | AWS Certified Developer Associate | 65 | 4 | 350 |
| SOA-C03 | AWS Certified CloudOps Engineer Associate | 65 | 5 | 400 |
| DEA-C01 | AWS Certified Data Engineer Associate | 65 | 4 | 350 |
| MLA-C01 | AWS Certified Machine Learning Engineer Associate | 65 | 4 | 350 |
| SAP-C02 | AWS Certified Solutions Architect Professional | 75 | 4 | 500 |
| DOP-C02 | AWS Certified DevOps Engineer Professional | 75 | 6 | 500 |
| AIP-C01 | AWS Certified Generative AI Developer Professional | 75 | 5 | 500 |
| SCS-C03 | AWS Certified Security Specialty | 65 | 6 | 400 |
| ANS-C01 | AWS Certified Advanced Networking Specialty | 65 | 4 | 350 |

Professional-tier certs sit at the top of the range — more domains and tasks are tested in more depth, so more questions are needed before a per-domain score is trustworthy. A bank this size also depends on the ingested source documents actually covering the cert's full in-scope service list, not just a handful of frequently-retrieved services — see [Adding a new certification](#adding-a-new-certification) and keep `DOCUMENT_CATALOGUE` broad, not just large.

Generate the recommended 500 for SAP-C02 (5 x 100 — see the table above for other certs' targets):

**PowerShell**

```powershell
for ($i = 1; $i -le 5; $i++) {
    python main.py generate --cert SAP-C02 --count 100 --output json
}
```

**Bash**

```bash
for i in $(seq 1 5); do
    python main.py generate --cert SAP-C02 --count 100 --output json
done
```

Generate each certification's recommended bank size from the table above (runs in batches of 100, rounded up):

**PowerShell**

```powershell
$certBatches = @{
    "CLF-C02" = 3; "AIF-C01" = 4; "SAA-C03" = 4; "DVA-C02" = 4; "SOA-C03" = 4
    "DEA-C01" = 4; "MLA-C01" = 4; "SAP-C02" = 5; "DOP-C02" = 5; "AIP-C01" = 5
    "SCS-C03" = 4; "ANS-C01" = 4
}

foreach ($cert in $certBatches.Keys) {
    for ($i = 1; $i -le $certBatches[$cert]; $i++) {
        python main.py generate --cert $cert --count 200 --output json
    }
}
```

**Bash**

```bash
declare -A cert_batches=(
    [CLF-C02]=3 [AIF-C01]=4 [SAA-C03]=4 [DVA-C02]=4 [SOA-C03]=4
    [DEA-C01]=4 [MLA-C01]=4 [SAP-C02]=5 [DOP-C02]=5 [AIP-C01]=5
    [SCS-C03]=4 [ANS-C01]=4
)

for cert in "${!cert_batches[@]}"; do
    for i in $(seq 1 "${cert_batches[$cert]}"); do
        python main.py generate --cert "$cert" --count 100 --output json
    done
done
```

Supported `--output` formats: `json`, `csv`, `moodle_xml`. Other useful flags: `--domain "<name>"` (restrict to one exam domain), `--difficulty <tier>`, `--show-questions` (print results to the console), `--out-file <path>` (explicit output path instead of a timestamped default).

> Since the LLM reviewer rejects some drafts, each 100-question run typically yields somewhat fewer than 100 *approved* questions. Run extra batches per certification if you need to top up to the recommended bank size.

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
