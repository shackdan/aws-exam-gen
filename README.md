# AWS Exam Gen

A local, offline pipeline for generating AWS certification practice exams. It downloads official AWS source documentation (exam guides, whitepapers, service FAQs), ingests it into a local ChromaDB vector store, and uses a local LLM (Ollama, or MLX-LM on Apple Silicon Macs) with a RAG (retrieval-augmented generation) + AI-reviewer loop to draft, validate, and export multiple-choice questions.

## Prerequisites

- Python 3.10+
- An LLM backend — either:
  - [Ollama](https://ollama.com/) installed and running locally (default), **or**
  - MLX-LM on an Apple Silicon Mac — see [Using MLX-LM on a MacBook](#using-mlx-lm-on-a-macbook-apple-silicon)
- An Ollama model pulled that matches `OLLAMA_MODEL` in [config.py](config.py) (defaults to `qwen2.5:7b-instruct`) — skip this if using MLX-LM instead

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

> This tuning section is Ollama-specific. If you're on an Apple Silicon Mac, [MLX-LM](#using-mlx-lm-on-a-macbook-apple-silicon) runs Metal-native and is usually both faster and less memory-hungry than Ollama's llama.cpp backend on the same hardware.

**PowerShell**

```powershell
$env:OLLAMA_MODEL = "qwen2.5:14b-instruct"
$env:OLLAMA_NUM_CTX = "16384"
$env:RAG_TOP_K = "8"
```

**Bash** (for m1 macbook air 16gb)

```bash
export OLLAMA_MODEL="qwen2.5:7b-instruct"
export OLLAMA_NUM_CTX="8192"
export RAG_TOP_K="8"
```

## Using MLX-LM on a MacBook (Apple Silicon)

[MLX-LM](https://github.com/ml-explore/mlx-lm) runs on Apple's MLX framework instead of Ollama's llama.cpp backend, so inference goes straight to Metal on the GPU. On an M-series MacBook this is typically faster and lighter on memory than the same model through Ollama — worth switching to if you're generating large question banks locally.

The pipeline talks to MLX-LM through `mlx_lm.server`, which exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Set `LLM_BACKEND=mlx` and it's a drop-in replacement for Ollama — `ingest` is unaffected either way, since only `generate` calls the LLM.

**1. Install MLX-LM** (separate from `requirements.txt` — it's macOS/Apple Silicon-only, so it isn't a default dependency):

```bash
pip install mlx-lm
```

If your `venv` predates this feature, `openai` won't be installed yet — re-run `pip install -r requirements.txt` to pick it up (it's used only as an HTTP client for `mlx_lm.server`'s OpenAI-compatible API, no OpenAI account needed).

**2. Start the MLX-LM server** with a model from the [mlx-community](https://huggingface.co/mlx-community) org on Hugging Face (pre-quantized for Apple Silicon). It downloads the model on first run and keeps it resident in memory between requests:

```bash
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

> `mlx_lm.server` prints no "ready" banner — after the `Fetching N files: 100%` line it goes quiet but is already serving. Don't mistake the silence for a hang; confirm it's up with `curl http://localhost:8080/v1/models`.

**3. Point the pipeline at it** by setting `LLM_BACKEND=mlx` (and `MLX_MODEL` to match whatever you passed to `--model` above, if different from the default):

```bash
export LLM_BACKEND="mlx"
export MLX_MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"
export MLX_BASE_URL="http://localhost:8080/v1"   # default — omit unless you changed --port
```

Then run `ingest` / `generate` as usual — `python main.py generate --cert SAA-C03 --count 10` works unchanged, it's just talking to MLX-LM instead of Ollama.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_BACKEND` | `ollama` | Set to `mlx` to use MLX-LM instead of Ollama |
| `MLX_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Model to request from `mlx_lm.server` |
| `MLX_BASE_URL` | `http://localhost:8080/v1` | `mlx_lm.server`'s OpenAI-compatible base URL |
| `MLX_NUM_CTX` | `8192` | Context window used for prompt-side truncation (raise for a model with a larger trained context) |

To switch back to Ollama, unset `LLM_BACKEND` (or set it to `ollama`) and make sure `ollama serve` is running.

### Model size vs. approval rate

The 7B model is fast but doesn't always follow the exact-option-count instructions in the prompt (e.g. 5 options on a single-answer question), which the schema validator rejects outright — not a reviewer rejection, a generation failure. Bumping to a 14B model fixed this in testing:

| Model | Hardware | Result (5-question test run) | Pace |
| --- | --- | --- | --- |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | M1 MacBook Air, 16 GB | Some generation failures from malformed option counts | Faster |
| `mlx-community/Qwen2.5-14B-Instruct-4bit` | M1 MacBook Air, 16 GB | 5/5 approved (100%) | ~2.5 min/question |

```bash
mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit --port 8080
export MLX_MODEL="mlx-community/Qwen2.5-14B-Instruct-4bit"
```

The 14B model is a tight fit on 16 GB unified memory — expect a slow first request (~1 min, one-time weight load) and close other memory-heavy apps if you hit swapping. On 16 GB+ systems it's worth the trade for large batch runs; for quick iteration the 7B model is still the faster choice.

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

$certBatches = @{
    "SAA-C03" = 2; "DOP-C02" = 3; "AIP-C01" = 3
}

foreach ($cert in $certBatches.Keys) {
    for ($i = 1; $i -le $certBatches[$cert]; $i++) {
        python main.py generate --cert $cert --count 100 --output json
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
- **`MLX-LM server not reachable`** (`LLM_BACKEND=mlx`) — start it with `mlx_lm.server --model <MLX_MODEL>` (see [Using MLX-LM on a MacBook](#using-mlx-lm-on-a-macbook-apple-silicon)).
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
| [llm_backend.py](llm_backend.py) | Backend-agnostic LLM client — adapts Ollama and MLX-LM (via `mlx_lm.server`'s OpenAI-compatible API) to the same interface |
| [schemas.py](schemas.py) | Pydantic v2 models (`ExamQuestion`, `ValidationCritique`, `GenerationRequest`, ...) |
| [config.py](config.py) | Central config (paths, Ollama/MLX-LM settings, embedding/generation tuning) |
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
