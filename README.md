# Glimpse Backend

FastAPI backend for multimodal search powered by ImageBind embeddings and ChromaDB. Includes optional NSFW flagging (precomputed boolean on items) and a simple text-based batch flagging script.

## Features
- Multimodal search endpoints (web, image, audio, video, all)
- Persistent ChromaDB index stored under `index_data/` (git-ignored)
- ImageBind model loading with CUDA support and diagnostics
- Optional NSFW metadata on results (`is_nsfw`, `nsfw_score`, `nsfw_confidence`)
- Batch NSFW flagging scripts (GPU-aware and text-only)
- Telemetry suppression for ChromaDB/PostHog noise-free logs

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (optional but recommended)
- uv (fast Python package/environment manager)

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies
From this directory:
```bash
uv sync
```

### Run the API
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The API will start on `http://localhost:8000`.

### Index paths and telemetry
- ChromaDB persistence directory: `index_data/chroma_db` (ignored in git)
- Telemetry is disabled in `main.py` to avoid PostHog warnings

## API Overview

- `GET /searchweb?query=...&top_k=100&filter_nsfw=false`
- `GET /searchimage?query=...&top_k=100&filter_nsfw=false`
- `GET /searchaudio?query=...&top_k=100&filter_nsfw=false`
- `GET /searchvideo?query=...&top_k=100&filter_nsfw=false`
- `GET /status`

When NSFW flags exist in metadata, results include fields like:
```json
{
  "is_nsfw": true,
  "nsfw_score": 0.8,
  "nsfw_confidence": 0.9
}
```
Use `filter_nsfw=true` to exclude flagged items from results.

## NSFW Flagging

Two approaches are provided to (pre)compute NSFW flags in the index metadata.

### 1) Text-based (fast, no model)
Scans documents/metadata for NSFW keywords and updates metadata.
```bash
uv run python simple_nsfw_flag.py
```
- Adds `is_nsfw`, `nsfw_score`, `nsfw_confidence`, and `nsfw_keywords`
- Works without GPU

### 2) Model-based (ImageBind zero-shot)
Computes similarities between content and NSFW/safe prompts and updates metadata.
```bash
# Prefer GPU 1 if present, otherwise fall back automatically
CUDA_VISIBLE_DEVICES=1 uv run python add_nsfw_flags.py
```
Notes:
- Attempts to use `cuda:1` when multiple GPUs are present; otherwise picks best available device
- Handles various embedding shapes/types retrieved from ChromaDB

## Development

### Code Style
- Follow PEP 8; include type hints where practical
- Keep functions small and well-named; prefer early returns
- Avoid catching exceptions without meaningful handling

### Linting
(If configured) run from this folder:
```bash
uv run ruff check .
```

### Tests
Place tests under `tests/` and run:
```bash
uv run pytest -q
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository and create a feature branch
   ```bash
   git checkout -b feat/your-feature
   ```
2. Set up the environment
   ```bash
   uv sync
   ```
3. Make focused, atomic commits with clear messages
   - Conventional commits encouraged (`feat:`, `fix:`, `docs:`, `chore:`)
4. Run the API locally and verify endpoints
5. Update documentation if behavior changes
6. Open a Pull Request
   - Describe the change, motivation, and testing steps
   - Link related issues if any

### PR Guidelines
- Keep changes scoped; avoid unrelated refactors
- Include before/after behavior where relevant
- Ensure no large binary or index files are committed

### Issue Reporting
Please include:
- Environment (OS, Python, CUDA, GPU)
- Logs (errors/warnings)
- Steps to reproduce

## Repository Hygiene
This backend ignores heavy and generated assets:
- `index_data/` (ChromaDB persistence)
- log files (e.g., `indexer.log`, `**/logs/**`, `*.log*`)
- virtual envs and caches

If you notice other large/generated paths, propose an update to `.gitignore`.

---
Maintained by the Glimpse team. Thanks for contributing!
