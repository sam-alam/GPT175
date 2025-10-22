
# GPT‑175 — Postgres + pgvector RAG Pack

This bundle gives you a production‑friendly RAG backend for Article 175 using **pgvector** and a small **FastAPI** service.

## Contents
- `db/schema.sql` — pgvector schema + indexes
- `scripts/load_pgvector.py` — loader from your `article_175_chunks.jsonl`
- `service/retriever_service.py` — FastAPI hybrid retriever (vector + BM25 via tsvector)
- `service/requirements.txt` — service deps
- `.env.example` — environment template

## 0) Prereqs
- Postgres 15/16 with `pgvector` extension installed
- Python 3.10+
- Your chunked corpus: `article_175_chunks.jsonl` (from your normalization pipeline)

## 1) Create DB schema
```bash
psql "$DATABASE_URL" -f db/schema.sql
```

## 2) Load data (embed + upsert)
```bash
export OPENAI_API_KEY=sk-...
python scripts/load_pgvector.py --jsonl /path/to/article_175_chunks.jsonl \
       --dsn "$DATABASE_URL" --embed_model text-embedding-3-small
```

This script:
- prefixes each chunk with `[Article: ...] [Section: ...] [Title: ...]` to boost retrieval
- embeds with OpenAI `text-embedding-3-small` (1536 dims)
- upserts rows and builds the `tsvector` for hybrid search

## 3) Start retriever API
```bash
cd service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export $(cat ../.env.example | xargs)   # or create your real .env
uvicorn retriever_service:app --host 0.0.0.0 --port 8000
```

### /search (POST)
Request:
```json
{ "query": "rules for mini c-arm image quality", "top_k": 12 }
```
Response:
```json
{
  "query": "rules for mini c-arm image quality",
  "items": [
    {
      "id": "§175.53(a)::37",
      "article": "Article 175",
      "section": "§175.53(a)",
      "title": "Fluoroscopy equipment requirements",
      "heading_path": ["..."],
      "page_start": null,
      "page_end": null,
      "content": "[Article: Article 175] [Section: §175.53(a)] ...",
      "vec_score": 0.76
    }
  ]
}
```

## 4) Wire Streamlit to the service
Replace your in‑app retrieval with a call to `POST /search` and feed the returned `items` as context for the LLM.
Keep your strict, citation‑first prompt. Show `section` in “Sources consulted”.

## 5) Notes
- Keep embeddings model **identical** for indexing and querying.
- To add page‑accurate citations later, backfill `page_start/page_end` in the JSONL and re‑load.
- For synonym boosting, expand the user query before calling `/search` (as you already do).
- For re‑ranking, you can add a CrossEncoder step client‑side over the returned `items`.

---

**Security/Compliance**: This tool summarizes public regulation text with citations. It is not legal advice.
