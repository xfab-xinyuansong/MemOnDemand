# Configuration

MemOnDemand reads configuration from environment variables. By default it loads a
local `.env` file from `MEMONDEMAND_REPO_ROOT` or the current working directory.

## Environment Files

```bash
cp .env.example .env
```

Set `MEMONDEMAND_ENV_FILE` when you want to keep configuration outside the project
directory:

```bash
export MEMONDEMAND_ENV_FILE=/secure/path/memondemand.env
```

## API Settings

| Variable | Required for | Notes |
| --- | --- | --- |
| `MEMONDEMAND_API_BASE_URL` | Model-backed answering and judging | OpenAI-compatible chat-completions base URL. |
| `MEMONDEMAND_API_KEY` | Model-backed answering and judging | API key for your model gateway. |
| `MEMONDEMAND_API_MODEL` | Model-backed answering and judging | Chat model name or deployment alias. |
| `MEMONDEMAND_EMBED_API_BASE_URL` | API-backed embeddings | Optional OpenAI-compatible embedding endpoint. |
| `MEMONDEMAND_EMBED_API_KEY` | API-backed embeddings | Optional embedding API key. |
| `MEMONDEMAND_EMBED_API_MODEL` | API-backed embeddings | Optional embedding model or deployment alias. |
| `MEMONDEMAND_EMBED_DIM` | API-backed embeddings | Optional embedding dimension override. |

MemOnDemand's default public configuration is provider-neutral. Advanced deployments
can still add project-specific provider aliases in `memondemand.core.api_adapter`
when they need a custom gateway, tenancy layer, or hosted model runtime.

## Retrieval Settings

| Variable | Default | Notes |
| --- | --- | --- |
| `MEMONDEMAND_EMBED_BACKEND` | `minilm` | Local embedding backend for development. |
| `MEMONDEMAND_L0_RETRIEVAL` | `bm25` | L0 candidate retrieval mode. |
| `MEMONDEMAND_SKIP_L0_EMBED` | `0` | Set to `1` to avoid embedding all L0 nodes. |
| `MEMONDEMAND_INDEX_CACHE_DIR` | unset | Optional cache directory for retrieval indexes. |
| `MEMONDEMAND_ANSWER_MODE` | runner default | Use `detailed_truncated` for compact evidence contexts. |

## Secret Policy

Do not commit `.env`, generated JSONL answers, parquet manifests, logs, caches,
or provider keys. `.gitignore` already excludes common local artifacts.
