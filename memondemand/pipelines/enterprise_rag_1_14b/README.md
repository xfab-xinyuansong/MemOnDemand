# EnterpriseRAG-Bench 1.14B pipeline

This package contains the MemOnDemand pipeline used for the official FULL
EnterpriseRAG-Bench collection (618M tokens) and its project-conditioned 1.14B-token
extension. It keeps the three system mechanisms explicit: data-dependent hierarchy
construction, detailed and distilled memory, and on-demand promotion with decay. Gold
answers and expected source identifiers are loaded only by the evaluator.

## Stages

1. `hierarchy_build` constructs the source-resolved hierarchy and provides
   resumable utilities for full-corpus key-fact generation.
2. `answer_generation` performs multi-level retrieval, promotion, and grounded
   answer generation. `improved_answer_prompt.py` contains the final prompt.
3. `evaluation` computes retrieval, answer-quality, citation-validity, and
   promotion statistics from evaluator-only gold records.

Run modules from the repository root, for example:

```bash
python -m memondemand.pipelines.enterprise_rag_1_14b.hierarchy_build.build_hierarchy_erag \
  --erag_parquet /path/to/full_l0_nodes.parquet \
  --tier_label 1.14B \
  --out results/enterprise_rag_1_14b/hierarchy

python -m memondemand.pipelines.enterprise_rag_1_14b.answer_generation.run_stream_v5_truncate_patch \
  --help

python -m memondemand.pipelines.enterprise_rag_1_14b.evaluation.evaluate_v5 \
  --answers /path/to/answers.jsonl \
  --gold /path/to/evaluator_only_gold.jsonl \
  --out results/enterprise_rag_1_14b/evaluation
```

## Runtime configuration

The package uses `memondemand.core.api_adapter`; it contains no credentials or
private endpoints. Configure credentials through the repository's documented
environment variables. Model roles default to `gpt_5_4_mini` for high-volume
low-level work and `gpt_5_4` for high-level abstraction, answering, and judging.
Repositories exposing only the public `general` alias map both roles through
that gateway. Optional overrides use
`MEMONDEMAND_MODEL_ALIAS_GPT_5_4_MINI` and
`MEMONDEMAND_MODEL_ALIAS_GPT_5_4`.

The final runner also supports:

- `MEMONDEMAND_EMBED_BACKEND=azure_small` for the 1,536-dimensional embedding
  configuration;
- `MEMONDEMAND_ANSWER_MODE=detailed_truncated` and
  `MEMONDEMAND_TRUNCATE_CHARS=2000` for the detailed-memory budget;
- `MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE` for the grounded answer prompt;
- `MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR` for a validated promotion threshold;
- `MEMONDEMAND_TE3_INDEX_CACHE` for an existing embedding-index cache.

All input, output, and checkpoint locations are command-line arguments or
environment settings; no local paths or service endpoints are embedded here.

## Force-answer verification reruns

The default runner may return `STOP_INSUFFICIENT`. For an explicit best-effort
rerun, use the dedicated force-answer entry point with the same arguments as
the normal runner:

```bash
python -m memondemand.pipelines.enterprise_rag_1_14b.answer_generation.run_stream_v5_force_answer_patched \
  --method V5 \
  --hierarchy /path/to/hierarchy.json \
  --queries /path/to/rerun_queries.parquet \
  --out /path/to/rerun_output \
  --resume
```

`run_empty14_force_answer_wrapper` is the matching convenience wrapper for a
manifest containing the previously unanswered queries. It also applies the
configured detailed-memory truncation and answer-prompt override:

```bash
export MEMONDEMAND_ANSWER_MODE=detailed_truncated
export MEMONDEMAND_TRUNCATE_CHARS=2000
export MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE="$PWD/memondemand/pipelines/enterprise_rag_1_14b/answer_generation/improved_answer_prompt.py"

python -m memondemand.pipelines.enterprise_rag_1_14b.answer_generation.run_empty14_force_answer_wrapper \
  --method V5 \
  --hierarchy /path/to/hierarchy.json \
  --queries /path/to/empty14_queries.parquet \
  --out /path/to/empty14_rerun \
  --resume
```

Force-answer mode changes only the terminal abstention decision. Retrieval,
evidence selection, citation extraction, and output accounting remain active.
