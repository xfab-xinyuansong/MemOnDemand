#!/usr/bin/env python3
"""Re-evaluate existing answers with the configured GPT-5.4 alias."""

import sys

from memondemand.pipelines.enterprise_rag_1_14b.evaluation import evaluate_v5


def main() -> int:
    evaluate_v5.JUDGE_ALIAS = "gpt_5_4"
    return evaluate_v5.main()


if __name__ == "__main__":
    sys.exit(main())
