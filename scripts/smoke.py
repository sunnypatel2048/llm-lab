"""Thin wrapper so `uv run scripts/smoke.py "prompt"` works alongside `uv run llm-lab "prompt"`.
The logic lives in llm_lab.smoke so it can be tested."""

import sys

from llm_lab.smoke import main

if __name__ == "__main__":
    sys.exit(main())
