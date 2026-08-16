"""
SprintLM entrypoint. This project is driven via `python -m` module calls
and the tools/ scripts rather than a single main() — see README.md for the
full command list. This file just prints a pointer so `python main.py`
doesn't dead-end.
"""

COMMANDS = """
SprintLM — available commands:

  Data prep:
    uv run python tools/prepare_hf.py dataset_name=... out_dir=...

  Train (single run):
    uv run python -m sprintlm.train.train dataset.train_bin=... dataset.val_bin=...

  Run full ablation ladder:
    uv run python tools/run_ablations.py dataset.train_bin=... dataset.val_bin=...

  Seed variance check:
    uv run python tools/seed_variance.py dataset.train_bin=... dataset.val_bin=... --seeds 7 13 42

  Sample text:
    uv run python tools/decode_cli.py --ckpt outputs/<run>/ckpts/step5000.pt --prompt "..." --use_cache

  Benchmark KV-cache vs no-cache:
    uv run python tools/benchmark_inference.py --ckpt outputs/<run>/ckpts/step5000.pt

  Plot metrics:
    uv run python tools/plot_metrics.py outputs/<run>

  Run tests:
    uv run pytest tests/ -v

See README.md for full details.
"""

if __name__ == "__main__":
    print(COMMANDS)
