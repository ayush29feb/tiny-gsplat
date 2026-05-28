# CLAUDE.md

## Rules

- Never delete anything under `outputs/` without explicitly asking the user first. These contain experiment results that may be needed for comparison.
- After each experiment, document it in `EXPERIMENTS.md` with: config, results table, plots, auto observations, and user observations (pending until user provides them).
- Between experiments, include a **Changelog** section listing the git commits and code changes that happened since the previous experiment. This tracks what code was modified and why.
- Always look at val render and test render images when making conclusions — PSNR numbers alone don't tell the full story. Include render comparisons in the experiment doc.

## Project

See [README.md](README.md) for project overview, setup, usage, and architecture.

Experiment results and observations are tracked in [EXPERIMENTS.md](EXPERIMENTS.md).
