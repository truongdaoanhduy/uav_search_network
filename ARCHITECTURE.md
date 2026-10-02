# UAV MARL production architecture

## Rule 1: production code is Python, notebook is frozen

`fix_test_gpu.ipynb` is retained only as a frozen reference/regression artifact.
Production training, evaluation, checkpointing and visualization must import
Python modules under `uav_marl/`. Do not add new production logic to the
notebook.

## Recommended reading order

1. `configs/task/uav_search.yaml` — task/environment parameters.
2. `configs/reward/paper_v1.yaml` — reward definition.
3. `uav_marl/world/` — entities, world generation, motion, sensing, reports,
   reward/energy and observation construction.
4. `uav_marl/network/` — communication semantics and network backends.
5. `uav_marl/envs/uav_search.py` — CPU/reference Gym environment.
6. `uav_marl/algorithms/masac.py` or `matd3.py` — learner implementation.
7. `uav_marl/envs/gpu.py` — batched tensor environment used for large GPU runs.
8. `uav_marl/training/gpu.py` — replay, DDP, W&B, checkpointing and GPU trainer.
9. `uav_marl/evaluation/postprocess.py` — reference CPU evaluation/rendering.
10. `uav_marl/runner.py` and `train.py` — experiment orchestration.
11. `scripts/kaggle_pipeline.py` — Kaggle GPU-session to CPU-session automation.

## Package layout

```text
uav_marl/
├── common.py
├── world/
│   ├── entities.py
│   ├── generation.py
│   ├── motion.py
│   ├── sensing.py
│   ├── reports.py
│   ├── reward_energy.py
│   └── observations.py
├── network/
│   ├── communication.py
│   └── backends.py
├── envs/
│   ├── uav_search.py
│   └── gpu.py
├── algorithms/
│   ├── registry.py
│   ├── masac.py
│   └── matd3.py
├── training/
│   └── gpu.py
├── evaluation/
│   └── postprocess.py
├── configuration.py
├── runner.py
└── handoff.py
```

## Design rules

- Environment/task semantics must not depend on which algorithm is selected.
- Algorithm modules must consume the environment contract; they must not
  duplicate world/network/reward configuration.
- Runtime/provider-specific code belongs in runtime/training/orchestration
  layers, not in MASAC/MATD3.
- New algorithms should be registered through `algorithms/registry.py` and
  reuse the same task/runtime/evaluation pipeline.
- Hydra/YAML is the user-facing configuration surface. `config.py` remains a
  compatibility bridge while legacy flat CONFIG access still exists internally.
- CPU/reference environment and GPU batched environment must be regression
  tested for semantic parity.
- W&B/checkpoint/visualization are experiment services and should remain outside
  algorithm math whenever possible.

## Why this structure

The layout intentionally combines three proven research-code patterns:

- BenchMARL/EPyMARL: separate task/environment configuration from algorithm
  configuration so algorithms can be compared on the same task.
- CleanRL: keep each algorithm implementation easy to inspect in its own module.
- Acme/RLlib-style composition: make algorithms interchangeable behind common
  experiment/runtime infrastructure.

This repository does not copy those frameworks directly; it borrows the
separation of concerns that best fits a custom UAV MARL simulator.
