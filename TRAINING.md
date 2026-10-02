# UAV MARL training

Production training is Python-only. The frozen fix_test_gpu.ipynb notebook is
not imported or executed by train.py, torchrun workers, CPU post-processing, or
the Kaggle pipeline.

See ARCHITECTURE.md for the full module map and recommended reading order.

## Configuration groups

Experiments are composed from five independent Hydra config groups:

- task: UAV-search environment/sensing/communication/energy settings.
- reward: reward profile or ablation.
- algorithm: MASAC/MATD3 learner hyperparameters.
- runtime: provider, device, vectorization and GPU/DDP settings.
- experiment: training budget, W&B, checkpoints and visualization policy.

config.py remains a compatibility dictionary for the migrated implementation.
Do not use it as the normal experiment interface; use configs/ and CLI
overrides.

## Direct training commands

Kaggle-style 2-GPU configuration:
  python train.py algorithm=masac runtime=kaggle_2xt4 experiment=paper_50k seed=44

Vast one-GPU configuration:
  python train.py algorithm=masac runtime=vast_1gpu experiment=paper_50k seed=44

MATD3 with the same task/reward/runtime:
  python train.py algorithm=matd3 runtime=kaggle_2xt4 experiment=paper_50k seed=44

Reward ablation:
  python train.py algorithm=masac reward=search_heavy runtime=kaggle_2xt4 experiment=paper_50k

One-off algorithm override:
  python train.py algorithm=masac algorithm.actor_lr=1e-4

Inspect the fully resolved run without training:
  python train.py runtime=vast_1gpu dry_run=true

experiment.total_episodes is the user-facing budget. The runner derives the
internal transition ceiling automatically.

## Automated Kaggle GPU -> CPU pipeline

Recommended full pipeline:
  python scripts/kaggle_pipeline.py --algorithm masac --experiment paper_50k

MATD3:
  python scripts/kaggle_pipeline.py --algorithm matd3 --experiment paper_50k

The launcher:
1. checks Kaggle GPU quota and selects/resolves an account;
2. launches a GPU training kernel;
3. waits until that GPU kernel reaches COMPLETE;
4. starts a second CPU-only kernel with enable_gpu=false;
5. attaches the GPU output through kernel_sources;
6. verifies the checkpoint handoff SHA256;
7. runs authoritative CPU/reference evaluation and renders PNG/MP4;
8. downloads and verifies the CPU completion manifest locally.

The GPU session is therefore not kept alive while CPU visualization runs.

## Production source locations

Reference Gymnasium environment:
  uav_marl/envs/uav_search.py

Batched GPU environment:
  uav_marl/envs/gpu.py

MASAC:
  uav_marl/algorithms/masac.py

MATD3:
  uav_marl/algorithms/matd3.py

GPU/DDP trainer:
  uav_marl/training/gpu.py

CPU evaluation/rendering:
  uav_marl/evaluation/postprocess.py

Experiment orchestration:
  uav_marl/runner.py

GPU-to-CPU handoff:
  uav_marl/handoff.py

## Adding a new algorithm

1. Add configs/algorithm/NAME.yaml.
2. Add uav_marl/algorithms/NAME.py.
3. Register metadata in uav_marl/algorithms/registry.py.
4. Preserve the existing observation/action/state contract, or version it
   explicitly if the algorithm requires a different contract.
5. Reuse the existing world, network, runtime, W&B, checkpoint/handoff and CPU
   evaluation pipeline.
6. Add deterministic CPU and GPU smoke tests before a long paper run.

## Frozen notebook

fix_test_gpu.ipynb is retained as a reference/regression artifact only. Its
migration-time SHA256 is asserted by the test suite. Production code must not
load it. New production fixes belong in the Python modules.
