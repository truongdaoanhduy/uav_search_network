# UAV MARL architecture

Production code lives in Python modules. The historical fix_test_gpu.ipynb
notebook is frozen as a regression/reference artifact and is not imported by
train.py, the DDP launcher, CPU post-processing, or the Kaggle pipeline.

## Design principles

This layout intentionally borrows ideas from established RL/MARL projects:

- BenchMARL: separate Task, Algorithm, Model/Experiment configuration so one
  component can change without duplicating the rest of the experiment.
- DeepMind Acme: reusable RL building blocks that can run in both single-stream
  and distributed settings.
- Isaac Lab: task/environment code is separated from agent/training
  configuration and selected through entrypoints.
- CleanRL: algorithm implementations remain readable and self-contained enough
  to study directly rather than being hidden behind many indirection layers.

## Production module graph

uav_marl/
  common.py
  world/
    entities.py
    generation.py
    motion.py
    sensing.py
    reports.py
    reward_energy.py
    observations.py
  network/
    communication.py
    backends.py
  envs/
    uav_search.py
    gpu.py
  algorithms/
    registry.py
    masac.py
    matd3.py
  training/
    gpu.py
  evaluation/
    postprocess.py
  configuration.py
  runner.py
  handoff.py

The dependency direction is intentionally one-way:

common
  -> world
  -> network
  -> reward/observations
  -> envs
  -> algorithms
  -> GPU env
  -> training
  -> evaluation

This keeps environment semantics independent from any specific learner and
allows a new algorithm to reuse the existing UAV world/network/evaluation code.

## Where the environment lives

Reference CPU/Gymnasium environment:
  uav_marl/envs/uav_search.py
  class UAVSearchEnv

Batched tensor/GPU environment:
  uav_marl/envs/gpu.py
  class FullGpuUAVBatchEnv

World construction and entities:
  uav_marl/world/entities.py
  uav_marl/world/generation.py

Motion:
  uav_marl/world/motion.py

Sensing, Bayesian belief, confirmation:
  uav_marl/world/sensing.py

Report/buffer/TTL handling:
  uav_marl/world/reports.py

Communication/link model:
  uav_marl/network/communication.py

Simple and UavNetSim backends:
  uav_marl/network/backends.py

Reward, information gain and energy:
  uav_marl/world/reward_energy.py

Observation/global state construction:
  uav_marl/world/observations.py

## Algorithms

MASAC:
  uav_marl/algorithms/masac.py

MATD3:
  uav_marl/algorithms/matd3.py

Algorithm selection metadata:
  uav_marl/algorithms/registry.py

A new algorithm should be implemented in its own file and registered without
copying the task, network, reward, runtime, W&B, checkpoint or visualization
implementation.

## Training

User-facing entrypoint:
  train.py

Experiment orchestration:
  uav_marl/runner.py

GPU/single-GPU/DDP implementation:
  uav_marl/training/gpu.py

Kaggle GPU -> CPU session orchestration:
  scripts/kaggle_pipeline.py

## Evaluation and visualization

CPU/reference evaluation and rendering:
  uav_marl/evaluation/postprocess.py

Standalone CPU visualization entrypoint:
  visualize.py

GPU-to-CPU checkpoint handoff:
  uav_marl/handoff.py

## Configuration

Hydra config groups:
  configs/task/
  configs/reward/
  configs/algorithm/
  configs/runtime/
  configs/experiment/

config.py is retained only as a compatibility dictionary used by the migrated
implementation. New experiments should be configured through configs/ and CLI
overrides rather than editing config.py.

## Frozen notebook policy

fix_test_gpu.ipynb is intentionally retained unchanged.

Its SHA256 at the time of the Python migration is:
  47132f07f1fb0ffc84f90171fc422d72a36679b51f01a87ba3dc969e8dbe1a62

The test suite checks this hash. Production modules do not load or execute the
notebook. If notebook experiments are changed later, that should be treated as
an independent test/reference change rather than a production code change.

## Recommended reading order

1. configs/task/uav_search.yaml
2. configs/reward/paper_v1.yaml
3. uav_marl/world/entities.py
4. uav_marl/world/generation.py
5. uav_marl/world/motion.py
6. uav_marl/world/sensing.py
7. uav_marl/world/reports.py
8. uav_marl/network/communication.py
9. uav_marl/network/backends.py
10. uav_marl/world/reward_energy.py
11. uav_marl/world/observations.py
12. uav_marl/envs/uav_search.py
13. uav_marl/algorithms/masac.py
14. uav_marl/algorithms/matd3.py
15. uav_marl/envs/gpu.py
16. uav_marl/training/gpu.py
17. uav_marl/evaluation/postprocess.py
18. uav_marl/runner.py
19. scripts/kaggle_pipeline.py

## Adding a new algorithm

1. Add configs/algorithm/my_algorithm.yaml.
2. Implement uav_marl/algorithms/my_algorithm.py.
3. Register it in uav_marl/algorithms/registry.py.
4. Keep the existing environment observation/action/state contract, or version
   that contract explicitly if it changes.
5. Reuse runner.py, training infrastructure, W&B, checkpoint handoff and CPU
   evaluation wherever the algorithm contract allows.
6. Add deterministic CPU/GPU smoke tests before a long paper run.
