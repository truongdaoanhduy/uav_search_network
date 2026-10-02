# UAV MARL training layout

The repository composes experiments from five independent config groups:

- task: UAV-search environment, sensing, communication, observation and energy.
- reward: reward definition or ablation profile.
- algorithm: learner-only hyperparameters such as MASAC or MATD3.
- runtime: provider, hardware and vectorization such as Kaggle 2xT4 or Vast 1 GPU.
- experiment: episode budget, W&B, checkpointing, evaluation and visualization.

config.py remains a compatibility bridge for the notebook-first engine.
New runs should use train.py.

Common commands

Kaggle 2xT4 paper run:
  python train.py algorithm=masac runtime=kaggle_2xt4 experiment=paper_50k seed=44

Same task and reward on one Vast GPU:
  python train.py algorithm=masac runtime=vast_1gpu experiment=paper_50k seed=44

Fair algorithm comparison:
  python train.py algorithm=matd3 runtime=kaggle_2xt4 experiment=paper_50k seed=44

Reward ablation:
  python train.py algorithm=masac reward=search_heavy runtime=kaggle_2xt4 experiment=paper_50k

One-off hyperparameter override:
  python train.py algorithm=masac algorithm.actor_lr=1e-4

Inspect the resolved run without training:
  python train.py runtime=vast_1gpu dry_run=true

experiment.total_episodes is the user-facing budget. The runner derives the
legacy transition ceiling internally from total episodes, number of vector
environments and max episode steps.

Full-flow training requires W&B online. Production visualization is now split
into a separate CPU session by default. The GPU job trains, writes/publishes the
final checkpoint, creates a verified handoff package, and then exits. Only after
the GPU kernel reaches COMPLETE does the launcher start a CPU-only Kaggle kernel
with enable_gpu=false. The CPU kernel consumes the GPU kernel output through
kernel_sources, verifies the checkpoint SHA256, runs the authoritative CPU
reference evaluation, and renders PNG/MP4. Visualization media stays out of W&B
by default, while the final model checkpoint can still be published as a W&B
artifact.

Automated Kaggle GPU -> CPU pipeline:
  python scripts/kaggle_pipeline.py --algorithm masac --experiment paper_50k

MATD3:
  python scripts/kaggle_pipeline.py --algorithm matd3 --experiment paper_50k

The launcher:
1. checks Kaggle GPU quota and chooses an account;
2. updates/runs one persistent private GPU kernel;
3. waits until the GPU kernel is COMPLETE;
4. starts a second private CPU-only kernel with the GPU output attached;
5. waits for CPU visualization to complete;
6. downloads and verifies the completion manifest, PNG and MP4 locally.

The CPU visualization kernel does not need W&B credentials to obtain the
checkpoint because it reads the GPU kernel output directly. Kaggle API/CLI
versions do not automatically inherit interactive Notebook Secrets, so W&B
credential transport for an API-launched GPU job is treated as a separate
credential concern rather than being embedded in source or in the handoff.

The base requirements intentionally do not install the full UavNetSim dependency
tree because GPU training uses uavnetsim_gpu. Kaggle CPU post-processing
bootstraps the pinned reference simulator with --no-deps to preserve the trained
CUDA/NumPy environment. For local authoritative CPU/reference evaluation, run:
  pip install -r requirements-uavnetsim.txt

Adding a new algorithm

1. Add configs/algorithm/NAME.yaml.
2. Register metadata in uav_marl/algorithms.py.
3. Implement the learner while preserving the task observation/action/state contract.
4. Connect that learner to the engine registry.
5. Reuse train.py, task, reward, runtime, W&B, checkpoint and visualization code.

The notebook loader is transitional. It rejects unexpected top-level executable
notebook cells so train.py cannot silently execute benchmark or experiment calls
while loading the current notebook-based engine.
