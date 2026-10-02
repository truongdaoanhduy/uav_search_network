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

Full-flow training requires W&B online. After the final checkpoint, CPU
reference evaluation and PNG/MP4 visualization run in the same machine and
session. Kaggle stays on the same Kaggle account. Vast stays on the same Vast
instance. Visualization media is local by default, while the final checkpoint
is published as a W&B model artifact.

Adding a new algorithm

1. Add configs/algorithm/NAME.yaml.
2. Register metadata in uav_marl/algorithms.py.
3. Implement the learner while preserving the task observation/action/state contract.
4. Connect that learner to the engine registry.
5. Reuse train.py, task, reward, runtime, W&B, checkpoint and visualization code.

The notebook loader is transitional. It rejects unexpected top-level executable
notebook cells so train.py cannot silently execute benchmark or experiment calls
while loading the current notebook-based engine.
