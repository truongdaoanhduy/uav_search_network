# MASAC Kaggle failure: masked categorical entropy overflow

Affected kernel: haibro1234/uav-masac-gha-37644852694-1-gpu.
Affected W&B run: uav_search_paper/uav_search_target/b6a0169c.
Investigated 2026-10-08 against base commit 6534361.

## Evidence and mechanism

Kaggle reports ERROR and W&B reports failed. The terminal traceback is
Normal(loc=mean, scale=std) rejecting NaN actor means on both CUDA ranks.
The last recorded global_step is 25,165,824, with 16,049 of 20,000 episodes
completed and update_count 98,304. Last alpha_discrete is 127.46006775;
the recorded critic gradient norm (before clipping) is 321,424.875.

The categorical helper represented invalid logits using float32 finfo.min.
Although exp(log_probability) * log_probability has finite forward values
on those entries, entropy backward scales the sentinel by alpha / batch.
With the actual per-rank minibatch of 128, alpha above approximately 128
overflows this intermediate. The exp backward then encounters 0 * infinity,
producing NaN gradients even for a silent-only row. The next optimizer step
can corrupt actor weights; Normal reports the corruption on a later forward.

A controlled 128-row, six-agent reproduction gives finite gradients at
alpha=127 and NaN at alpha=129. This matches the last recorded temperature.
The old run did not save its first failing replay minibatch, so this is a
reproduced code-level failure mechanism consistent with the cloud trace,
not a bitwise replay of its final optimizer step.

The earlier finite-gradient guard detects bad gradients but does not remove
this arithmetic failure. Reducing target entropy alone is not a root-cause
fix and does not prove a long training run will remain stable.

## Changes

- Mask invalid log probabilities to zero *before* the entropy product.
  Legal-action entropy and its analytic gradients are unchanged.
- Gather the chosen legal log probability instead of multiplying all entries
  by a one-hot vector. The old expression also produced 0 * -infinity in
  float16 for a legal logit of 64.
- Replace clamped tanh log-Jacobians with PyTorch TanhTransform's stable
  expression, including the radial motion Jacobian and both single-device
  and DDP power paths. At motion radius 20 the correct derivative is -2.1;
  the old clamp yielded -0.1. At power latent 20 the old derivative vanished.
- Synchronize configs/algorithm/masac.yaml target entropy ratio to 0.89.
  The prior patch changed config.py only; Hydra still overwrote it with 0.98.

Frozen fix_test_gpu.ipynb is unchanged. No scenario, reward, observation
dimension, or network semantics are changed.

## Normalization audit

The configured paper scenario yields actor observations [E, 6, 333] and
critic states [E, 3075]. CPU and tensor builders normalize physical
quantities by map/altitude/speed/capacity/TTL. Signed features remain in
[-1, 1], probability/fraction channels in [0, 1]. Belief preprocessing
centers the patch and converts coarse uncertainty to certainty.
The reset audit returned finite actor/state tensors within their bounds.
Mixed bounded ranges are not the source of the reproduced NaN.

## Verification

- Red/green tests reproduced masked-entropy overflow, Hydra preset mismatch,
  lost saturated Jacobian gradients, and float16 selected log-probability NaN.
- Main suite: 165 passed, 2 optional UavNetSim tests skipped.
- Explicit RUN_UAVNETSIM_INTEGRATION=1 run: both skipped tests passed.
  Total: 167 executable tests passed across these runs.
- Additional float64 autograd-Jacobian oracle: zero and two moderate
  three-dimensional inputs matched the analytic log determinant.
- Original episode-15010 checkpoint: model/optimizer load and high-alpha
  actor-gradient check passed. Its resume_safe flag is false; it is not
  sufficient for exact replay/environment/RNG resumption.
- The bounded two-T4 Kaggle validator
  `haibro1234/uav-masac-nan-fix-validation-20261008` completed successfully.
  Both Tesla T4 ranks reproduced the historical CUDA masked-entropy failure,
  then kept the fixed entropy finite at alpha=1000 and completed 64 real DDP
  optimizer updates (batch 128 per rank) plus 128 CUDA Graph action steps.
  Actor/critic gradient norms and all reported metrics remained finite; actor
  observations and critic states stayed finite within [-1, 1].
- The detached GPU -> CPU smoke pipeline was also reconciled successfully:
  the CPU UavNetSim stage produced both PNG and MP4 evaluation artifacts.

The failed historical run remains failed. No new 20,000-episode training
run was launched during this investigation. Passing regression/smoke tests
and the bounded two-T4 stress test do not by themselves establish
full-horizon learning quality or absence of all bugs.
