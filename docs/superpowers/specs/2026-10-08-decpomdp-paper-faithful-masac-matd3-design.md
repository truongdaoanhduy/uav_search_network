# Dec-POMDP Paper-Faithful MASAC/MATD3 Design

Date: 2026-10-08

## Goal

Refactor the learning algorithms so a paper can accurately describe the baselines as MASAC and MATD3 under a cooperative Dec-POMDP/CTDE formulation. Preserve only task-specific adaptations that are required by the UAV search-and-communication problem. Remove optional algorithmic modifications that make the baselines harder to identify as standard MASAC/MATD3.

## Dec-POMDP formulation

Model the task as a cooperative Dec-POMDP

G = <I, S, {O_i}, {A_i}, P, O, R, gamma>.

- I: six UAV agents.
- S_t: the simulator global state used only during centralized training/evaluation.
- o_i^t: UAV i's local observation at execution time.
- eta_i^t: UAV i's local information state. It contains the current local observation plus recursively maintained local memory already present in the environment, especially the UAV-specific belief map and report buffer.
- A_i: hybrid action consisting of 3-D acceleration, transmission power, and a discrete next-hop destination.
- a_t = (a_1^t, ..., a_N^t): joint action.
- R(s_t, a_t): cooperative team reward. The existing scalar team reward is used as r_i = r_team for each agent-specific critic.
- Joint decentralized policy: pi(a_t | eta_t) = product_i pi_i(a_i^t | eta_i^t).

The actor must never receive S_t or another UAV's hidden/local memory directly. The centralized critic may use S_t and the full joint action only while training. Critics are discarded during decentralized execution.

## Why no recurrent network in the baseline

A general Dec-POMDP policy may depend on action-observation history tau_i^t. This project already maintains a recursive per-UAV belief map and local report buffer. These variables summarize important local history and are available in the local observation. The baseline therefore uses eta_i^t = f_i(tau_i^t) as a feed-forward information state.

Do not add GRU/LSTM to the baseline. A recurrent actor would create a Recurrent MASAC/Recurrent MATD3 variant and make the comparison less paper-faithful. Recurrent policies can be a later ablation if the local information state proves insufficient.

## Local observation boundary

Keep task-specific local information required for decentralized execution:

- own position, velocity, battery, active flag, mission time;
- GCS relative geometry;
- only contact-visible peer features;
- local obstacle features;
- UAV-specific belief patch and coarse belief representation;
- local report-buffer summary;
- local destination feasibility mask.

The current peer observation implementation already zeros peer features outside contact range, so it does not reveal remote peer state to a decentralized actor.

The destination mask is an action-availability constraint. It remains available to the policy action-selection routine but is not concatenated into the neural encoder input. This avoids making the network learn legality that is already enforced explicitly.

Remove actor-only handcrafted belief centering from the baseline. The environment already supplies bounded normalized features, so the baseline actor consumes the normalized local information state directly.

## Network baseline

Use the same simple MLP family for both methods unless a paper-specific rule requires otherwise:

- hidden layers: 256, 256 (kept as a task-scale hyperparameter, not an algorithmic modification);
- activation: ReLU;
- PyTorch default Linear initialization;
- no LayerNorm, BatchNorm, PReLU, custom Kaiming initialization, residual blocks, attention, transformer, or recurrent layers in the baseline.

The earlier LeakyReLU/Kaiming experiment is therefore reverted for the paper-faithful baseline.

## MASAC architecture

For N UAVs, instantiate N independent stochastic actors pi_i and, for each agent i, two centralized soft critics Q_i,1 and Q_i,2 plus their target critics.

Actor i input:

eta_i^t only.

Actor i output is the minimum task-required hybrid extension:

1. continuous branch: Gaussian mean/log_std for [a_x, a_y, a_z, tx_power], followed by the existing bounded action transforms required by UAV physics;
2. discrete branch: categorical logits for next-hop destination, masked by the local feasible-destination mask.

The hybrid policy is factorized:

pi_i(a_i^c, a_i^d | eta_i) = pi_i^c(a_i^c | eta_i) * pi_i^d(a_i^d | eta_i, mask_i).

Use one SAC temperature alpha_i for the complete action of agent i. The policy log-probability is

log pi_i = log pi_i^c + log pi_i^d.

This replaces the current separate alpha_continuous and alpha_discrete design. The baseline uses a fixed configurable alpha by default so it does not require a custom hybrid automatic-entropy target. Automatic entropy tuning may be added only as an explicitly named ablation/variant.

Critic i input:

Q_i,k(S_t, a_1^t, ..., a_N^t), k in {1,2}.

Critic target:

y_i = r_team + gamma * (1-d) * [min_k Q'_i,k(S_{t+1}, a'_{1:N}) - alpha_i log pi_i(a'_i | eta'_i)].

Next actions are sampled from the current decentralized stochastic actors. MASAC does not require target actors.

Actor-i objective:

J_pi_i = E[alpha_i log pi_i(a_i | eta_i) - Q_i,1(S, a_1, ..., a_i, ..., a_N)].

Only actor i receives gradient from its actor loss. Other agents' actions are detached when optimizing actor i.

Retain safe masked-log-probability arithmetic and stable tanh/radial Jacobian formulas because they implement the intended probability calculation correctly and prevent numerical NaNs; they do not define a new algorithm.

## MATD3 architecture

For N UAVs, instantiate:

- N independent deterministic actors mu_i;
- N target actors mu'_i;
- for every agent i, two centralized critics Q_i,1 and Q_i,2;
- two target critics per agent.

Actor i input:

eta_i^t only.

Actor output uses the minimum hybrid-action extension:

- continuous branch for acceleration and transmit power;
- discrete destination logits.

During actor optimization, the discrete branch uses a masked straight-through Gumbel-Softmax/argmax relaxation so the centralized critic can provide a gradient. During deterministic execution, use masked argmax. This is the task-required bridge from deterministic policy gradients to the discrete routing component.

Remove the current decaying epsilon-greedy destination exploration from the paper-faithful MATD3 baseline. Continuous behavior exploration remains additive Gaussian noise. If discrete exploration beyond Gumbel relaxation is later required, report it explicitly as an ablation/variant.

Target policy smoothing is applied to the continuous target actions only. The smoothed 3-D motion component is projected back into the legal acceleration ball; transmit power is clipped to its legal normalized interval. The discrete target action comes from each target actor under the feasibility mask and does not receive TD3 Gaussian target noise.

For each agent i:

y_i = r_team + gamma * (1-d) * min(Q'_i,1(S', a'_{1:N}), Q'_i,2(S', a'_{1:N})).

Both critics update every critic step.

Actor i updates only every policy_delay steps using

J_mu_i = -E[Q_i,1(S, a_1, ..., mu_i(eta_i), ..., a_N)].

Only actor i receives gradient from this loss. Other actors' actions are detached. Target actors and target critics are Polyak-updated only on delayed policy-update steps, matching MATD3/TD3 semantics.

## Replay buffer

Keep a centralized replay buffer storing:

- local observations for all agents;
- global state;
- continuous actions for all agents;
- discrete destination index/one-hot for all agents;
- destination masks;
- scalar cooperative team reward;
- next local observations;
- next global state;
- next masks;
- terminated/truncated flags.

The scalar team reward is broadcast logically to r_i for per-agent critic losses. No artificial individual reward shaping is introduced.

## Training/execution separation

Training:

local observations -> independent actors -> joint action
                                |
global state -------------------+--> per-agent centralized twin critics

Execution:

eta_i -> actor_i -> action_i

No critic and no global state are required at deployment/execution time.

## What remains custom and why

These are retained because the environment requires them, not because they are algorithmic enhancements:

1. Hybrid action: acceleration + power + route destination.
2. Destination feasibility mask.
3. Radial projection/squash enforcing the 3-D acceleration ball.
4. Local belief map and report buffer as the UAV's information state for partial observability.
5. Centralized global state definition for CTDE training.
6. Team reward corresponding to the cooperative Dec-POMDP objective.
7. Safe probability/Jacobian arithmetic necessary for numerical correctness.
8. DDP, AMP, CUDA Graphs, vector environments, W&B, and checkpointing as implementation infrastructure only.

## Removed/changed relative to the current implementation

1. Shared actor -> one actor per UAV for MASAC and MATD3.
2. One shared team twin-critic pair -> two centralized critics per agent.
3. MATD3 shared target actor -> one target actor per UAV.
4. LeakyReLU -> ReLU.
5. Explicit custom Kaiming initialization -> default Linear initialization.
6. Handcrafted actor belief centering -> direct normalized local information input.
7. Destination mask included inside actor feature vector -> mask used only as action availability.
8. MASAC two entropy temperatures -> one SAC temperature per agent over the factorized hybrid action.
9. MASAC automatic discrete target-entropy mechanism -> removed from baseline.
10. MATD3 epsilon-greedy discrete schedule -> removed from baseline.
11. Shared joint actor loss -> independent actor-i losses with other-agent actions detached.
12. Shared critic target -> agent-specific centralized twin-critic targets.

## What is unchanged

- Dec-POMDP environment dynamics.
- Six UAVs, targets, obstacles, sensing, belief updates, communications, energy model, reports, APF/safety, and termination logic.
- Cooperative team reward calculation.
- Replay-buffer off-policy training concept.
- gamma = 0.99 and tau = 0.005 unless an experiment configuration says otherwise.
- MATD3 twin critics, clipped target Q, target-policy smoothing, delayed actor updates, and Polyak targets.
- MASAC stochastic policy, twin soft critics, entropy regularization, replay buffer, and Polyak target critics.

## Correctness invariants

1. Changing hidden global state while holding eta_i fixed must not change actor_i output.
2. Changing another UAV's hidden/local memory outside observable contact information must not change actor_i input.
3. Critic inputs must include the centralized state and all agents' actions.
4. Each MATD3 agent owns exactly two critics and one target actor.
5. Each MASAC agent owns exactly two soft critics and no target actor.
6. Actor-i backward must not create gradients in actor-j for j != i.
7. A destination mask must never allow an illegal destination to be selected.
8. MASAC log probabilities/entropy and MATD3 relaxed discrete gradients must stay finite under extreme logits.
9. Same seed must reproduce initialization, sampled replay batches, updates, and final model fingerprint under the deterministic test configuration.
10. CPU and GPU environment observation/state contracts must match.

## Test strategy

TDD/refactor tests will cover:

- actor locality / no global-state leakage;
- independent actor parameter ownership;
- per-agent twin critic count and input dimensions;
- correct MASAC Bellman target and actor gradients;
- correct MATD3 twin-min target, target noise, policy delay, and target update timing;
- no cross-agent actor gradients;
- hybrid action legality and destination masking;
- finite masked categorical entropy/log-probability;
- deterministic seed regression for simple and UavNetSim backends;
- CPU training smoke for MASAC and MATD3;
- GPU/DDP contract tests after CPU tests pass;
- existing scenario, reward, networking, energy, APF, W&B, Kaggle pipeline, and world-generation tests.

## Paper wording

The baseline should be described as a cooperative Dec-POMDP solved under CTDE using MASAC and MATD3. State explicitly that decentralized actors use only local information states, while centralized critics use the global state and joint actions during training. State separately that the environment has a hybrid continuous-discrete action space, requiring a factorized Gaussian/categorical MASAC policy and a straight-through categorical relaxation for the discrete MATD3 branch.

Do not claim the hybrid-action handling is part of the original SAC/TD3 papers; label it as the minimum task-specific action-space adaptation.

## References used to anchor the design

- Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, ICML 2018.
- Lowe et al., Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, NeurIPS 2017 (CTDE/decentralized actors and centralized critics).
- Ackermann et al., Multi-Agent Twin Delayed Deep Deterministic Policy Gradient, arXiv:1910.01465.
- Dec-POMDP literature defining decentralized policies over local action-observation histories/information states.
- MASAC literature using decentralized stochastic actors and centralized soft critics; there is no single universally canonical MASAC implementation, so the baseline is deliberately defined as the direct SAC+CTDE extension above.
