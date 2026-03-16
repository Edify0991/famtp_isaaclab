# famtp-isaaclab

External Isaac Lab research project for **FaMTP (Factorized Manifold Transition Priors)**.
This stage compares baseline priors in one codebase using a shared no-transition dataset protocol
and shared switch-window evaluation protocol.

## Installation (editable)

```bash
pip install -e source/famtp_lab
```

## Build no-transition dataset

```bash
python scripts/build_no_transition_dataset.py \
  --input datasets/motion_index.json \
  --output datasets/no_transition_motion_index.json \
  --summary datasets/no_transition_summary.json \
  --boundary-window-s 0.25
```

## Train baselines (RSL-RL, Direct workflow)

`prior_mode` options:
- `ppo_cmd`
- `fullbody_amp`
- `partwise_raw`
- `famtp_stage1` (reserved hook)
- `famtp_full` (reserved hook)

Train `ppo_cmd`:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode ppo_cmd --chain-mode random --headless
```

Train `fullbody_amp`:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode fullbody_amp --chain-mode random --headless
```

Train `partwise_raw`:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode partwise_raw --chain-mode random --headless
```

## Evaluate baselines with shared switch protocol

Single-skill run:

```bash
python scripts/eval_switching.py --experiment-name week1_single_skill --mode single_skill --episodes 20
```

Random-switch run:

```bash
python scripts/eval_switching.py --experiment-name week1_random_switch --mode random_switch --episodes 20
```

Per-method run (optional):

```bash
python scripts/eval_switching.py --experiment-name week1_random_switch --mode random_switch --method fullbody_amp --episodes 20
```

## Plot and table exports

Baseline comparison plots:

```bash
python scripts/plot_baseline_comparison.py --single-exp week1_single_skill --switch-exp week1_random_switch
```

Baseline comparison tables:

```bash
python scripts/export_baseline_tables.py --single-exp week1_single_skill --switch-exp week1_random_switch
```

Outputs:
- plots: `outputs/plots/baseline_comparison/`
- tables: `outputs/tables/baseline_comparison/`

## Week-1 report workflow

```bash
python scripts/plot_week1_figures.py \
  --dataset-summary datasets/no_transition_summary.json \
  --single-skill-exp week1_single_skill \
  --random-switch-exp week1_random_switch

week1
Current stage focuses on **week-1 feasibility**: proving skill-switching difficulty is real and measurable
before adding AMP/FaMTP components.

## Project goals (week 1)

- Build a no-transition motion dataset index (within-skill only).
- Train a simple command-conditioned baseline policy (`ppo_cmd`).
- Compare single-skill vs random-switch rollouts.
- Quantify degradation with switch-window metrics.
External Isaac Lab research project for **FaMTP (Factorized Manifold Transition Priors)** with a
Direct workflow humanoid skill-switching environment.

## Project goals

- Zero-transition humanoid multi-skill composition experiments.
- Isaac Lab **DirectRLEnv** workflow (not manager-based).
- RSL-RL as primary RL backend.

## Installation (editable)

```bash
pip install -e source/famtp_lab
```

## Core commands

List registered FaMTP tasks:

```bash
python scripts/list_envs.py
```

Build no-transition dataset index:

```bash
python scripts/build_no_transition_dataset.py \
  --input datasets/motion_index.json \
  --output datasets/no_transition_motion_index.json \
  --summary datasets/no_transition_summary.json \
  --boundary-window-s 0.25
```

Train week-1 baseline (`ppo_cmd`):

```bash
python scripts/rsl_rl/train.py \
  --task FaMTP-Humanoid-Switch-Direct-v0 \
  --baseline-mode ppo_cmd \
  --chain-mode random \
  --headless
```

Play a checkpoint:

```bash
python scripts/rsl_rl/play.py --task FaMTP-Humanoid-Switch-Direct-v0 --checkpoint <path>
```

Evaluate single-skill condition:

```bash
python scripts/eval_switching.py --experiment-name week1_single_skill --mode single_skill --episodes 20
```

Evaluate random-switch condition:

```bash
python scripts/eval_switching.py --experiment-name week1_random_switch --mode random_switch --episodes 20
```

Generate week-1 figures:

```bash
python scripts/plot_week1_figures.py \
  --dataset-summary datasets/no_transition_summary.json \
  --single-skill-exp week1_single_skill \
  --random-switch-exp week1_random_switch
```

Generate week-1 markdown report:

```bash
python scripts/week1_feasibility_report.py \
  --dataset-summary datasets/no_transition_summary.json \
  --single-skill-exp week1_single_skill \
  --random-switch-exp week1_random_switch
```

## Notes on optional tracking_oracle baseline

A true short-horizon tracking oracle needs direct motion-state tracking targets integrated into the Isaac Lab
articulation state and reference clip playback pipeline. That integration is intentionally deferred to avoid
adding a half-implemented baseline. The current code keeps stable hooks (`prior_mode`, modular reward terms,
expert buffer utilities) so `tracking_oracle` can be added cleanly in a later stage.

## Tests

```bash
pytest -q tests/test_imports.py tests/test_registry.py tests/test_env_smoke.py tests/test_discriminator_shapes.py tests/test_expert_buffer.py tests/test_prior_modes_smoke.py
```
Run smoke tests:

```bash
pytest -q tests/test_imports.py tests/test_registry.py tests/test_env_smoke.py
```

## Directory overview

- `source/famtp_lab/famtp_lab/motion/`: motion index loading, filtering, and clip sampling.
- `source/famtp_lab/famtp_lab/tasks/direct/humanoid_switch/`: DirectRLEnv task and switch protocol.
- `source/famtp_lab/famtp_lab/agents/rsl_rl/`: week-1 PPO config and wrappers.
- `scripts/eval_switching.py`: switch-window metric evaluation outputs.
- `scripts/plot_week1_figures.py`: generates Figures 1–5.
- `scripts/week1_feasibility_report.py`: generates `week1_summary.md`.

## Week-1 architecture and future hooks

1. `motion/` modules isolate dataset protocol and are designed for later AMP dataset readers.
2. `policy_mode="ppo_cmd"` in env config is the baseline switch; future modes can add AMP/FaMTP
   reward/prior hooks without changing task registration.
3. `metrics.py` and `eval_switching.py` define switch-window outputs that will remain compatible
   when imitation and manifold priors are introduced.
4. `scripts/rsl_rl/train.py` includes baseline-mode hooks where future FaMTP model configs can be injected.
- `scripts/`: CLI utilities for listing tasks and RSL-RL workflows.
- `source/famtp_lab/famtp_lab/tasks/direct/humanoid_switch/`: environment registration, config, and core logic.
- `source/famtp_lab/famtp_lab/agents/rsl_rl/`: minimal PPO config and wrappers for RSL-RL.
- `source/famtp_lab/famtp_lab/utils/`: logging and Gym registry helpers.
- `tests/`: import, registry, and env smoke tests.

## How registration, env config, and training connect

1. Importing `famtp_lab.tasks` executes task registration in
   `tasks/direct/humanoid_switch/__init__.py` and registers
   `FaMTP-Humanoid-Switch-Direct-v0` with Gymnasium.
2. Gym entry point resolves to `HumanoidSwitchEnv` and uses `HumanoidSwitchEnvCfg`.
3. RSL-RL scripts call `gym.make(task_id)`, wrap observations for policy input, load PPO config,
   and create an `OnPolicyRunner` for train/play/evaluate workflows.
