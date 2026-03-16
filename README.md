# famtp-isaaclab

External Isaac Lab research project for **FaMTP (Factorized Manifold Transition Priors)**.
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
