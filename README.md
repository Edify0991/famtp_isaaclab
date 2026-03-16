# famtp-isaaclab

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

## Quickstart commands

List registered FaMTP tasks:

```bash
python scripts/list_envs.py
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --headless
```

Play a checkpoint:

```bash
python scripts/rsl_rl/play.py --task FaMTP-Humanoid-Switch-Direct-v0 --checkpoint <path>
```

Evaluate a checkpoint:

```bash
python scripts/rsl_rl/evaluate.py --task FaMTP-Humanoid-Switch-Direct-v0 --checkpoint <path>
```

Run smoke tests:

```bash
pytest -q tests/test_imports.py tests/test_registry.py tests/test_env_smoke.py
```

## Directory overview

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
