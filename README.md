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
