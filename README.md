# famtp-isaaclab

External Isaac Lab project for FaMTP research using **Direct workflow** + **RSL-RL**.
This stage implements **FaMTP Stage 1**:
- part-wise manifold encoders,
- latent part priors,
- global coupling prior,
- no bridge generation yet.

## Installation

```bash
pip install -e source/famtp_lab
```

## 1) Build no-transition dataset

```bash
python scripts/build_no_transition_dataset.py \
  --input datasets/motion_index.json \
  --output datasets/no_transition_motion_index.json \
  --summary datasets/no_transition_summary.json \
  --boundary-window-s 0.25
```

## 2) Train part encoders (Stage 1 pretraining)

```bash
python scripts/train_part_encoders.py \
  --steps 1000 \
  --batch-size 64 \
  --history-steps 4 \
  --residual-dim 2
```

Outputs:
- `outputs/checkpoints/part_encoders/`
- `outputs/logs/part_encoders/`

## 3) Train RL baselines

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

Train `famtp_stage1`:

```bash
python scripts/rsl_rl/train.py \
  --task FaMTP-Humanoid-Switch-Direct-v0 \
  --prior-mode famtp_stage1 \
  --chain-mode random \
  --use-manifold-encoder 1 \
  --use-latent-part-priors 1 \
  --use-global-coupling 1 \
  --latent-history-steps 4 \
  --latent-dim-residual 2 \
  --headless
```

## 4) Run FaMTP Stage 1 ablations

Disable latent part priors:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode famtp_stage1 --use-latent-part-priors 0 --use-global-coupling 1 --use-manifold-encoder 1 --headless
```

Disable global coupling:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode famtp_stage1 --use-latent-part-priors 1 --use-global-coupling 0 --use-manifold-encoder 1 --headless
```

Disable manifold encoder:

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode famtp_stage1 --use-manifold-encoder 0 --use-latent-part-priors 0 --use-global-coupling 0 --headless
```

## 5) Evaluate methods with shared switch protocol

Single-skill:

```bash
python scripts/eval_switching.py --experiment-name week1_single_skill --mode single_skill --episodes 20
```

Random-switch:

```bash
python scripts/eval_switching.py --experiment-name week1_random_switch --mode random_switch --episodes 20
```

Evaluate only `famtp_stage1`:

```bash
python scripts/eval_switching.py --experiment-name stage1_random_switch --mode random_switch --method famtp_stage1 --episodes 20
```

## 6) Plotting and table exports

```bash
python scripts/plot_baseline_comparison.py --single-exp week1_single_skill --switch-exp week1_random_switch
python scripts/export_baseline_tables.py --single-exp week1_single_skill --switch-exp week1_random_switch
python scripts/plot_famtp_stage1_latents.py --experiment-name week1_random_switch
python scripts/plot_reward_decomposition.py --experiment-name week1_random_switch
```

Outputs:
- `outputs/plots/baseline_comparison/`
- `outputs/tables/baseline_comparison/`
- `outputs/plots/famtp_stage1_latents/`

## Stage-2 attachment point (no bridge implemented yet)

Stage 2 bridge generation will attach in `HumanoidSwitchEnv._get_rewards()` under `prior_mode == "famtp_full"`.
At that point, bridge-generated latent transitions will feed additional transition-specific rewards and/or
latent-target constraints. Stage 1 currently runs independently without any bridge logic.

## Tests

```bash
pytest -q tests/test_imports.py tests/test_registry.py tests/test_env_smoke.py tests/test_discriminator_shapes.py tests/test_expert_buffer.py tests/test_prior_modes_smoke.py tests/test_manifold_stage1_shapes.py
```
