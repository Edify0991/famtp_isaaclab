# famtp-isaaclab

External Isaac Lab project for FaMTP humanoid skill switching (Direct workflow + RSL-RL).
This version includes **FaMTP full** with an explicit latent bridge generator and ablation tooling.

## Installation

```bash
pip install -e source/famtp_lab
```

## Full pipeline commands

### 1) Build no-transition dataset

```bash
python scripts/build_no_transition_dataset.py \
  --input datasets/motion_index.json \
  --output datasets/no_transition_motion_index.json \
  --summary datasets/no_transition_summary.json \
  --boundary-window-s 0.25
```

### 2) Train ppo_cmd

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode ppo_cmd --chain-mode random --headless
```

### 3) Train fullbody_amp

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode fullbody_amp --chain-mode random --headless
```

### 4) Train partwise_raw

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode partwise_raw --chain-mode random --headless
```

### 5) Train famtp_stage1

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

### 6) Train bridge generator (FaMTP full component)

```bash
python scripts/train_bridge_generator.py \
  --steps 1000 \
  --batch-size 64 \
  --history-steps 4 \
  --bridge-horizon 12 \
  --latent-dim-total 20 \
  --use-target-anchor 1
```

### 7) Train famtp_full

```bash
python scripts/rsl_rl/train.py \
  --task FaMTP-Humanoid-Switch-Direct-v0 \
  --prior-mode famtp_full \
  --chain-mode random \
  --use-manifold-encoder 1 \
  --use-latent-part-priors 1 \
  --use-global-coupling 1 \
  --bridge-horizon-steps 12 \
  --bridge-replan-mode on_switch \
  --use-target-anchor 1 \
  --bridge-update-interval 2 \
  --headless
```

### 8) Train famtp_nobridge (full ablation baseline)

```bash
python scripts/rsl_rl/train.py --task FaMTP-Humanoid-Switch-Direct-v0 --prior-mode famtp_nobridge --chain-mode random --headless
```

### 9) Evaluate all methods

```bash
python scripts/eval_switching.py --experiment-name week1_single_skill --mode single_skill --episodes 20
python scripts/eval_switching.py --experiment-name week1_random_switch --mode random_switch --episodes 20
```

Batch mode (deterministic names + seeds):

```bash
python scripts/run_experiments.py --methods ppo_cmd fullbody_amp partwise_raw famtp_stage1 famtp_nobridge famtp_full --seeds 0 1 2 --max-iterations 300 --dry-run
python scripts/run_evaluations.py --methods ppo_cmd fullbody_amp partwise_raw famtp_stage1 famtp_nobridge famtp_full --seeds 0 1 2 --mode random_switch --episodes 20 --dry-run
```

### 10) Generate paper plots and tables

```bash
python scripts/plot_switch_metrics.py --single-exp week1_single_skill --switch-exp week1_random_switch
python scripts/plot_ablations.py --switch-exp week1_random_switch
python scripts/plot_representative_traces.py --switch-exp week1_random_switch
python scripts/plot_famtp_stage1_latents.py --experiment-name week1_random_switch
python scripts/plot_reward_decomposition.py --experiment-name week1_random_switch
python scripts/export_tables.py --single-exp week1_single_skill --switch-exp week1_random_switch --bootstrap-samples 1000
python scripts/make_paper_artifacts.py --single-exp week1_single_skill --switch-exp week1_random_switch --dry-run
```

## FaMTP full ablation variants

- `famtp_nobridge`: latent priors without bridge generator.
- `famtp_full`: latent priors + global coupling + bridge generator.
- `no_global_coupling`: set `--use-global-coupling 0`.
- `no_manifold_encoder`: set `--use-manifold-encoder 0`.
- `no_target_anchor`: set `--use-target-anchor 0`.
- `short_bridge`: `--bridge-horizon-steps 6`.
- `long_bridge`: `--bridge-horizon-steps 18`.

## Data flow (end-to-end)

1. `datasets/motion_index.json` → `scripts/build_no_transition_dataset.py` creates within-skill filtered index.
2. No-transition index drives expert/latent prior assumptions (no true transition labels).
3. Priors stack (`manifold_encoders`, `latent_part_discriminators`, `coupling`, `bridge_generator`) feeds env reward terms.
4. `HumanoidSwitchEnv` combines task reward + prior rewards + bridge rewards and emits observations for PPO.
5. RSL-RL PPO (`scripts/rsl_rl/train.py`) optimizes policies per method.
6. `scripts/eval_switching.py` exports switch metrics, per-switch metrics, latent states, reward decomposition fields.
7. Plot/table scripts produce publication artifacts in `outputs/plots/*` and `outputs/tables/*`.

## Stage-2 attachment note

Bridge generation is fully integrated for `famtp_full`. Future Stage-2 upgrades attach at:
- bridge model architecture in `source/famtp_lab/famtp_lab/priors/bridge_generator.py`
- bridge reward path in `HumanoidSwitchEnv._stage_latent_terms(use_bridge=True)`
- switch-time bridge invocation in `HumanoidSwitchEnv._make_bridge()`
without requiring changes to baseline modes.

## Tests

```bash
pytest -q tests/test_imports.py tests/test_registry.py tests/test_env_smoke.py tests/test_discriminator_shapes.py tests/test_expert_buffer.py tests/test_prior_modes_smoke.py tests/test_manifold_stage1_shapes.py tests/test_bridge_generator_shapes.py
```
