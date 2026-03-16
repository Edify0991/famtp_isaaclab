# famtp-isaaclab

Isaac Lab external project for Unitree **G1** motion dataset inspection and week-1 dataset entry building.

## Week-1 motion viewing and dataset pipeline

External Isaac Lab project for FaMTP humanoid skill switching (Direct workflow + RSL-RL).
This version includes **FaMTP full** with an explicit latent bridge generator and ablation tooling.
External Isaac Lab project for FaMTP research using **Direct workflow** + **RSL-RL**.
This stage implements **FaMTP Stage 1**:
- part-wise manifold encoders,
- latent part priors,
- global coupling prior,
- no bridge generation yet.
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

### 1) 扫描 G1 数据集
```bash
python scripts/inspect_g1_amass_dataset.py --data-root <path>
```

### 2) 启动 viewer（随机浏览）
```bash
python scripts/view_g1_motion_dataset.py --data-root <path>
```
pip install -e source/famtp_lab
```

## Full pipeline commands

### 1) Build no-transition dataset

### 3) 启动 viewer（指定 clip）
```bash
python scripts/view_g1_motion_dataset.py --clip <path/to/file.npz>
```

### 4) viewer 仅浏览某个 skill
```bash
python scripts/view_g1_motion_dataset.py --motion-index datasets/canonical/week1_motion_index.json --skill dribble_like
```

### 5) 挖 week1 skill
```bash
python scripts/mine_week1_skills_from_babel.py --babel-json <path> --g1-root <path>
```

### 6) 构建 week1 motion index
```bash
python scripts/build_week1_motion_index.py --input datasets/canonical/week1_candidate_motion_index.json
```

### 7) 构建 no-transition 数据集
```bash
python scripts/build_no_transition_dataset.py --input datasets/canonical/week1_motion_index.json
```

### 8) 生成图表
```bash
python scripts/plot_week1_dataset_figures.py --motion-index datasets/canonical/week1_motion_index.json
```

### 9) 生成 markdown 周报
```bash
python scripts/write_week1_dataset_report.py --report-root outputs/reports/week1_feasibility
```

## Viewer controls

Main path: Isaac Lab standalone app loop + terminal command polling.
Fallback path: terminal-only command loop (for environments where graphical key events are unavailable).

Commands:
- `space`: 播放/暂停
- `left` / `right`: 单帧后退/前进
- `up` / `down`: 上一个/下一个 clip
- `shift+left` / `shift+right`: 快退/快进 10 帧
- `[` / `]`: 降低/提高播放速度
- `l`: 循环播放开关
- `r`: 重置 clip
- `g`: 随机 clip
- `i`: overlay 开关
- `k`: 保存当前帧截图
- `p`: 导出关键帧条带图
- `t`: 导出 root trajectory 图
- `q` / `esc`: 退出

Outputs:
- `outputs/viewer_screenshots/`
- `outputs/viewer_keyframes/`
- `outputs/viewer_plots/`

## Scope note

当前阶段只做：
- G1 重定向动作数据可视化
- schema 检查/诊断
- 第一周技能候选挖掘与数据入口构建
- 周报素材导出

不包含 AMP、FaMTP、bridge 或高级训练逻辑。
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
---
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
-----------------------
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
