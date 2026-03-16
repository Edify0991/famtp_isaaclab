# famtp-isaaclab

Isaac Lab external project for Unitree **G1** motion dataset inspection and week-1 dataset entry building.

## Week-1 motion viewing and dataset pipeline

### 1) 扫描 G1 数据集
```bash
python scripts/inspect_g1_amass_dataset.py --data-root <path>
```

### 2) 启动 viewer（随机浏览）
```bash
python scripts/view_g1_motion_dataset.py --data-root <path>
```

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
