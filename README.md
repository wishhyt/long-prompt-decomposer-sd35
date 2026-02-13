# LPD (Long Prompt Decomposition)

LPD 训练/推理工程，覆盖 SD1/SDXL/SD3 分支，包含注意力可视化、LoRA 训练与评分脚本。

## 当前目录结构

```text
LPD/
├── src/lpd/                  # 核心库代码（模型、pipeline、工具）
├── scripts/train/            # 训练脚本
├── scripts/infer/            # 推理与采样脚本
├── scripts/eval/             # 评估与检查脚本
├── scripts/exp/              # 实验封装（已收敛到 profile 入口）
├── scripts/run_profile.py    # 统一主入口（profile 驱动）
├── configs/                  # 训练配置与 profile
└── docs/FILE_MAP.md          # 文件地图
```

## 已完成整理

1. 清理：删除 `__pycache__`/`.DS_Store` 并修复 `.gitignore`。
2. 迁移：核心库迁移到 `src/lpd`，脚本按 `train/infer/eval/exp` 分层。
3. 去重：
   - `src/lpd/pipeline_lpd_ct5.py` 收敛为对 `pipeline_prompt_decomposition` 的薄封装。
   - `scripts/exp/dev.py`、`scripts/exp/data_test.py` 收敛为 `scripts/run_profile.py + configs/profiles/*.json` 的配置化入口。

## 环境

```bash
pip install -r requirements.txt
```

多卡训练配置文件：`configs/acc_config.yaml`

## 运行方式

统一入口（推荐）：

```bash
python scripts/run_profile.py --profile configs/profiles/lpd_ella_ct5.json -- --help
python scripts/run_profile.py --profile configs/profiles/dev.json -- --help
```

说明：`--` 后的参数会透传给目标脚本。

直接运行脚本（旧方式，仍可用）：

```bash
python scripts/train/lpd_ella_ct5.py --help
python scripts/infer/sample_pipeline_ella_ct5.py --help
```

