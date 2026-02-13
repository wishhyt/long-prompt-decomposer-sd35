# 文件地图（Final）

## 1) Core Library (`src/lpd`)

基础模型与工具：
- `src/lpd/model.py`
- `src/lpd/models.py`
- `src/lpd/tools.py`
- `src/lpd/loss.py`

Pipeline：
- `src/lpd/pipeline_prompt_decomposition.py`（主实现）
- `src/lpd/pipeline_lpd_ct5.py`（薄封装，复用主实现）
- `src/lpd/pipeline_prompt_split.py`
- `src/lpd/pipeline_lpd_sd3.py`
- `src/lpd/pipeline_lpd_sdxl.py`

子模块：
- `src/lpd/modules/adapters.py`
- `src/lpd/modules/lora.py`
- `src/lpd/attention_map_diffusers/__init__.py`
- `src/lpd/attention_map_diffusers/modules.py`
- `src/lpd/attention_map_diffusers/utils.py`

## 2) Train Scripts (`scripts/train`)

- `scripts/train/long_prompt_decomposition.py`
- `scripts/train/lpd_ella.py`
- `scripts/train/lpd_ella_ct5.py`
- `scripts/train/lpd_ella_sd3.py`
- `scripts/train/lpd_ella_sdxl.py`
- `scripts/train/lpd_ella_t5_sd3.py`
- `scripts/train/lpd_reward.py`
- `scripts/train/lpd_reward_sd3.py`
- `scripts/train/unet_align_ct5.py`
- `scripts/train/unet_lcm.py`
- `scripts/train/unet_reward.py`
- `scripts/train/unet_reward_lcm.py`

## 3) Inference Scripts (`scripts/infer`)

- `scripts/infer/infer.py`
- `scripts/infer/ct5_infer.py`
- `scripts/infer/lpd_sd3_infer.py`
- `scripts/infer/sd1_sample.py`
- `scripts/infer/sd3_infer.py`
- `scripts/infer/sample.py`
- `scripts/infer/sample_ella.py`
- `scripts/infer/sample_longalign.py`
- `scripts/infer/sample_pipeline.py`
- `scripts/infer/sample_pipeline_ella.py`
- `scripts/infer/sample_pipeline_ella_ct5.py`
- `scripts/infer/sample_prompt_split.py`
- `scripts/infer/vis_attn_maps.py`

## 4) Eval Scripts (`scripts/eval`)

- `scripts/eval/denscore.py`
- `scripts/eval/show_attn.py`
- `scripts/eval/test.py`

## 5) Experiment Entry (`scripts/exp`)

- `scripts/exp/dev.py`（profile 封装）
- `scripts/exp/data_test.py`（profile 封装）
- `scripts/exp/llm4gen.py`
- `scripts/exp/shear.py`

## 6) Profile Runner

- `scripts/run_profile.py`：统一主入口（根据 profile 启动目标脚本）

配置文件：
- `configs/profiles/dev.json`
- `configs/profiles/data_test.json`
- `configs/profiles/lpd_ella_ct5.json`

## 7) Repo Meta

- `configs/acc_config.yaml`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `LICENSE`

