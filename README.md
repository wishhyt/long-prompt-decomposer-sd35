# long-prompt-decomposer-sd35

This repository provides a minimal, runnable pipeline for `Stable Diffusion 3.5 Medium + PromptDecomposer (T5 branch)`.
After downloading model weights and a minimal dataset subset (COCO images), you can train and run inference directly.

## Repository Layout

```text
long-prompt-decomposer-sd35/
├── src/lpd/                    # Core library code (models, pipelines, utilities)
├── scripts/train/              # Training scripts
├── scripts/infer/              # Inference scripts
├── scripts/eval/               # Evaluation scripts
├── scripts/exp/                # Experiment wrappers
├── scripts/run_profile.py      # Unified profile-driven entry point
├── configs/                    # Accelerate config and profiles
└── docs/                       # Notes and documentation
```

## 1. Training Checklist (Confirmed)

1. Training entry and task type
- Main runnable entry: `scripts/train/lpd_ella_t5_sd3.py`.
- Train target: decomposer only (`TextDecomposer -> PromptResampler`), not full UNet/LoRA finetuning.
- Branch: SD3 (`src/lpd/pipeline_lpd_sd3.py`).

2. Decomposer contract
- Implementation class: `TextDecomposer` in `scripts/train/lpd_ella_t5_sd3.py`.
- Internal module: `PromptResampler` in `src/lpd/models.py`.
- `num_components`: controlled by `--num_components` (default `4`).
- Input branch: `decomposer(encoder_hidden_states_t5)`.
- Output: list of length `num_components`; each element has shape `[B, num_tokens, hidden_dim]`.

3. Text features used in training
- Main decomposer input is T5 hidden states (`encoder_hidden_states_t5`).
- CLIP embeddings are still used in SD3 transformer conditioning via concatenation.
- In this pipeline, "CT5" means combined CLIP + T5 conditioning behavior, not a separate custom T5 model.

4. CFG and negative prompt behavior
- Training uses caption dropout (`--component_dropout`) to include empty-condition cases.
- Inference uses `[negative_prompt, prompt]` when CFG is enabled.
- Default negative prompt is `""`.

5. Attention map supervision
- No `attn_maps` supervision is used in this minimal training pipeline.

6. Freeze vs. trainable modules
- Frozen: `transformer`, `vae`, `text_encoder`, `text_encoder_2`, `text_encoder_3`.
- Trainable: decomposer only.

7. Dataset schema and input format
- Default metadata source: `luping-liu/LongAlign` (Hugging Face datasets, streaming by default).
- Important fields: `caption`, `path`, `source` (with fallback caption keys handled in code).
- Image resolution path: `--image_root + path`.
- Minimal setup recommendation: `--source_filter coco`.

8. Outputs and checkpoints
- Each checkpoint includes accelerate state + `model.safetensors` (decomposer weights).
- Final `model.safetensors` is also saved in `output_dir`.
- Inference can load either a checkpoint directory or a direct `.safetensors` path.

## 2. Key Fixes Included

- Removed private hard-coded data paths from training; replaced with CLI-driven data loading.
- Added explicit `model.safetensors` export for checkpoints and final model.
- Added configurable W&B project/run/entity options.
- Fixed logging robustness when specific losses are disabled.
- Added iterable dataset worker sharding fallback and safer default `--dataloader_num_workers=0`.
- Rewrote SD3 inference scripts to fully parameterized CLI (no hard-coded local paths).

## 3. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -U huggingface_hub
```

## 4. Download Models and Minimal Data

### 4.1 Download SD3.5 Medium

Make sure your Hugging Face account has accepted the model license first.

```bash
huggingface-cli login

mkdir -p /ABS_PATH/long-prompt-decomposer-sd35/weights/sd3.5-medium
huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
  --local-dir /ABS_PATH/long-prompt-decomposer-sd35/weights/sd3.5-medium
```

### 4.2 Download Minimal Training Images (COCO train2017)

LongAlign metadata rows with `source=coco` typically use paths like `coco2017/train2017/xxxx.jpg`.

```bash
mkdir -p /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_images/coco2017
curl -L -o /ABS_PATH/long-prompt-decomposer-sd35/data/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip -q /ABS_PATH/long-prompt-decomposer-sd35/data/train2017.zip -d /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_images/coco2017
rm /ABS_PATH/long-prompt-decomposer-sd35/data/train2017.zip
```

Optional: localize LongAlign metadata (instead of streaming from Hub):

```bash
mkdir -p /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_meta
huggingface-cli download --repo-type dataset luping-liu/LongAlign \
  --local-dir /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_meta
```

### 4.3 Optional: Download DetailMaster (for eval/reproduction)

```bash
mkdir -p /ABS_PATH/long-prompt-decomposer-sd35/data/detailmaster
huggingface-cli download --repo-type dataset datajuicer/DetailMaster \
  --local-dir /ABS_PATH/long-prompt-decomposer-sd35/data/detailmaster
```

## 5. Path Arguments You Need

Core required paths:

- Model path: `--pretrained_model_name_or_path /ABS_PATH/long-prompt-decomposer-sd35/weights/sd3.5-medium`
- Image root path: `--image_root /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_images`

Metadata source defaults to `luping-liu/LongAlign`.
If you downloaded local metadata parquet files:

- `--dataset_name /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_meta`
- `--no_streaming`

## 6. Minimal Training Command (with W&B)

```bash
cd /ABS_PATH/long-prompt-decomposer-sd35

wandb login

accelerate launch --config_file configs/acc_config.yaml scripts/train/lpd_ella_t5_sd3.py \
  --pretrained_model_name_or_path /ABS_PATH/long-prompt-decomposer-sd35/weights/sd3.5-medium \
  --output_dir /ABS_PATH/long-prompt-decomposer-sd35/logs/lpd_sd35_minimal \
  --image_root /ABS_PATH/long-prompt-decomposer-sd35/data/longalign_images \
  --source_filter coco \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_train_steps 2000 \
  --checkpointing_steps 200 \
  --validation_steps 200 \
  --num_components 4 \
  --num_tokens 128 \
  --decomposer_heads 32 \
  --decomposer_layers 6 \
  --token_length 512 \
  --report_to wandb \
  --tracker_project_name lpd_sd3 \
  --wandb_run_name sd35_coco_minimal \
  --wandb_entity <YOUR_WANDB_ENTITY> \
  --dataloader_num_workers 0 \
  --mixed_precision bf16
```

Notes:
- For single-GPU training, set `num_processes: 1` in `configs/acc_config.yaml`.
- Default behavior uses `dataset_name=luping-liu/LongAlign` with streaming enabled.

## 7. Inference Command (Load Trained Decomposer)

```bash
python scripts/infer/lpd_sd3_infer.py \
  --pretrained_model_name_or_path /ABS_PATH/long-prompt-decomposer-sd35/weights/sd3.5-medium \
  --decomposer_ckpt /ABS_PATH/long-prompt-decomposer-sd35/logs/lpd_sd35_minimal/checkpoint-2000 \
  --prompt "A cinematic long prompt ..." \
  --negative_prompt "" \
  --output_dir /ABS_PATH/long-prompt-decomposer-sd35/output/infer \
  --save_decompose \
  --num_inference_steps 20 \
  --guidance_scale 4.5 \
  --max_sequence_length 512 \
  --dtype bf16 \
  --device cuda
```

Outputs:
- `compose.png`
- `component_0.png ... component_{N-1}.png` (if `--save_decompose` is enabled)

## 8. W&B Tracking Contents

During training, logs include:
- Scalars: `loss`, `diff` (if enabled), `kd` (if enabled), `lr`
- Validation images: composed output + per-component outputs
- Full run config from CLI args

## 9. End-to-End Minimal Workflow

1. Create environment and install dependencies.
2. `huggingface-cli login`, then download `stable-diffusion-3.5-medium`.
3. Download and extract COCO train2017 to `data/longalign_images/coco2017/train2017`.
4. `wandb login`.
5. Run the training command in Section 6.
6. Run the inference command in Section 7 and inspect generated outputs.

## 10. Optional Profile Entry

```bash
python scripts/run_profile.py --profile configs/profiles/lpd_ella_t5_sd3.json -- --help
```
