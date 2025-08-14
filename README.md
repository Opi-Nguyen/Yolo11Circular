# YOLOv11-Circular — Training README

This repo trains a circular object detector (center cx, cy, radius r, + class) using a YOLOv11-style backbone/neck and a custom circle head. It remains backward-compatible with legacy 7×7×C labels via an internal adapter (no data rewrite needed).

## Project layout
```
├── dataloader/
│   └── CircleYOLOv11Dataloader.py
├── models/
│   └── yolo11_circular.py
├── utils/
│   ├── map_utilities.py
│   └── yolo_training.py
├── configs/
│   ├── default.yaml
│   └── autocfg.py          # config loader utilities (used by train.py)
└── train.py                # CLI training script (this is what you run)
```

## Requirements

- Python 3.9–3.12
- PyTorch 2.x + CUDA (for GPU training)
- torchvision
- pyyaml
- numpy, opencv-python (likely needed by your dataloader)
- (Optional) A100/RTX30xx/RTX40xx for TF32/AMP speedups

### Install (example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA version
pip install pyyaml numpy opencv-python
```

## Config system

Training uses config defaults → optional YAML override → CLI overwrite (highest priority).

- Default config: `configs/default.yaml`
- Optional override via `--cfg path/to/your.yaml`
- Any CLI flag (e.g. `--img-size 640`) will overwrite both defaults and overrides
- Each run creates a unique directory: `runs/<session>` (or `runs/<session>_1`, `_2`, … if it already exists)
- The final merged config is saved to `runs/<session>/config.yaml` so you can reproduce or reuse for inference later

All outputs always live under the run folder:
```
runs/<session>/best.pth
runs/<session>/ckpt.pth
runs/<session>/train_log.csv
runs/<session>/config.yaml
```

Key legacy-mapping flags if your labels are 7×7×C:

```
--legacy-cx 0 --legacy-cy 1 --legacy-r 2 --legacy-obj 3 --legacy-cls 4
```

Set `--legacy-obj -1` if there is no objectness channel, and `--legacy-cls -1` for single-class data.

## Quick start

### Quick training for BCCD

```bash
python train.py \
  --session bccd_dp \
  --data datasets/7_classes_BC_xyr_converted/7_class_bccd_circle.yaml \
  --model-size n --img-size 640 \
  --devices 0,1 --device cuda
```
### 1) Single-GPU examples

```bash
# Model size "n", 448x448 images, on GPU 0
python train.py \  
    --session bccd_y11n_448 \
    --data datasets/7_classes_BC_xyr_converted/7_class_bccd_circle.yaml \
    --model-size n \
    --img-size 448 \
    --device cuda:0 

# Model size "s", 640x640 images, with speed features enabled
python train.py \
    --session bccd_y11s_640_fast \
    --data /path/to/your.yaml 
    --model-size s \
    --img-size 640 \
    --device cuda:0 \
    --amp --tf32 --compile
```

### 2) Using a config file + CLI overrides
```bash
# Start from defaults and override a few things on the CLI
python train.py \
    --cfg configs/default.yaml \
    --session exp_from_yaml \
    --img-size 512 \
    --batch-size 16 \
    --lr 0.001
```

## Multi-GPU (single machine)

This script supports DataParallel (DP) out of the box (simple and zero changes to your environment). For best scaling, consider DDP later.

**DataParallel**
```bash
# Use GPUs 0 and 1 (note: set --device cuda, not cuda:0)
python train.py \
    --session exp_dp \
    --data /path/to/your.yaml \
    --model-size n \
    --img-size 640 \ 
    --devices 0,1 \
    --device cuda
```

**Notes:**

- `--devices 0,1` makes those GPUs visible and the script wraps the model with `nn.DataParallel`.
- Keep `--device cuda` (no index) when using DP so the wrapper can see all visible GPUs.
- Batch will be split across GPUs automatically. Increase your batch size accordingly if memory allows.
- DDP is not wired in this script yet. If you want it, we can add a `torchrun`/`DistributedSampler` patch.

## Outputs

Inside `runs/<session>` you’ll find:

- `best.pth` — best checkpoint by validation AP@0.5
- `ckpt.pth` — last checkpoint of the current epoch (rolling)
- `train_log.csv` — simple CSV with epoch,loss,mAP,lr
- `config.yaml` — the exact merged config used (defaults + overrides + CLI)

## Common flags (abbrev)

**Data & I/O:**
```
--data path/to/dataset.yaml
--batch-size, --workers, --session, --cfg
```

**Model & image:**
```
--model-size {n,s,m,l,x}, --num-classes
--img-size (square, e.g. 448 / 512 / 640)
```

**Speed:**
```
--amp (mixed precision), --tf32 (Ampere+), --compile (PyTorch 2.x)
```

**Optim & sched:**
```
--optimizer {adam,adamw,sgd}, --lr, --weight-decay, --momentum
--sched {none,step,cos}, --step-size, --gamma
```

**Eval:**
```
--eval-interval, --val-batches (e.g. 1 for quick eval, -1 for full)
--conf-thres, --iou-thres, --max-det
```

**Legacy mapping:**
```
--legacy-cx, --legacy-cy, --legacy-r, --legacy-obj, --legacy-cls, --img-size-hint
```

## Tips & troubleshooting

- **Data YAML:** ensure your dataloader expects your dataset YAML format (image/label roots etc.).
- **Classes:** set `--num-classes` to match your dataset. For single-class legacy labels set `--legacy-cls -1`.
- **Loss exploding or NaNs:** try a smaller LR, enable `--amp`, or reduce `--img-size`.
- **Too slow?** enable `--tf32` (Ampere+), `--amp`, `--compile`, raise `--workers`, and consider a larger batch on multi-GPU.
- **AP stays 0:** check legacy mapping (`--legacy-*`) and `img_size_hint` if your labels are normalized 0..1.

## Reproducibility & inference

Every run saves a frozen config at `runs/<session>/config.yaml`. For inference, load:

- the `best.pth` from your session
- the saved `config.yaml` (for class count, input size, legacy mapping, etc.)

If you’d like, we can add a minimal `infer.py` that consumes the run folder directly:

```bash
python infer.py --run runs/your_session --images /path/to/images
```
