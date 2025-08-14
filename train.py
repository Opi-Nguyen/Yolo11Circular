
# -*- coding: utf-8 -*-
"""
Training script using config defaults (configs/default.yaml) + optional override YAML + CLI overwrite.
Each training run writes outputs under runs/<session_name>_k and saves the final merged config as config.yaml there.
"""
import os
import csv
import math
import argparse
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn as nn

from utils import yolo_training
from utils.map_utilities import get_bboxes_v11, mean_average_precision as mAP
from dataloader.CircleYOLOv11Dataloader import load_circle_yolo11_datasets as loader

from models.yolo11_circular import YOLOv11Circular, DEFAULT_CFGS, CircleDetectionLoss
from configs.autocfg import load_config, create_unique_folder, _in_run_dir


def parse_args():
    p = argparse.ArgumentParser("Train YOLOv11-Circular", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Config / session
    p.add_argument("--cfg", type=str, default=None, help="Optional override YAML (merged on top of configs/default.yaml)")
    p.add_argument("--session", type=str, default=None, help="Training session name (overrides config.train_session_name)")

    # Allow overriding any common hyper-parameters from CLI (optional)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)

    p.add_argument("--model-size", type=str, default=None, choices=["n","s","m","l","x"])
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    p.add_argument("--tf32", action="store_true", help="Enable TF32")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32")

    p.add_argument("--optimizer", type=str, default=None, choices=["adam","adamw","sgd"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="Enable AMP")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument("--sched", type=str, default=None, choices=["none","step","cos"])
    p.add_argument("--step-size", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--no-aug", action="store_true", help="Disable ColorJitter augmentation")

    p.add_argument("--l1", type=float, default=None)
    p.add_argument("--legacy-cx", type=int, default=None)
    p.add_argument("--legacy-cy", type=int, default=None)
    p.add_argument("--legacy-r",  type=int, default=None)
    p.add_argument("--legacy-obj", type=int, default=None)
    p.add_argument("--legacy-cls", type=int, default=None)
    p.add_argument("--img-size-hint", type=int, default=None)

    p.add_argument("--conf-thres", type=float, default=None)
    p.add_argument("--iou-thres",  type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--max-boxes-per-image", type=int, default=None)

    p.add_argument("--early-stop-ap", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--devices", type=str, default=None, help="Comma-separated GPU ids for DataParallel, e.g. '0,1,2'")

    return p.parse_args()


def build_transforms(img_size: int, no_aug: bool):
    ops = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(img_size, img_size), antialias=True, interpolation=3),
    ]
    if not no_aug:
        ops.append(v2.ColorJitter(brightness=(0.75, 1.5), contrast=(0.5, 1.5),
                                  saturation=(0.5, 1.5), hue=(-0.25, 0.25)))
    return v2.Compose(ops)


def build_optimizer(params, name: str, lr: float, wd: float, momentum: float):
    name = (name or "adamw").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
    raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(optimizer, sched: str, epochs: int, step_size: int, gamma: float):
    if sched == "none":
        return None
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler {sched}")


def eval_one_pass_v11(model, val_loader, device, legacy_idx, img_size_hint,
                      conf_thres, iou_thres, max_det, max_batches, max_boxes_per_image):
    model.eval()
    all_pred, all_true = [], []
    n_batches = 0
    with torch.no_grad():
        for bidx, (images, targets_legacy) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            preds_eval = model(images)  # [B, 3+nc, A]

            pred_box, true_box = get_bboxes_v11(
                preds_eval     = preds_eval.detach().cpu(),
                targets_legacy = targets_legacy,
                legacy_idx     = legacy_idx,
                img_size_hint  = img_size_hint or images.shape[-1],
                conf_thres     = conf_thres,
                iou_thres      = iou_thres,
                max_det        = max_det,
            )

            if max_boxes_per_image:
                from collections import defaultdict
                per_img_pred = defaultdict(list)
                for item in pred_box:
                    per_img_pred[item[0]].append(item)
                clipped = []
                for k, items in per_img_pred.items():
                    clipped.extend(items[:max_boxes_per_image])
                pred_box = clipped

            all_pred.extend(pred_box)
            all_true.extend(true_box)
            n_batches += 1
            if (max_batches is not None) and (max_batches > 0) and (n_batches >= max_batches):
                break
    return all_pred, all_true


def main():
    args = parse_args()

    # Load configs/default.yaml then override with optional --cfg and finally overwrite with CLI flags
    DEFAULT_CFG = "configs/default.yaml"
    cfg = load_config(DEFAULT_CFG, args.cfg, args)

    # Session / run directory
    session_name = args.session or cfg.get("train_session_name", "exp")
    run_dir = create_unique_folder(os.path.join("runs", session_name))

    # Persist final config (and ensure save paths are under run_dir)
    cfg.set("save_model", os.path.join(run_dir, cfg.get("save_model", "best.pth")))
    cfg.set("save_ckpt",  os.path.join(run_dir, cfg.get("save_ckpt",  "ckpt.pth")))
    cfg.set("csv_log",    os.path.join(run_dir, cfg.get("csv_log",    "train_log.csv")))
    cfg.set("run_dir",    run_dir)

    # Save the merged config snapshot for reproducibility
    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg.save(cfg_path)
    print(f"🔧 Using config saved at: {cfg_path}")

    # TF32 (perf on Ampere+)
    if cfg.get("tf32", True):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    torch.manual_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))

    # Transforms & Datasets
    transform = build_transforms(int(cfg.get("img_size", 448)), bool(cfg.get("no_aug", False)))
    dataset_train, dataset_val, dataset_test = loader(cfg.get("data"), transform=transform, img_size=int(cfg.get("img_size", 448)))

    workers = cfg.get("workers")
    if workers in (None, "None", "none"):
        workers = os.cpu_count()
    else:
        workers = int(workers)

    train_loader = DataLoader(dataset_train, batch_size=int(cfg.get("batch_size", 8)), shuffle=True,  num_workers=workers)
    val_loader   = DataLoader(dataset_val,   batch_size=int(cfg.get("batch_size", 8)), shuffle=False, num_workers=workers)

    # Model
    cfg_key = str(cfg.get("model_size", "n")).lower()
    model = YOLOv11Circular(DEFAULT_CFGS[cfg_key], num_classes=int(cfg.get("num_classes", 1))).to(device)
    
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        # dùng 'cuda' thay vì 'cuda:0' để DP thấy hết GPU visible
        model = model.to("cuda")
        model = nn.DataParallel(model)

    compile_flag = cfg.get("compile", False)
    if (args.no_compile):
        compile_flag = False
    if args.compile:
        compile_flag = True
    if compile_flag and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Optim / Loss / Sched
    optimizer = build_optimizer(model.parameters(), cfg.get("optimizer", "adamw"),
                                float(cfg.get("lr", 5e-4)), float(cfg.get("weight_decay", 1e-4)),
                                float(cfg.get("momentum", 0.937)))

    legacy_idx = {
        'cx':  int(cfg.get("legacy_cx", 0)),
        'cy':  int(cfg.get("legacy_cy", 1)),
        'r':   int(cfg.get("legacy_r", 2)),
        'obj': None if int(cfg.get("legacy_obj", 3)) < 0 else int(cfg.get("legacy_obj", 3)),
        'cls': None if int(cfg.get("legacy_cls", 4)) < 0 else int(cfg.get("legacy_cls", 4)),
    }
    try:
        loss_fn = CircleDetectionLoss(num_classes=int(cfg.get("num_classes", 1)),
                                      l1_weight=float(cfg.get("l1", 1.0)),
                                      legacy_idx=legacy_idx,
                                      img_size_hint=int(cfg.get("img_size_hint", cfg.get("img_size", 448))))
    except TypeError:
        loss_fn = CircleDetectionLoss(num_classes=int(cfg.get("num_classes", 1)),
                                      l1_weight=float(cfg.get("l1", 1.0)))

    amp_flag = cfg.get("amp", True)
    if args.no_amp:
        amp_flag = False
    if args.amp:
        amp_flag = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp_flag)

    # Scheduler
    scheduler = None
    sched_name = cfg.get("sched", "cos")
    if sched_name and sched_name != "none":
        if sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.get("step_size", 50)),
                                                        gamma=float(cfg.get("gamma", 0.1)))
        elif sched_name == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 100)))
        else:
            scheduler = None

    # Checkpoint helper
    best_ckpt = yolo_training.ModelCheckpoint(filepath=cfg.get("save_model"), monitor='AP', mode='max')
    start_epoch = 1

    # Training loop
    train_time_start_model = timer()

    # CSV header
    if not os.path.isfile(cfg.get("csv_log")):
        with open(cfg.get("csv_log"), "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "mAP", "lr"])

    epochs = int(cfg.get("epochs", 100))
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch: {epoch:-<70}")

        # Train one epoch
        loss = yolo_training.train_step(model=model, data_loader=train_loader,
                                        yolo_loss_fn=loss_fn, scaler=scaler, optimizer=optimizer, device=device)
        print(f"Epoch loss: {loss:.4f}")

        # Step scheduler (per-epoch)
        if scheduler is not None:
            scheduler.step()

        # Eval interval
        do_eval = (epoch % int(cfg.get("eval_interval", 1)) == 0) or (epoch == epochs)
        if do_eval:
            pred_box, true_box = eval_one_pass_v11(
                model, val_loader, device,
                legacy_idx=legacy_idx,
                img_size_hint=int(cfg.get("img_size_hint", cfg.get("img_size", 448))),
                conf_thres=float(cfg.get("conf_thres", 0.25)),
                iou_thres=float(cfg.get("iou_thres", 0.50)),
                max_det=int(cfg.get("max_det", 300)),
                max_batches=(None if int(cfg.get("val_batches", 1)) == -1 else int(cfg.get("val_batches", 1))),
                max_boxes_per_image=int(cfg.get("max_boxes_per_image", 100)),
            )
            AP = mAP(pred_boxes=pred_box, true_boxes=true_box,
                     threshold_mAP=0.5, step_threshold=1, stop_threshold_mAP=0.95,
                     C=int(cfg.get("num_classes", 1)), epsilon=1e-6)
            print(f"Average Precision @50 (val): {(AP * 100):.2f}%")
            best_ckpt(AP, model)
        else:
            AP = float('nan')

        # Save running checkpoint
        total_train_time = yolo_training.print_train_time(start=train_time_start_model, end=timer(), device=device)
        save_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_time": total_train_time,
            "best_mAP": best_ckpt.get_mAP()
        }
        torch.save(save_payload, cfg.get("save_ckpt"))

        # Append CSV
        with open(cfg.get("csv_log"), "a", newline="") as f:
            lr_now = optimizer.param_groups[0]["lr"]
            csv.writer(f).writerow([epoch, float(loss), float(AP) if not math.isnan(AP) else "", float(lr_now)])

        # Early stop on AP
        if (not math.isnan(AP)) and (AP >= float(cfg.get("early_stop_ap", 0.95))):
            print(f"Early stopping at epoch {epoch}: AP@0.5 >= {cfg.get('early_stop_ap')}")
            break


if __name__ == "__main__":
    main()
