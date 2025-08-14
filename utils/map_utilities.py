
import torch

# ===== YOLOv11 eval helpers =====
def _legacy_targets_to_xyxy_list(grid, legacy_idx, img_size_hint=448):
    """
    grid: Tensor [S,S,C] (legacy 7x7xC)
    legacy_idx: {'cx':int,'cy':int,'r':int,'obj':int or None,'cls':int or None}
    Trả: list [[cls_id, x1,y1,x2,y2], ...]  (không kèm batch index ở đây)
    """
    S1, S2, C = grid.shape
    assert S1 == S2, "Legacy grid must be SxSxC"
    cx_ch, cy_ch, r_ch = legacy_idx['cx'], legacy_idx['cy'], legacy_idx['r']
    obj_ch = legacy_idx.get('obj', None)
    cls_ch = legacy_idx.get('cls', None)

    cx = grid[..., cx_ch]
    cy = grid[..., cy_ch]
    r  = grid[..., r_ch]

    if obj_ch is None:
        obj = (r > 0)
    else:
        obj = grid[..., obj_ch] > 0.5

    ys, xs = (obj).nonzero(as_tuple=True)

    # detect normalized
    max_v = torch.stack([cx.max(), cy.max(), r.max()]).max()
    normalized = bool(max_v <= 1.5)
    if normalized:
        img_w = img_h = float(img_size_hint)
    else:
        img_w = img_h = None

    gts = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        cxv = float(cx[y, x].item())
        cyv = float(cy[y, x].item())
        rv  = float(r[y, x].item())
        if normalized:
            cxv *= img_w
            cyv *= img_h
            rv  *= img_w  # ảnh vuông → ok

        x1, y1, x2, y2 = cxv - rv, cyv - rv, cxv + rv, cyv + rv
        if cls_ch is not None:
            cls_id = int(round(float(grid[y, x, cls_ch].item())))
        else:
            cls_id = 0
        gts.append([cls_id, x1, y1, x2, y2])
    return gts


def get_bboxes_v11(preds_eval, targets_legacy, legacy_idx, img_size_hint=448,
                   conf_thres=0.25, iou_thres=0.50, max_det=300):
    """
    preds_eval: Tensor [B, 3+nc, A] (đầu ra YOLOv11 ở chế độ eval)
    targets_legacy: Tensor [B, S, S, C] hoặc List[T[S,S,C]]  (nhãn v1)
    Trả:
      pred_boxes: List[[b, cls, score, x1,y1,x2,y2], ...]
      true_boxes: List[[b, cls, x1,y1,x2,y2], ...]
    """
    # NOTE: use local models.* import (project layout)
    from models.yolo11_circular import postprocess_circular

    if isinstance(targets_legacy, torch.Tensor) and targets_legacy.dim() == 4:
        grids = [targets_legacy[b] for b in range(targets_legacy.shape[0])]
    elif isinstance(targets_legacy, list):
        grids = targets_legacy
    else:
        raise TypeError("targets_legacy must be Tensor[B,S,S,C] or List of [S,S,C]")

    B = preds_eval.shape[0]
    dets = postprocess_circular(preds_eval, conf_thres=conf_thres,
                                iou_thres=iou_thres, max_det=max_det)

    pred_boxes, true_boxes = [], []

    for b in range(B):
        # predictions
        db = dets[b]  # [N,6] = xyxy, score, cls
        if db.numel() > 0:
            for row in db:
                x1, y1, x2, y2, score, cls = row.tolist()
                pred_boxes.append([b, int(cls), float(score), x1, y1, x2, y2])

        # ground-truth (convert legacy grid -> xyxy)
        gts = _legacy_targets_to_xyxy_list(grids[b], legacy_idx, img_size_hint=img_size_hint)
        for (cls_id, x1, y1, x2, y2) in gts:
            true_boxes.append([b, int(cls_id), x1, y1, x2, y2])

    return pred_boxes, true_boxes


# ==== Below here is the existing legacy YOLOv1 mAP utilities (kept for compatibility) ====
from utils.circle_intersection_over_union import intersection_over_union, intersection_over_union_batch

def non_max_suppression(detections, iou_threshold=0.5, threshold = 0.5):
    import torch as _torch
    detections_tensor = _torch.tensor(detections)
    detections_tensor = detections_tensor[detections_tensor[:, 1] > threshold]
    if len(detections_tensor) == 0:
        return []
    sorted_indices = _torch.argsort(detections_tensor[:, 1], descending=True)
    detections_tensor = detections_tensor[sorted_indices]
    selected_detections = []
    while len(detections_tensor) > 0:
        chosen_detection = detections_tensor[0]
        selected_detections.append(chosen_detection.tolist())
        ious = intersection_over_union_batch(
            pred=chosen_detection[2:].to(detections_tensor.device),
            labels=detections_tensor[:, 2:]
        )
        overlapping_indices = ious >= iou_threshold
        detections_tensor = detections_tensor[~overlapping_indices]
    return selected_detections


def convert_grid_boxes(predictions, device = 'cuda', S=7, C=1):
    batch_size = predictions.shape[0]
    shape_size_loop = int((predictions.shape[-1] - C)/4)
    for bbox in range(1, shape_size_loop, 1):
        mask = predictions[..., C:(C+1)] < predictions[..., (C+4*bbox):(C+1+4*bbox)]
        bbox_slice = slice(C + 4 * bbox, C + 4 * bbox + 4)
        predictions[..., C:(C+4)] = torch.where(mask, predictions[..., bbox_slice], predictions[..., C:(C+4)])
    prob = (predictions[...,C:(C+1)])
    cell_indicies = torch.arange(S).to(device).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1/S * (predictions[...,(C+1):(C+2)] + cell_indicies)
    y = 1/S * (predictions[...,(C+2):(C+3)] + cell_indicies.permute(0, 2, 1, 3))
    r = predictions[..., (C+3):(C+4)]
    if C < 2:
        class_idx = torch.zeros_like(r)
    else:
        class_idx = predictions[..., 0:C].argmax(-1).unsqueeze(-1)
    converted_boxes = torch.cat((class_idx, prob, x, y, r), dim=-1)
    return converted_boxes

def grid_boxes_to_boxes(grid_boxes, device='cuda', S: int = 7, C: int = 1):
    converted_boxes = convert_grid_boxes(grid_boxes, device=device, S=S, C=C).reshape(grid_boxes.shape[0], S*S, -1)
    converted_boxes[..., 0] = converted_boxes[..., 0].long()
    all_boxes = []
    for batch_idx in range(grid_boxes.shape[0]):
        boxes = []
        for box_idx in range(S*S):
            boxes.append([x.item() for x in converted_boxes[batch_idx, box_idx, :]])
        all_boxes.append(boxes)
    return all_boxes

import torch.cuda.amp
import multiprocessing as mp
import signal

class TimeoutException(Exception): pass

def handler(signum, frame):
    raise TimeoutException("NMS timeout")

signal.signal(signal.SIGALRM, handler)

def get_bboxes(
    model,
    loader,
    use_amp=False,
    device='cuda',
    IoU_threshold: float = 0.5,
    threshold: float = 0.5,
    S: int = 7,
    C: int = 1,
    max_batches: int = 1,
    max_boxes_per_image: int = 100
):
    all_pred_box = []
    all_true_box = []
    train_idx = 0
    model.eval()

    for batch_idx, (x, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            print(f"===> Stopping early at batch {batch_idx} (for debugging)")
            break

        x, labels = x.to(device), labels.to(device)
        print(f"\n===> Batch {batch_idx} | Getting predictions...")
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions = model(x)
            else:
                predictions = model(x)

        print("===> Converting to boxes...")
        bboxes = grid_boxes_to_boxes(predictions.cpu(), device='cpu', S=S, C=C)
        true_labels = grid_boxes_to_boxes(labels.cpu(), device='cpu', S=S, C=C)
        batch_size = x.shape[0]

        for idx in range(batch_size):
            preds = bboxes[idx]
            print(f"→ Image {train_idx} | Predicted {len(preds)} boxes")

            if len(preds) == 0:
                print("⚠️  No predicted boxes, skipping.")
                train_idx += 1
                continue

            if len(preds) > max_boxes_per_image:
                preds = sorted(preds, key=lambda x: x[1], reverse=True)[:max_boxes_per_image]

            try:
                signal.alarm(5)  # timeout sau 5 giây
                nms_predictions = non_max_suppression(preds, IoU_threshold, threshold)
                signal.alarm(0)
            except TimeoutException:
                print(f"❌ Timeout at image {train_idx}. Skipping.")
                train_idx += 1
                continue
            except Exception as e:
                print(f"❌ Error at image {train_idx}: {e}")
                train_idx += 1
                continue

            for pred in nms_predictions:
                all_pred_box.append([train_idx] + pred)

            for true_leb in true_labels[idx]:
                if true_leb[1] > threshold:
                    all_true_box.append([train_idx] + true_leb)

            train_idx += 1

    model.train()
    return all_pred_box, all_true_box

from collections import Counter

def mean_average_precision(
    pred_boxes,
    true_boxes,
    threshold_mAP=0.5,
    step_threshold=0.05,
    stop_threshold_mAP=0.95,
    C=1,
    epsilon=1e-12
    ):
    mean_average_precision = []
    while threshold_mAP < stop_threshold_mAP:
        average_precision = []
        for c in range(C):
            detection_list = []
            ground_truhts = []
            for detection in pred_boxes:
                if detection[1] == c:
                    detection_list.append(detection)
            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truhts.append(true_box)
            amount_bboxes = Counter([gt[0] for gt in ground_truhts])
            for key, val in amount_bboxes.items():
                amount_bboxes[key]= torch.zeros(val)
            detection_list.sort(key= lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detection_list)))
            FP = torch.zeros((len(detection_list)))
            total_true_boxes = len(ground_truhts)
            for detection_idx, detection in enumerate(detection_list):
                ground_truth_img = [ bbox for bbox in ground_truhts if bbox[0] == detection[0] ]
                best_iou = 0
                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(torch.tensor(gt[3:]), torch.tensor(detection[3:]))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                if best_iou > threshold_mAP:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_boxes+epsilon)
            precision = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precision = torch.cat((torch.tensor([1]), precision))
            recalls = torch.cat((torch.tensor([0]), recalls))
            average_precision.append(torch.trapz(precision, recalls))
        mean_average_precision.append(sum(average_precision)/len(average_precision))
        threshold_mAP += step_threshold
    return sum(mean_average_precision)/len(mean_average_precision)
