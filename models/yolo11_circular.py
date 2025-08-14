# YOLOv11-Circular (single file)
# - Same backbone/neck as YOLOv11 rewrite
# - Head predicts (cx, cy, r) + cls
# - Eval outputs [B, 3+nc, A] with absolute pixel units
# - Postprocess converts circle -> square box for NMS

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== helpers =====

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):
    @staticmethod
    def forward(x): return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = autopad(k, p)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act  = SiLU() if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))
    def fuse_forward(self, x): return self.act(self.conv(x))

# ===== core blocks =====

class Residual(nn.Module):
    def __init__(self, c, e=0.5):
        super().__init__()
        h = int(c * e)
        self.m = nn.Sequential(Conv(c, h, 3, 1), Conv(h, c, 3, 1))
    def forward(self, x): return x + self.m(x)

class C3K(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1  = Conv(c1, c2 // 2, 1, 1)
        self.cv2  = Conv(c1, c2 // 2, 1, 1)
        self.res  = nn.Sequential(Residual(c2 // 2, e=1.0), Residual(c2 // 2, e=1.0))
        self.fuse = Conv(c2, c2, 1, 1)
    def forward(self, x):
        a = self.res(self.cv1(x)); b = self.cv2(x)
        return self.fuse(torch.cat([a, b], 1))

class C3K2(nn.Module):
    def __init__(self, c1, c2, n, use_c3k, r=2):
        super().__init__()
        self.start = Conv(c1, 2 * (c2 // r), 1, 1)
        blocks = [C3K(c2 // r, c2 // r) if use_c3k else Residual(c2 // r) for _ in range(n)]
        self.blocks = nn.ModuleList(blocks)
        self.fuse   = Conv((2 + n) * (c2 // r), c2, 1, 1)
    def forward(self, x):
        a, b = self.start(x).chunk(2, 1)
        parts = [a, b]
        for m in self.blocks:
            b = m(b); parts.append(b)
        return self.fuse(torch.cat(parts, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x); y2 = self.m(y1); y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class Attention(nn.Module):
    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = max(1, num_head)
        self.dim_head = ch // self.num_head
        self.dim_key  = max(1, self.dim_head // 2)
        self.scale    = self.dim_key ** -0.5
        self.qkv  = Conv(ch, ch + self.dim_key * self.num_head * 2, 1, 1, act=False)
        self.conv1 = Conv(ch, ch, 3, 1, g=ch, act=False)
        self.conv2 = Conv(ch, ch, 1, 1, act=False)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)
        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)

class PSABlock(nn.Module):
    def __init__(self, ch, num_head):
        super().__init__()
        self.a = Attention(ch, num_head)
        self.mlp = nn.Sequential(Conv(ch, ch * 2, 1, 1), Conv(ch * 2, ch, 3, 1))
    def forward(self, x):
        x = x + self.a(x); return x + self.mlp(x)

class PSA(nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.s = Conv(ch, ch, 1, 1)
        self.blocks = nn.Sequential(*(PSABlock(ch, max(1, ch // 128)) for _ in range(n)))
        self.f = Conv(ch, ch, 1, 1)
    def forward(self, x):
        x = self.s(x)
        x = self.blocks(x)
        return self.f(x)

# ===== backbone & neck =====

class CSP(nn.Module):
    def __init__(self, c1, c2, n, use_c3k, r=2):
        super().__init__()
        self.block = C3K2(c1, c2, n, use_c3k, r)
    def forward(self, x): return self.block(x)

class DarkNet(nn.Module):
    def __init__(self, width, depth, csp_flags):
        super().__init__()
        self.p1 = nn.Sequential(Conv(width[0], width[1], 3, 2))
        self.p2 = nn.Sequential(Conv(width[1], width[2], 3, 2),
                                CSP(width[2], width[3], depth[0], csp_flags[0], r=4))
        self.p3 = nn.Sequential(Conv(width[3], width[3], 3, 2),
                                CSP(width[3], width[4], depth[1], csp_flags[0], r=4))
        self.p4 = nn.Sequential(Conv(width[4], width[4], 3, 2),
                                CSP(width[4], width[4], depth[2], csp_flags[1], r=2))
        self.p5 = nn.Sequential(Conv(width[4], width[5], 3, 2),
                                CSP(width[5], width[5], depth[3], csp_flags[1], r=2),
                                SPPF(width[5], width[5]),
                                PSA(width[5], depth[4]))
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5  # strides 8/16/32

class DarkFPN(nn.Module):
    def __init__(self, width, depth, csp_flags):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp_flags[0], r=2)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp_flags[0], r=2)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp_flags[0], r=2)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp_flags[1], r=2)
    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat([self.up(p5), p4], 1))
        p3 = self.h2(torch.cat([self.up(p4), p3], 1))
        p4 = self.h4(torch.cat([self.h3(p3), p4], 1))
        p5 = self.h6(torch.cat([self.h5(p4), p5], 1))
        return p3, p4, p5

# ===== anchor generator =====

def make_anchors(x: List[torch.Tensor], strides: torch.Tensor, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if torch.__version__ >= '1.10' else torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

# ===== Circle Head =====

class CircleHead(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)
    def __init__(self, nc=80, filters: Tuple[int, int, int] = (256, 512, 512)):
        super().__init__()
        self.nc = nc
        self.nl = len(filters)
        self.no = nc + 3  # (dx, dy, r_raw) + cls
        self.stride = torch.zeros(self.nl)
        reg_w = max(64, filters[0] // 4)
        cls_w = max(80, filters[0], nc)
        self.reg = nn.ModuleList(
            nn.Sequential(
                Conv(c, reg_w, 3, 1),
                Conv(reg_w, reg_w, 3, 1),
                nn.Conv2d(reg_w, 3, 1)          # dx, dy, r_raw
            ) for c in filters
        )
        self.cls = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1, g=c),
                Conv(c, cls_w, 1, 1),
                Conv(cls_w, cls_w, 3, 1, g=cls_w),
                Conv(cls_w, cls_w, 1, 1),
                nn.Conv2d(cls_w, nc, 1)
            ) for c in filters
        )
        self._init_cls_bias()

    @torch.no_grad()
    def _init_cls_bias(self, prior=0.01):
        for m in self.cls:
            last = [l for l in m.modules() if isinstance(l, nn.Conv2d)][-1]
            b = last.bias.view(self.nc)
            b.data.fill_(-math.log((1 - prior) / prior))
            last.bias = nn.Parameter(b)

    def forward(self, feats: List[torch.Tensor]):
        xs = []
        for i, (r, c) in enumerate(zip(self.reg, self.cls)):
            xs.append(torch.cat([r(feats[i]), c(feats[i])], 1))  # [B, 3+nc, H, W]
        # DEBUG: in train mode, print shapes once
        if self.training and not hasattr(self, "_debug_shapes_done"):
            print("[YOLOv11Circular][Train] head maps:",
                  [tuple(t.shape) for t in xs])  # [(B,3+nc,H,W), ... x3]
            self._debug_shapes_done = True
        if self.training:
            return xs

        device = feats[0].device
        if self.stride.device != device:
            self.stride = self.stride.to(device)

        # anchors: [A,2] (grid centers in feature units, offset=0.5), strides: [A,1]
        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(xs, self.stride))
        # concat maps -> [B, 3+nc, A]
        x = torch.cat([i.view(xs[0].shape[0], 3 + self.nc, -1) for i in xs], 2)
        if not hasattr(self, "_debug_eval_shape_done"):
            print("[YOLOv11Circular][Eval] out shape:", tuple(x.shape))  # [B,3+nc,A]
            self._debug_eval_shape_done = True
        reg, cls = x.split((3, self.nc), 1)  # reg=[B,3,A] = (dx, dy, r_raw)

        # broadcast anchors/strides to [B,1,A]
        a = self.anchors.unsqueeze(0).to(device)         # [1,2,A]
        s = self.strides.unsqueeze(1).to(device)         # [1,1,A]
        B = reg.shape[0]

        # decode: cx = (ax*stride) + dx, cy = (ay*stride) + dy, r = softplus(r_raw) * stride
        ax = (a[:, 0:1, :] * s).expand(B, -1, -1)        # [B,1,A]
        ay = (a[:, 1:2, :] * s).expand(B, -1, -1)        # [B,1,A]
        dx = reg[:, 0:1, :]                               # [B,1,A]  (pixels)
        dy = reg[:, 1:2, :]                               # [B,1,A]  (pixels)
        rr = torch.nn.functional.softplus(reg[:, 2:3, :]) * s  # [B,1,A]
        cx = ax + dx
        cy = ay + dy

        out = torch.cat((torch.cat([cx, cy, rr], 1), cls.sigmoid()), 1)  # [B, 3+nc, A]
        return out

# ===== Full model =====

@dataclass
class YOLOv11Config:
    csp: Tuple[bool, bool]
    depth: List[int]
    width: List[int]

DEFAULT_CFGS = {
    'n': YOLOv11Config(csp=(False, True), depth=[1,1,1,1,1,1], width=[3,16,32,64,128,256]),
    's': YOLOv11Config(csp=(False, True), depth=[1,1,1,1,1,1], width=[3,32,64,128,256,512]),
    'm': YOLOv11Config(csp=(True,  True), depth=[1,1,1,1,1,1], width=[3,64,128,256,512,512]),
    'l': YOLOv11Config(csp=(True,  True), depth=[2,2,2,2,2,2], width=[3,64,128,256,512,512]),
    'x': YOLOv11Config(csp=(True,  True), depth=[2,2,2,2,2,2], width=[3,96,192,384,768,768]),
}

class YOLOv11Circular(nn.Module):
    def __init__(self, cfg: YOLOv11Config, num_classes=80):
        super().__init__()
        self.backbone = DarkNet(cfg.width, cfg.depth, list(cfg.csp))
        self.neck     = DarkFPN(cfg.width, cfg.depth, list(cfg.csp))
        with torch.no_grad():
            d = torch.zeros(1, cfg.width[0], 256, 256)
            p3, p4, p5 = self.neck(self.backbone(d))
        self.head = CircleHead(num_classes, (p3.shape[1], p4.shape[1], p5.shape[1]))
        self.head.stride = torch.tensor([256 / p3.shape[-2], 256 / p4.shape[-2], 256 / p5.shape[-2]])
        self.stride = self.head.stride

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck((p3, p4, p5))
        return self.head([p3, p4, p5])

    def fuse(self):
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv(m.conv, m.bn)
                m.forward = m.fuse_forward
                delattr(m, 'bn')
        return self

# ===== postprocess & utils =====

def circle_to_xyxy(cxyr: torch.Tensor) -> torch.Tensor:
    # cxyr: [..., 3], return [..., 4] xyxy square box
    cx, cy, r = cxyr.unbind(-1)
    return torch.stack([cx - r, cy - r, cx + r, cy + r], -1)

def nms(boxes, scores, iou_thres=0.65, topk=300):
    keep = []
    idxs = scores.argsort(descending=True)[:topk]
    def box_iou(b1, b2):
        area1 = (b1[:,2]-b1[:,0]).clamp(0)*(b1[:,3]-b1[:,1]).clamp(0)
        area2 = (b2[:,2]-b2[:,0]).clamp(0)*(b2[:,3]-b2[:,1]).clamp(0)
        lt = torch.max(b1[:,None,:2], b2[:,:2])
        rb = torch.min(b1[:,None,2:], b2[:,2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[...,0]*wh[...,1]
        return inter / (area1[:,None] + area2 - inter + 1e-9)
    while idxs.numel() > 0:
        i = idxs[0]; keep.append(i)
        if idxs.numel() == 1: break
        iou = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][iou < iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)

@torch.no_grad()
def postprocess_circular(model_out, conf_thres=0.25, iou_thres=0.50, max_det=300):
    """From YOLOv11Circular.eval() output [B,3+nc,A] → list of det [N,6] (xyxy, score, cls)."""
    p = model_out if not isinstance(model_out, dict) else (model_out.get("infer", model_out))
    # transpose to [B, A, 3+nc] if needed
    if p.shape[1] <= p.shape[2]:  # [B,3+nc,A]
        p = p.transpose(1, 2)     # [B,A,3+nc]
    out = []
    for b in range(p.shape[0]):
        pb = p[b]
        cxyr = pb[:, :3]
        cls  = pb[:, 3:]
        cls_conf, cls_id = cls.max(dim=1)
        m = cls_conf > conf_thres
        if m.sum() == 0:
            out.append(torch.zeros((0,6), device=p.device)); continue
        boxes = circle_to_xyxy(cxyr[m])
        scores = cls_conf[m]
        cids   = cls_id[m]
        dets = []
        for c in cids.unique():
            idx = (cids == c).nonzero(as_tuple=False).squeeze(1)
            k = nms(boxes[idx], scores[idx], iou_thres, max_det)
            det = torch.cat([boxes[idx][k], scores[idx][k, None], c.float().repeat(k.numel(),1)], 1)
            dets.append(det)
        out.append(torch.cat(dets, 0)[:max_det] if dets else torch.zeros((0,6), device=p.device))
    return out

# ===== loss (classification + simple L1 for circles) =====

class CircleDetectionLoss(nn.Module):
    """
    Hỗ trợ 2 kiểu target:
      - Mới: List[Tensor], mỗi Tensor [N,6] = (cls, cx, cy, r, 0, stride_idx)
      - Legacy YOLOv1 grid: Tensor [S,S,C] hoặc List chứa Tensor[S,S,C]
        -> chuyển thành [N,6] theo chỉ số kênh cấu hình trong legacy_idx.
    """
    def __init__(self, num_classes, l1_weight=1.0, legacy_idx=None, img_size_hint=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.l1_weight = float(l1_weight)
        # legacy_idx: {'cx':int,'cy':int,'r':int,'obj':int,'cls':int or None}
        # ví dụ thường gặp: {'cx':0,'cy':1,'r':2,'obj':3,'cls':4}
        # nếu cls=None hoặc num_classes==1 thì coi tất cả positive đều là cls=0
        self.legacy_idx = legacy_idx or {'cx':0,'cy':1,'r':2,'obj':3,'cls':4}
        # gợi ý kích thước ảnh để scale nếu dữ liệu normalized 0..1
        self.img_size_hint = img_size_hint  # ví dụ 448 hoặc 640; nếu None dùng 64*S gần đúng

    @staticmethod
    def _pick_stride_idx(r_pixel: float, strides=(8.0, 16.0, 32.0), k: float = 3.0):
        for si, s in enumerate(strides):
            if r_pixel / s <= k:
                return si
        return len(strides) - 1

    def _legacy_grid_to_list(self, grid: torch.Tensor) -> torch.Tensor:
        S1, S2, C = grid.shape
        assert S1 == S2, "Legacy grid phải là SxSxC"
        dev = grid.device
        idx = self.legacy_idx
        cx_ch, cy_ch, r_ch = idx['cx'], idx['cy'], idx['r']
        obj_ch = idx.get('obj', None)
        cls_ch = idx.get('cls', None)

        cx_raw = grid[..., cx_ch]
        cy_raw = grid[..., cy_ch]
        r_raw  = grid[..., r_ch]

        # obj mask
        if obj_ch is None:
            # nếu không có obj channel, mặc định mọi cell có r>0 là positive
            obj = (r_raw > 0)
        else:
            obj = grid[..., obj_ch] > 0.5

        ys, xs = (obj).nonzero(as_tuple=True)

        # phát hiện normalized hay pixel
        max_v = torch.stack([cx_raw.max(), cy_raw.max(), r_raw.max()]).max()
        normalized = bool(max_v <= 1.5)

        # scale về pixel nếu cần
        if normalized:
            if self.img_size_hint is not None:
                img_w = img_h = float(self.img_size_hint)
            else:
                cell_size = 64.0
                img_w = img_h = float(S1) * cell_size
        else:
            img_w = img_h = None  # đã là pixel

        out = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            cx = float(cx_raw[y, x].item())
            cy = float(cy_raw[y, x].item())
            r  = float(r_raw[y, x].item())
            if normalized:
                cx *= img_w
                cy *= img_h
                r  *= (img_w + img_h) * 0.5

            if self.num_classes > 1 and cls_ch is not None:
                cls_id = int(round(float(grid[y, x, cls_ch].item())))
                cls_id = max(0, min(self.num_classes - 1, cls_id))
            else:
                cls_id = 0

            si = self._pick_stride_idx(r)
            out.append([cls_id, cx, cy, r, 0.0, float(si)])

        if len(out) == 0:
            return torch.zeros((0, 6), dtype=torch.float32, device=dev)
        return torch.tensor(out, dtype=torch.float32, device=dev)

    def forward(self, preds_train: List[torch.Tensor], targets):
        # --- chuẩn hóa targets ---
        normalized_targets = []
        if isinstance(targets, torch.Tensor) and targets.dim() == 4:  # [B,S,S,C]
            for b in range(targets.shape[0]):
                normalized_targets.append(self._legacy_grid_to_list(targets[b]))
        elif isinstance(targets, list):
            for t in targets:
                if isinstance(t, torch.Tensor) and t.dim() == 3:    # [S,S,C]
                    normalized_targets.append(self._legacy_grid_to_list(t))
                else:
                    normalized_targets.append(t)                    # [N,6]
        else:
            raise TypeError("targets phải là List[Tensor[N,6]] hoặc Tensor[B,S,S,C]")

        # --- debug: đếm positive mỗi batch ---
        if not hasattr(self, "_dbg_once"):
            pos_counts = [int(t.shape[0]) for t in normalized_targets]
            print(f"[Loss][Legacy->List] positives per batch item: {pos_counts}")
            self._dbg_once = True

        # --- loss ---
        device = preds_train[0].device
        dtype  = preds_train[0].dtype
        loss_cls = torch.zeros((), device=device, dtype=dtype)
        loss_l1  = torch.zeros((), device=device, dtype=dtype)
        total_pos = 0

        for b in range(preds_train[0].shape[0]):
            t = normalized_targets[b]  # [N,6]
            if t.numel() == 0:
                continue

            for si, P in enumerate(preds_train):
                # P: [B, (3+nc), H, W]
                _, C, H, W = P.shape
                nc = C - 3
                cls_logits = P[b, 3:, :, :]      # [nc,H,W]
                reg_logits = P[b, :3, :, :]      # [3,H,W]

                th = t[t[:, 5] == si]
                if th.numel() == 0:
                    continue

                th = th.to(device)

                stride = 8 * (2 ** si)
                gx = (th[:, 1] / stride).clamp(0, W - 1)
                gy = (th[:, 2] / stride).clamp(0, H - 1)
                gi = gx.long().clamp(0, W - 1)
                gj = gy.long().clamp(0, H - 1)

                # --- classification ---
                for k in range(th.shape[0]):
                    cls_id = int(th[k, 0].item())
                    logits = cls_logits[:, gj[k], gi[k]]  # [nc]
                    target = torch.zeros_like(logits, device=device)
                    if nc == 1:
                        target[:] = 1.0
                    else:
                        target[cls_id] = 1.0
                    loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                        logits, target, reduction='sum'
                    )

                # --- regression L1 ---
                cx_cell = (gi.float() + 0.5) * stride
                cy_cell = (gj.float() + 0.5) * stride
                dx = reg_logits[0, gj, gi]
                dy = reg_logits[1, gj, gi]
                r_p = F.softplus(reg_logits[2, gj, gi]) * stride
                cx_p = cx_cell + dx
                cy_p = cy_cell + dy

                cx_t = th[:, 1]
                cy_t = th[:, 2]
                r_t  = th[:, 3]

                l1 = F.smooth_l1_loss(cx_p, cx_t, reduction='sum') \
                   + F.smooth_l1_loss(cy_p, cy_t, reduction='sum') \
                   + F.smooth_l1_loss(r_p,  r_t,  reduction='sum')
                loss_l1 = loss_l1 + l1
                total_pos += th.shape[0]

        if total_pos == 0:
            # Trả đúng zero dương, tránh -0.0000
            return torch.zeros((), device=device, dtype=dtype)

        return (loss_cls / total_pos) + self.l1_weight * (loss_l1 / total_pos)

# ===== fuse & export =====

def fuse_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                      stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1,1)).reshape(-1) + b_bn)
    return fused

@torch.no_grad()
def export_onnx(model: YOLOv11Circular, path="yolov11_circular.onnx", opset=12, dynamic=False):
    model = copy.deepcopy(model).eval()
    x = torch.zeros(1, 3, 640, 640)
    torch.onnx.export(model, x, path, input_names=['images'], output_names=['preds'],
                      opset_version=opset, dynamic_axes={'images': {0: 'batch'}, 'preds': {0: 'batch'}} if dynamic else None)
    return path
