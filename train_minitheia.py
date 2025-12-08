#!/usr/bin/env python3
"""
train_minitheia.py

Mini-Theia training (ViT-Tiny student) distilling CLIP ViT-B/32 + DINOv2_vits14.

Features:
 - Student ViT-Tiny backbone (timm)
 - Two projection heads: CLIP (512) and DINOv2 (384)
 - FP16 training with torch.amp.autocast('cuda', ...)
 - Optional pre-extraction (caching) of teacher embeddings to disk
 - Checkpointing (per epoch + best)
 - Robust clip.load handling and DINOv2 hub usage
 - CLI args for dataset paths and hyperparams

Usage example:
  python train_minitheia.py \
    --data-dir /data/imagenet \
    --train-subdir train \
    --val-subdir val_sorted \
    --out-dir ./runs/theia_mini \
    --epochs 8 \
    --batch-size 64 \
    --image-size 224 \
    --fp16 \
    --cache-teacher \
    --cache-dir ./teacher_cache \
    --num-workers 6
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import timm
import clip
from tqdm import tqdm
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

# ---------------------
# Models
# ---------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class TheiaStudent(nn.Module):
    def __init__(self, backbone_name='vit_tiny_patch16_224', proj_clip=512, proj_dino=384):
        super().__init__()
        # create backbone without classifier head
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        # heads
        self.head_clip = ProjectionHead(feat_dim, proj_clip)
        self.head_dino = ProjectionHead(feat_dim, proj_dino)

    def forward(self, x):
        feat = self.backbone(x)            # (B, feat_dim)
        clip_out = self.head_clip(feat)    # (B, 512)
        dino_out = self.head_dino(feat)    # (B, 384)
        clip_out = nn.functional.normalize(clip_out, dim=-1)
        dino_out = nn.functional.normalize(dino_out, dim=-1)
        return clip_out, dino_out, feat

# ---------------------
# Dataset
# ---------------------
class ImageFolderPILDual(Dataset):
    def __init__(self, root: str, student_transform, clip_transform, dino_transform):
        self.root = Path(root)
        classes = [d.name for d in sorted(self.root.iterdir()) if d.is_dir()]
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        for c in classes:
            p = self.root / c
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in ('.jpeg','.jpg','.png'):
                    self.samples.append((str(f), self.class_to_idx[c]))
        self.student_transform = student_transform
        self.clip_transform = clip_transform
        self.dino_transform = dino_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        s = self.student_transform(img) if self.student_transform else None
        c = self.clip_transform(img) if self.clip_transform else None
        d = self.dino_transform(img) if self.dino_transform else None
        return s, c, d, label, path

def collate_fn(batch):
    students = torch.stack([b[0] for b in batch])
    clips = torch.stack([b[1] for b in batch])
    dinos = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    paths = [b[4] for b in batch]
    return students, clips, dinos, labels, paths

# ---------------------
# Teacher loaders
# ---------------------
def load_clip(device):
    # clip.load may return 2 or 3 values depending on version
    loaded = clip.load("ViT-B/32", device=device, jit=False)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, preprocess = loaded
    elif isinstance(loaded, tuple) and len(loaded) >= 2:
        model, preprocess = loaded[0], loaded[1]
    else:
        raise RuntimeError("Unexpected return from clip.load")
    model.eval()
    def extract(img_tensor):
        # expects tensor on device, preprocessed
        with torch.no_grad():
            feats = model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    return model, preprocess, extract

def load_dinov2(device, variant="dinov2_vits14"):
    # may download on first run via torch.hub
    print("Loading DINOv2 variant:", variant)
    dinov2 = torch.hub.load("facebookresearch/dinov2", variant)
    dinov2.to(device)
    dinov2.eval()
    def extract(img_tensor):
        with torch.no_grad():
            feats = dinov2(img_tensor)
            # variants may return tensor or dict
            if isinstance(feats, dict):
                # try common keys
                feats = feats.get('feat', feats.get('feats', feats.get('features', None)))
                if feats is None:
                    raise RuntimeError("Unexpected DINOv2 output structure (dict).")
            # normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    return dinov2, extract

# ---------------------
# Caching helpers
# ---------------------
def cache_teacher_embeddings(dataloader: DataLoader, extract_fn, out_dir: Path, device: torch.device, suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching teacher embeddings ({suffix}) -> {out_dir}")
    for batch in tqdm(dataloader, desc=f"Cache-{suffix}"):
        # batch: students, clip_imgs, dino_imgs, labels, paths
        students, clip_imgs, dino_imgs, labels, paths = batch
        imgs = clip_imgs if suffix == "clip" else dino_imgs
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            feats = extract_fn(imgs).cpu()  # (B, dim)
        for p, f in zip(paths, feats):
            fname = Path(p).stem + f".{suffix}.pt"
            torch.save(f, out_dir / fname)

# ---------------------
# Training & validation
# ---------------------
def train_epoch(student: nn.Module, device, train_loader: DataLoader, optimizer, scaler,
                clip_extract_fn, dino_extract_fn, cfg, teacher_cache: Optional[Path]=None):
    student.train()
    mse = nn.MSELoss()
    running = 0.0
    iters = 0
    pbar = tqdm(train_loader, desc="Train")
    for s_imgs, c_imgs, d_imgs, labels, paths in pbar:
        s_imgs = s_imgs.to(device, non_blocking=True)
        optimizer.zero_grad()
        # get teacher targets either from cache or compute on the fly
        if teacher_cache is not None:
            clip_targets = []
            dino_targets = []
            for p in paths:
                clip_path = teacher_cache / f"{Path(p).stem}.clip.pt"
                dino_path = teacher_cache / f"{Path(p).stem}.dino.pt"
                if not clip_path.exists() or not dino_path.exists():
                    raise FileNotFoundError(f"Missing cached teacher file for {p}. Expected {clip_path}, {dino_path}")
                clip_targets.append(torch.load(clip_path))
                dino_targets.append(torch.load(dino_path))
            clip_targets = torch.stack(clip_targets).to(device)
            dino_targets = torch.stack(dino_targets).to(device)
        else:
            c_imgs = c_imgs.to(device, non_blocking=True)
            d_imgs = d_imgs.to(device, non_blocking=True)
            with torch.no_grad():
                clip_targets = clip_extract_fn(c_imgs)
                dino_targets = dino_extract_fn(d_imgs)

        # forward student and loss
        with torch.amp.autocast(device_type='cuda', enabled=cfg['fp16']):
            s_clip, s_dino, _ = student(s_imgs)
            # sanity check dims
            if s_clip.shape[1] != clip_targets.shape[1]:
                raise RuntimeError(f"Dimension mismatch CLIP: student {s_clip.shape[1]} vs teacher {clip_targets.shape[1]}")
            if s_dino.shape[1] != dino_targets.shape[1]:
                raise RuntimeError(f"Dimension mismatch DINO: student {s_dino.shape[1]} vs teacher {dino_targets.shape[1]}")
            loss_clip = mse(s_clip, clip_targets)
            loss_dino = mse(s_dino, dino_targets)
            loss = 0.5 * (loss_clip + loss_dino)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        iters += 1
        pbar.set_postfix({'loss': running / (iters if iters>0 else 1)})
    return running / max(1, iters)

def validate(student: nn.Module, device, val_loader: DataLoader, clip_extract_fn, dino_extract_fn, cfg, teacher_cache: Optional[Path]=None, max_batches: Optional[int]=200):
    student.eval()
    mse = nn.MSELoss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for s_imgs, c_imgs, d_imgs, labels, paths in tqdm(val_loader, desc="Validate", leave=False):
            s_imgs = s_imgs.to(device, non_blocking=True)
            if teacher_cache is not None:
                clip_targets = []
                dino_targets = []
                for p in paths:
                    clip_targets.append(torch.load(teacher_cache / f"{Path(p).stem}.clip.pt"))
                    dino_targets.append(torch.load(teacher_cache / f"{Path(p).stem}.dino.pt"))
                clip_targets = torch.stack(clip_targets).to(device)
                dino_targets = torch.stack(dino_targets).to(device)
            else:
                c_imgs = c_imgs.to(device, non_blocking=True)
                d_imgs = d_imgs.to(device, non_blocking=True)
                clip_targets = clip_extract_fn(c_imgs)
                dino_targets = dino_extract_fn(d_imgs)

            s_clip, s_dino, _ = student(s_imgs)
            # ensure dims match
            if s_clip.shape[1] != clip_targets.shape[1] or s_dino.shape[1] != dino_targets.shape[1]:
                raise RuntimeError("Dimension mismatch during validation.")
            l_clip = mse(s_clip, clip_targets).item()
            l_dino = mse(s_dino, dino_targets).item()
            total += 0.5 * (l_clip + l_dino)
            count += 1
            if max_batches and count >= max_batches:
                break
    return total / max(1, count)

# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--train-subdir', default='train')
    parser.add_argument('--val-subdir', default='val_sorted')
    parser.add_argument('--out-dir', default='./runs/theia_mini')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--fp16', action='store_true', help='Use AMP (recommended)')
    parser.add_argument('--cache-teacher', action='store_true')
    parser.add_argument('--cache-dir', default='./teacher_cache')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # transforms
    student_train = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    student_val = T.Compose([
        T.Resize(int(args.image_size*1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # CLIP normalization (approx)
    clip_train = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]),
    ])
    clip_val = T.Compose([
        T.Resize(int(args.image_size*1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]),
    ])
    # DINOv2 normalization (approx)
    dino_train = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    dino_val = T.Compose([
        T.Resize(int(args.image_size*1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = ImageFolderPILDual(os.path.join(args.data_dir, args.train_subdir), student_train, clip_train, dino_train)
    val_ds   = ImageFolderPILDual(os.path.join(args.data_dir, args.val_subdir), student_val, clip_val, dino_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # create student
    student = TheiaStudent(backbone_name='vit_tiny_patch16_224', proj_clip=512, proj_dino=384)
    student.to(device)

    # gradient checkpointing if available
    try:
        if hasattr(student.backbone, 'set_grad_checkpointing'):
            student.backbone.set_grad_checkpointing(True)
            print("Enabled gradient checkpointing on backbone.")
    except Exception as e:
        print("Could not enable gradient checkpointing:", e)

    # load teachers
    print("Loading CLIP...")
    clip_model, clip_preproc, clip_extract_fn = load_clip(device)
    print("Loading DINOv2...")
    dino_model, dino_extract_fn = load_dinov2(device, variant="dinov2_vits14")

    # freeze teachers
    for p in clip_model.parameters(): p.requires_grad = False
    for p in dino_model.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    teacher_cache = None
    if args.cache_teacher:
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        # detect if cache present by checking a few sample files
        need = False
        if len(train_ds) > 0:
            sample_path = Path(train_ds.samples[0][0]).stem
            if not (cache_root / f"{sample_path}.clip.pt").exists() or not (cache_root / f"{sample_path}.dino.pt").exists():
                need = True
        else:
            need = True

        if need:
            # make small dataloader for pre-extract to limit memory
            pre_bs = max(8, min(64, args.batch_size//4))
            pre_dl = DataLoader(train_ds, batch_size=pre_bs, shuffle=False, num_workers=max(2, args.num_workers//2), pin_memory=True, collate_fn=collate_fn)
            cache_teacher_embeddings(pre_dl, clip_extract_fn, cache_root, device, suffix="clip")
            cache_teacher_embeddings(pre_dl, dino_extract_fn, cache_root, device, suffix="dino")
            # val as well
            pre_vdl = DataLoader(val_ds, batch_size=pre_bs, shuffle=False, num_workers=max(2, args.num_workers//2), pin_memory=True, collate_fn=collate_fn)
            cache_teacher_embeddings(pre_vdl, clip_extract_fn, cache_root, device, suffix="clip")
            cache_teacher_embeddings(pre_vdl, dino_extract_fn, cache_root, device, suffix="dino")
        teacher_cache = cache_root

    # resume best if requested
    best_val = float('inf')
    best_path = out_dir / "theia_mini_best.pt"
    start_epoch = 0
    if args.resume and best_path.exists():
        print("Resuming student weights from best checkpoint...")
        sd = torch.load(best_path, map_location=device)
        student.load_state_dict(sd)
        print("Loaded best student weights.")
    # training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(student, device, train_loader, optimizer, scaler, clip_extract_fn, dino_extract_fn, {'fp16': args.fp16}, teacher_cache)
        val_loss = validate(student, device, val_loader, clip_extract_fn, dino_extract_fn, {'fp16': args.fp16}, teacher_cache, max_batches=200)
        t1 = time.time()
        print(f"[Epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} epoch_time={t1-t0:.1f}s")
        # save epoch checkpoint
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(ckpt, out_dir / f"theia_mini_epoch{epoch+1}.pth")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), best_path)
            print("Saved best model (val_loss improved).")

    print("Training finished. Best val loss:", best_val)
    print("Best model at:", best_path)

if __name__ == "__main__":
    main()
