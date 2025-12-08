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
    def __init__(self, backbone_name='vit_tiny_patch16_224', proj_clip=512, proj_dino=384, proj_fastsam=256):
        super().__init__()
        # create backbone without classifier head
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        
        # heads for each teacher
        self.head_clip = ProjectionHead(feat_dim, proj_clip)
        self.head_dino = ProjectionHead(feat_dim, proj_dino)
        self.head_fastsam = ProjectionHead(feat_dim, proj_fastsam)

    def forward(self, x):
        feat = self.backbone(x)               # (B, feat_dim)
        clip_out = self.head_clip(feat)       # (B, 512)
        dino_out = self.head_dino(feat)       # (B, 384)
        fastsam_out = self.head_fastsam(feat) # (B, 256)
        
        # Normalize all outputs
        clip_out = nn.functional.normalize(clip_out, dim=-1)
        dino_out = nn.functional.normalize(dino_out, dim=-1)
        fastsam_out = nn.functional.normalize(fastsam_out, dim=-1)
        
        return clip_out, dino_out, fastsam_out, feat

# ---------------------
# Dataset
# ---------------------
class ImageFolderPILTriple(Dataset):
    def __init__(self, root: str, student_transform, clip_transform, dino_transform, fastsam_transform):
        self.root = Path(root)
        classes = [d.name for d in sorted(self.root.iterdir()) if d.is_dir()]
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        
        for c in classes:
            p = self.root / c
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in ('.jpeg','.jpg','.png','.bmp','.tiff'):
                    self.samples.append((str(f), self.class_to_idx[c]))
        
        self.student_transform = student_transform
        self.clip_transform = clip_transform
        self.dino_transform = dino_transform
        self.fastsam_transform = fastsam_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        s = self.student_transform(img) if self.student_transform else None
        c = self.clip_transform(img) if self.clip_transform else None
        d = self.dino_transform(img) if self.dino_transform else None
        f = self.fastsam_transform(img) if self.fastsam_transform else None
        
        return s, c, d, f, label, path

def collate_fn(batch):
    students = torch.stack([b[0] for b in batch])
    clips = torch.stack([b[1] for b in batch])
    dinos = torch.stack([b[2] for b in batch])
    fastsams = torch.stack([b[3] for b in batch])
    labels = torch.tensor([b[4] for b in batch], dtype=torch.long)
    paths = [b[5] for b in batch]
    return students, clips, dinos, fastsams, labels, paths

# ---------------------
# Teacher loaders
# ---------------------
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    
    def extract(img_tensor):
        with torch.no_grad():
            feats = model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    
    return model, preprocess, extract

def load_dinov2(device, variant="dinov2_vits14"):
    print(f"Loading DINOv2 variant: {variant}")
    dinov2 = torch.hub.load("facebookresearch/dinov2", variant)
    dinov2.to(device)
    dinov2.eval()
    
    def extract(img_tensor):
        with torch.no_grad():
            feats = dinov2(img_tensor)
            if isinstance(feats, dict):
                feats = feats.get('x_norm_clstoken', feats.get('feat', None))
                if feats is None:
                    # Fallback to first tensor in dict
                    feats = list(feats.values())[0]
            
            # Handle different output dimensions
            if feats.dim() > 2:
                feats = feats.mean(dim=[2, 3])  # Global average pooling for feature maps
            
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    
    return dinov2, extract

def load_fastsam(device, variant='FastSAM-s'):
    """Load FastSAM model from Ultralytics"""
    try:
        from ultralytics import FastSAM
        print(f"Loading FastSAM variant: {variant}")
        
        # Load model
        model = FastSAM(variant)  # Options: 'FastSAM-s', 'FastSAM-x'
        model.to(device)
        model.eval()
        
        def extract(img_tensor):
            """Extract features from FastSAM"""
            with torch.no_grad():
                # FastSAM expects images in [0, 1] range
                # Ensure images are in correct range
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                
                # Convert RGB to BGR for FastSAM
                img_tensor_bgr = img_tensor[:, [2, 1, 0], :, :]
                
                # Run inference
                results = model(img_tensor_bgr, verbose=False, imgsz=640)
                
                # Extract features from the results
                features_list = []
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        # Get box features if available
                        if hasattr(r.boxes, 'feats') and r.boxes.feats is not None:
                            # Average pooling over box features
                            f = r.boxes.feats.mean(dim=0) if len(r.boxes.feats) > 0 else torch.zeros(256, device=device)
                        else:
                            # Use segmentation features if available
                            if hasattr(r, 'masks') and r.masks is not None and hasattr(r.masks, 'feats'):
                                f = r.masks.feats.mean(dim=(1, 2)).squeeze()
                            else:
                                # Use backbone features - need to access encoder output
                                # This is model-specific and may require modifications
                                f = torch.zeros(256, device=device)
                    else:
                        # No detections, use zeros
                        f = torch.zeros(256, device=device)
                    
                    # Ensure correct dimension
                    if f.dim() == 0:
                        f = f.unsqueeze(0)
                    if f.shape[0] != 256:
                        # Pad or truncate to 256
                        if f.shape[0] < 256:
                            padding = torch.zeros(256 - f.shape[0], device=device)
                            f = torch.cat([f, padding])
                        else:
                            f = f[:256]
                    
                    features_list.append(f)
                
                features = torch.stack(features_list, dim=0)
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                
                return features
        
        return model, extract
        
    except ImportError:
        print("Warning: Ultralytics not installed. Using dummy FastSAM features.")
        print("Install with: pip install ultralytics")
        
        # Create dummy extractor
        def dummy_extract(img_tensor):
            batch_size = img_tensor.shape[0]
            dummy_features = torch.randn(batch_size, 256, device=device)
            dummy_features = dummy_features / dummy_features.norm(dim=-1, keepdim=True)
            return dummy_features
        
        return None, dummy_extract

# ---------------------
# Caching helpers
# ---------------------
def cache_teacher_embeddings(dataloader: DataLoader, extract_fn, out_dir: Path, 
                            device: torch.device, suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching teacher embeddings ({suffix}) -> {out_dir}")
    
    for batch in tqdm(dataloader, desc=f"Cache-{suffix}"):
        students, clip_imgs, dino_imgs, fastsam_imgs, labels, paths = batch
        
        # Select the right images for this teacher
        if suffix == "clip":
            imgs = clip_imgs
        elif suffix == "dino":
            imgs = dino_imgs
        elif suffix == "fastsam":
            imgs = fastsam_imgs
        else:
            raise ValueError(f"Unknown suffix: {suffix}")
        
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
                clip_extract_fn, dino_extract_fn, fastsam_extract_fn, cfg, 
                teacher_cache: Optional[Path] = None):
    student.train()
    mse = nn.MSELoss()
    running = 0.0
    iters = 0
    
    pbar = tqdm(train_loader, desc="Train")
    for s_imgs, c_imgs, d_imgs, f_imgs, labels, paths in pbar:
        s_imgs = s_imgs.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        # Get teacher targets from cache or compute on the fly
        if teacher_cache is not None:
            clip_targets, dino_targets, fastsam_targets = [], [], []
            for p in paths:
                stem = Path(p).stem
                clip_path = teacher_cache / f"{stem}.clip.pt"
                dino_path = teacher_cache / f"{stem}.dino.pt"
                fastsam_path = teacher_cache / f"{stem}.fastsam.pt"
                
                if not (clip_path.exists() and dino_path.exists() and fastsam_path.exists()):
                    raise FileNotFoundError(f"Missing cached teacher file for {stem}")
                
                clip_targets.append(torch.load(clip_path))
                dino_targets.append(torch.load(dino_path))
                fastsam_targets.append(torch.load(fastsam_path))
            
            clip_targets = torch.stack(clip_targets).to(device)
            dino_targets = torch.stack(dino_targets).to(device)
            fastsam_targets = torch.stack(fastsam_targets).to(device)
        else:
            c_imgs = c_imgs.to(device, non_blocking=True)
            d_imgs = d_imgs.to(device, non_blocking=True)
            f_imgs = f_imgs.to(device, non_blocking=True)
            
            with torch.no_grad():
                clip_targets = clip_extract_fn(c_imgs)
                dino_targets = dino_extract_fn(d_imgs)
                fastsam_targets = fastsam_extract_fn(f_imgs)
        
        # Forward student and compute loss
        with torch.amp.autocast(device_type='cuda', enabled=cfg['fp16']):
            s_clip, s_dino, s_fastsam, _ = student(s_imgs)
            
            # Check dimensions
            if s_clip.shape[1] != clip_targets.shape[1]:
                raise RuntimeError(f"CLIP dim mismatch: student {s_clip.shape[1]} vs teacher {clip_targets.shape[1]}")
            if s_dino.shape[1] != dino_targets.shape[1]:
                raise RuntimeError(f"DINO dim mismatch: student {s_dino.shape[1]} vs teacher {dino_targets.shape[1]}")
            if s_fastsam.shape[1] != fastsam_targets.shape[1]:
                raise RuntimeError(f"FastSAM dim mismatch: student {s_fastsam.shape[1]} vs teacher {fastsam_targets.shape[1]}")
            
            # Weighted loss (adjust weights as needed)
            loss_clip = mse(s_clip, clip_targets)
            loss_dino = mse(s_dino, dino_targets)
            loss_fastsam = mse(s_fastsam, fastsam_targets)
            
            # You can adjust these weights based on importance
            weight_clip = 0.4
            weight_dino = 0.4
            weight_fastsam = 0.2
            
            loss = (weight_clip * loss_clip + 
                   weight_dino * loss_dino + 
                   weight_fastsam * loss_fastsam)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running += loss.item()
        iters += 1
        pbar.set_postfix({'loss': running / iters})
    
    return running / max(1, iters)

def validate(student: nn.Module, device, val_loader: DataLoader, 
             clip_extract_fn, dino_extract_fn, fastsam_extract_fn, cfg, 
             teacher_cache: Optional[Path] = None, max_batches: Optional[int] = 200):
    student.eval()
    mse = nn.MSELoss()
    total = 0.0
    count = 0
    
    with torch.no_grad():
        for s_imgs, c_imgs, d_imgs, f_imgs, labels, paths in tqdm(val_loader, desc="Validate", leave=False):
            s_imgs = s_imgs.to(device, non_blocking=True)
            
            if teacher_cache is not None:
                clip_targets, dino_targets, fastsam_targets = [], [], []
                for p in paths:
                    stem = Path(p).stem
                    clip_targets.append(torch.load(teacher_cache / f"{stem}.clip.pt"))
                    dino_targets.append(torch.load(teacher_cache / f"{stem}.dino.pt"))
                    fastsam_targets.append(torch.load(teacher_cache / f"{stem}.fastsam.pt"))
                
                clip_targets = torch.stack(clip_targets).to(device)
                dino_targets = torch.stack(dino_targets).to(device)
                fastsam_targets = torch.stack(fastsam_targets).to(device)
            else:
                c_imgs = c_imgs.to(device, non_blocking=True)
                d_imgs = d_imgs.to(device, non_blocking=True)
                f_imgs = f_imgs.to(device, non_blocking=True)
                
                clip_targets = clip_extract_fn(c_imgs)
                dino_targets = dino_extract_fn(d_imgs)
                fastsam_targets = fastsam_extract_fn(f_imgs)
            
            s_clip, s_dino, s_fastsam, _ = student(s_imgs)
            
            # Weighted validation loss
            weight_clip = 0.4
            weight_dino = 0.4
            weight_fastsam = 0.2
            
            l_clip = mse(s_clip, clip_targets).item()
            l_dino = mse(s_dino, dino_targets).item()
            l_fastsam = mse(s_fastsam, fastsam_targets).item()
            
            total += (weight_clip * l_clip + weight_dino * l_dino + weight_fastsam * l_fastsam)
            count += 1
            
            if max_batches and count >= max_batches:
                break
    
    return total / max(1, count)

# ---------------------
# Main function
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--train-subdir', default='train', help='Training data subdirectory')
    parser.add_argument('--val-subdir', default='val_sorted', help='Validation data subdirectory')
    parser.add_argument('--out-dir', default='./runs/theia_triple', help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (adjust for 12GB VRAM)')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (adjust based on CPU)')
    parser.add_argument('--fp16', action='store_true', help='Use AMP mixed precision')
    parser.add_argument('--cache-teacher', action='store_true', help='Cache teacher embeddings')
    parser.add_argument('--cache-dir', default='./teacher_cache_triple', help='Directory for cached embeddings')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--grad-checkpoint', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--fastsam-variant', default='FastSAM-s', choices=['FastSAM-s', 'FastSAM-x'], 
                       help='FastSAM variant to use')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define transforms for each model
    # Student transforms (standard ImageNet)
    student_train = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    student_val = T.Compose([
        T.Resize(int(args.image_size * 1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # CLIP transforms
    clip_train = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    clip_val = T.Compose([
        T.Resize(int(args.image_size * 1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    # DINO transforms
    dino_train = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dino_val = T.Compose([
        T.Resize(int(args.image_size * 1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # FastSAM transforms - SIN NORMALIZACIÓN para evitar el warning
    # FastSAM espera imágenes en rango [0, 1]
    fastsam_train = T.Compose([
        T.RandomResizedCrop(640, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Solo ToTensor, sin normalización
    ])
    
    fastsam_val = T.Compose([
        T.Resize(640),
        T.CenterCrop(640),
        T.ToTensor(),  # Solo ToTensor, sin normalización
    ])
    
    # Create datasets
    train_ds = ImageFolderPILTriple(
        os.path.join(args.data_dir, args.train_subdir),
        student_train, clip_train, dino_train, fastsam_train
    )
    
    val_ds = ImageFolderPILTriple(
        os.path.join(args.data_dir, args.val_subdir),
        student_val, clip_val, dino_val, fastsam_val
    )
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Create student model
    student = TheiaStudent(
        backbone_name='vit_tiny_patch16_224',
        proj_clip=512,
        proj_dino=384,
        proj_fastsam=256
    )
    
    # Apply gradient checkpointing if requested
    if args.grad_checkpoint:
        try:
            if hasattr(student.backbone, 'set_grad_checkpointing'):
                student.backbone.set_grad_checkpointing(True)
                print("Enabled gradient checkpointing on backbone.")
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")
    
    student.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Load teacher models
    print("Loading CLIP teacher...")
    clip_model, clip_preproc, clip_extract_fn = load_clip(device)
    
    print("Loading DINOv2 teacher...")
    dino_model, dino_extract_fn = load_dinov2(device, variant="dinov2_vits14")
    
    print(f"Loading FastSAM teacher ({args.fastsam_variant})...")
    fastsam_model, fastsam_extract_fn = load_fastsam(device, variant=args.fastsam_variant)
    
    # Freeze teacher models
    for model in [clip_model, dino_model]:
        if model is not None:
            for p in model.parameters():
                p.requires_grad = False
    
    if fastsam_model is not None:
        for p in fastsam_model.parameters():
            p.requires_grad = False
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # Teacher caching
    teacher_cache = None
    if args.cache_teacher:
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        
        # Check if cache already exists
        need_cache = False
        if len(train_ds) > 0:
            sample_path = Path(train_ds.samples[0][0]).stem
            cache_files = ['clip', 'dino', 'fastsam']
            for suffix in cache_files:
                if not (cache_root / f"{sample_path}.{suffix}.pt").exists():
                    need_cache = True
                    break
        else:
            need_cache = True
        
        if need_cache:
            print("Caching teacher embeddings...")
            
            # Use smaller batch size for caching to save memory
            cache_bs = max(4, min(16, args.batch_size // 4))
            
            # Cache for training set
            cache_train_loader = DataLoader(
                train_ds, 
                batch_size=cache_bs, 
                shuffle=False,
                num_workers=max(2, args.num_workers // 2),
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            cache_teacher_embeddings(cache_train_loader, clip_extract_fn, 
                                    cache_root, device, suffix="clip")
            cache_teacher_embeddings(cache_train_loader, dino_extract_fn, 
                                    cache_root, device, suffix="dino")
            cache_teacher_embeddings(cache_train_loader, fastsam_extract_fn, 
                                    cache_root, device, suffix="fastsam")
            
            # Cache for validation set
            cache_val_loader = DataLoader(
                val_ds,
                batch_size=cache_bs,
                shuffle=False,
                num_workers=max(2, args.num_workers // 2),
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            cache_teacher_embeddings(cache_val_loader, clip_extract_fn, 
                                    cache_root, device, suffix="clip")
            cache_teacher_embeddings(cache_val_loader, dino_extract_fn, 
                                    cache_root, device, suffix="dino")
            cache_teacher_embeddings(cache_val_loader, fastsam_extract_fn, 
                                    cache_root, device, suffix="fastsam")
            
            print("Teacher caching completed.")
        
        teacher_cache = cache_root
    
    # Resume from checkpoint if requested
    best_val = float('inf')
    best_path = out_dir / "theia_triple_best.pt"
    start_epoch = 0
    
    if args.resume and best_path.exists():
        print(f"Resuming from checkpoint: {best_path}")
        try:
            student.load_state_dict(torch.load(best_path, map_location=device))
            print("Successfully loaded student weights.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(
            student, device, train_loader, optimizer, scaler,
            clip_extract_fn, dino_extract_fn, fastsam_extract_fn,
            {'fp16': args.fp16}, teacher_cache
        )
        
        # Validate
        val_loss = validate(
            student, device, val_loader,
            clip_extract_fn, dino_extract_fn, fastsam_extract_fn,
            {'fp16': args.fp16}, teacher_cache, max_batches=100
        )
        
        # Update learning rate
        scheduler.step()
        
        t1 = time.time()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Time: {t1-t0:.1f}s")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save epoch checkpoint
        epoch_ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if args.fp16 else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        torch.save(epoch_ckpt, out_dir / f"theia_triple_epoch{epoch+1}.pth")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), best_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best model saved at: {best_path}")
    print(f"{'='*50}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    student.eval()
    final_val_loss = validate(
        student, device, val_loader,
        clip_extract_fn, dino_extract_fn, fastsam_extract_fn,
        {'fp16': args.fp16}, teacher_cache, max_batches=None
    )
    print(f"Final validation loss: {final_val_loss:.6f}")

if __name__ == "__main__":
    main()