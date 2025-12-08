import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch.nn as nn
import random

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = "runs/theia_mini/theia_mini_best.pt"
N_SAMPLES = 0

# -------- Student Model --------
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
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head_clip = ProjectionHead(feat_dim, proj_clip)
        self.head_dino = ProjectionHead(feat_dim, proj_dino)

    def forward(self, x):
        feat = self.backbone(x)
        clip_out = torch.nn.functional.normalize(self.head_clip(feat), dim=-1)
        dino_out = torch.nn.functional.normalize(self.head_dino(feat), dim=-1)
        return clip_out, dino_out, feat


# -------- Retrieval Utility --------
def load_image(path):
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tf(Image.open(path).convert("RGB"))

def cosine_similarity(a, b):
    return (a @ b.T).cpu().numpy()


def show_results(query_path, top_paths):
    fig, axs = plt.subplots(1, len(top_paths) + 1, figsize=(16, 4))

    # Query
    axs[0].imshow(Image.open(query_path))
    axs[0].set_title("Query")
    axs[0].axis("off")

    # Top-K
    for i, p in enumerate(top_paths):
        axs[i+1].imshow(Image.open(p))
        axs[i+1].set_title(f"Top {i+1}")
        axs[i+1].axis("off")

    plt.tight_layout()
    plt.show()


# -------- MAIN --------

image = "query.jpg"
dataset_dir = "test"
k = 5

# Load model
model = TheiaStudent()
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model = model.to(DEVICE).eval()

# Load query
query_img = load_image(image).unsqueeze(0).to(DEVICE)

# Compute query embedding
with torch.no_grad():
    _, dino_query, _ = model(query_img)

# Precompute gallery embeddings
gallery_paths = []
gallery_embeds = []

all_image_paths = []
for fname in os.listdir(dataset_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(dataset_dir, fname)
        all_image_paths.append(path)
if N_SAMPLES == 0:
    selected_paths = all_image_paths
elif len(all_image_paths) > N_SAMPLES:
    selected_paths = random.sample(all_image_paths, N_SAMPLES) #
else:
    selected_paths = all_image_paths
    print(f"Solo se encontraron {len(all_image_paths)} imágenes, procesando todas.")

for path in tqdm(selected_paths):
    img = load_image(path).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, emb, _ = model(img)

    gallery_paths.append(path)
    gallery_embeds.append(emb.cpu().numpy()[0])
    
# Compute similarity
sims = gallery_embeds @ dino_query.cpu().numpy()[0]

# Top-K
top_idx = np.argsort(-sims)[:k]
top_paths = [gallery_paths[i] for i in top_idx]

# Show result
show_results(image, top_paths)


