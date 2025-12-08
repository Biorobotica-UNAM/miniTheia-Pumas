# eval_dino_component.py
import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# -----------------------------
# Definición del modelo
# -----------------------------
import torch.nn as nn
import timm

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = "runs/theia_triple/theia_triple_best.pt"
N_SAMPLES = 1000

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
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head_clip = ProjectionHead(feat_dim, proj_clip)
        self.head_dino = ProjectionHead(feat_dim, proj_dino)
        self.head_fastsam = ProjectionHead(feat_dim, proj_fastsam)

    def forward(self, x):
        feat = self.backbone(x)
        clip_out = nn.functional.normalize(self.head_clip(feat), dim=-1)
        dino_out = nn.functional.normalize(self.head_dino(feat), dim=-1)
        fastsam_out = nn.functional.normalize(self.head_fastsam(feat), dim=-1)
        return clip_out, dino_out, fastsam_out, feat

# -----------------------------
# Funciones de utilidad
# -----------------------------
def load_image(path, size=224):
    """Carga y transforma una imagen para el estudiante"""
    tf = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(Image.open(path).convert("RGB"))

def cosine_similarity(a, b):
    """Calcula similitud coseno entre vectores"""
    return (a @ b.T).cpu().numpy()

def show_results(query_path, top_paths, scores, title="Resultados DINO"):
    """Muestra los resultados visualmente"""
    n_results = len(top_paths)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(4*(n_results + 1), 4))
    
    # Query
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("Query", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Top-K resultados
    for i, (path, score) in enumerate(zip(top_paths, scores)):
        axes[i+1].imshow(Image.open(path))
        axes[i+1].set_title(f"Top {i+1}\nscore: {score:.3f}", fontsize=10)
        axes[i+1].axis("off")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Función principal
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación del componente DINO del modelo Theia')
    parser.add_argument("--query", type=str, default="perro.jpg", help="Imagen de consulta")
    parser.add_argument("--gallery", type=str, default="test", help="Directorio con imágenes para búsqueda")
    parser.add_argument("--k", type=int, default=5, help="Número de resultados a mostrar")
    parser.add_argument("--output", type=str, default=None, help="Guardar resultados en archivo (opcional)")
    args = parser.parse_args()

    print(f"Usando dispositivo: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Cargar modelo
    print(f"\n📦 Cargando modelo desde: {MODEL_WEIGHTS}")
    student = TheiaStudent(
        backbone_name='vit_tiny_patch16_224',
        proj_clip=512,
        proj_dino=384,
        proj_fastsam=256
    )
    
    try:
        student.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    student = student.to(DEVICE).eval()
    
    # 2. Cargar y procesar imagen de consulta
    print(f"\n🔍 Procesando imagen de consulta: {args.query}")
    if not os.path.exists(args.query):
        print(f"❌ No se encuentra la imagen de consulta: {args.query}")
        return
    
    query_img = load_image(args.query).unsqueeze(0).to(DEVICE)
    
    # Extraer embedding DINO de la consulta
    with torch.no_grad():
        _, dino_query, _, _ = student(query_img)
    
    print(f"✅ Embedding DINO extraído (dimensión: {dino_query.shape[1]})")
    
    # 3. Procesar galería de imágenes
    print(f"\n📂 Explorando galería: {args.gallery}")
    if not os.path.exists(args.gallery):
        print(f"❌ No se encuentra el directorio de galería: {args.gallery}")
        return
    
    gallery_paths = []
    gallery_embeds = []
    
    all_image_paths = []
    for fname in os.listdir(args.gallery):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(args.gallery, fname)
            all_image_paths.append(path)

    if len(all_image_paths) > N_SAMPLES:
        selected_paths = random.sample(all_image_paths, N_SAMPLES) #
    else:
        selected_paths = all_image_paths
        print(f"Solo se encontraron {len(all_image_paths)} imágenes, procesando todas.")

    for path in tqdm(selected_paths):
        img = load_image(path).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, emb, _, _ = student(img)

        gallery_paths.append(path)
        gallery_embeds.append(emb.cpu().numpy()[0])
    
    gallery_embeds = np.stack(gallery_embeds)
    print(f"✅ {len(gallery_embeds)} embeddings extraídos exitosamente")
    
    # 4. Calcular similitudes
    print("\n📊 Calculando similitudes coseno...")
    sims = gallery_embeds @ dino_query.cpu().numpy()[0]
    
    # 5. Obtener top-K resultados
    top_idx = np.argsort(-sims)[:args.k]
    top_paths = [gallery_paths[i] for i in top_idx]
    top_scores = sims[top_idx]
    
    # 6. Mostrar resultados
    print(f"\n{'='*60}")
    print("🎯 TOP-K RESULTADOS (Componente DINO)")
    print(f"{'='*60}")
    
    print(f"Consulta: {args.query}")
    print(f"Galería: {args.gallery} ({len(gallery_paths)} imágenes)")
    print(f"K = {args.k}\n")
    
    print("Rank | Similaridad | Imagen")
    print("-" * 60)
    
    for i, (path, score) in enumerate(zip(top_paths, top_scores)):
        img_name = os.path.basename(path)
        print(f"{i+1:4d} | {score:10.4f} | {img_name}")
    
    # 7. Mostrar visualización
    print(f"\n📈 Mostrando visualización de resultados...")
    show_results(args.query, top_paths, top_scores, 
                 title=f"Retrieval DINO - Top {args.k} resultados")
    
    # 8. Guardar resultados si se solicita
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Consulta: {args.query}\n")
            f.write(f"Galería: {args.gallery}\n")
            f.write(f"Total imágenes: {len(gallery_paths)}\n")
            f.write(f"K: {args.k}\n\n")
            f.write("Rank,Similaridad,Imagen\n")
            for i, (path, score) in enumerate(zip(top_paths, top_scores)):
                img_name = os.path.basename(path)
                f.write(f"{i+1},{score:.4f},{img_name}\n")
        print(f"✅ Resultados guardados en: {args.output}")
    
    print(f"\n{'='*60}")
    print("✅ Evaluación del componente DINO completada")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()