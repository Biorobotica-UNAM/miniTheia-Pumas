# eval_fastsam_component.py
import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
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

def compute_similarity_matrix(embeddings):
    """Calcula matriz de similitud entre todos los embeddings"""
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j])
    
    return similarity_matrix

def visualize_similarity_matrix(paths, similarity_matrix, title="Matriz de Similitud FastSAM"):
    """Visualiza la matriz de similitud"""
    n = len(paths)
    
    fig, ax = plt.subplots(figsize=(max(8, n//2), max(6, n//3)))
    
    im = ax.imshow(similarity_matrix, cmap='viridis')
    
    # Configurar etiquetas
    short_names = [os.path.basename(p)[:10] for p in paths]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)
    
    # Añadir valores en las celdas
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="w" if similarity_matrix[i, j] < 0.5 else "k",
                          fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Similitud Coseno')
    plt.tight_layout()
    plt.show()

def show_results(query_path, top_paths, scores, title="Resultados FastSAM"):
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
    parser = argparse.ArgumentParser(description='Evaluación del componente FastSAM del modelo Theia')
    parser.add_argument("--query", required=True, help="Imagen de consulta")
    parser.add_argument("--gallery", required=True, help="Directorio con imágenes para búsqueda")
    parser.add_argument("--k", type=int, default=5, help="Número de resultados a mostrar")
    parser.add_argument("--matrix", action='store_true', help="Mostrar matriz de similitud completa")
    parser.add_argument("--output", type=str, default=None, help="Guardar resultados en archivo (opcional)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
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
        student.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    student = student.to(device).eval()
    
    # 2. Cargar y procesar imagen de consulta
    print(f"\n🔍 Procesando imagen de consulta: {args.query}")
    if not os.path.exists(args.query):
        print(f"❌ No se encuentra la imagen de consulta: {args.query}")
        return
    
    query_img = load_image(args.query).unsqueeze(0).to(device)
    
    # Extraer embedding FastSAM de la consulta
    with torch.no_grad():
        _, _, fastsam_query, _ = student(query_img)
    
    print(f"✅ Embedding FastSAM extraído (dimensión: {fastsam_query.shape[1]})")
    
    # 3. Procesar galería de imágenes
    print(f"\n📂 Explorando galería: {args.gallery}")
    if not os.path.exists(args.gallery):
        print(f"❌ No se encuentra el directorio de galería: {args.gallery}")
        return
    
    gallery_paths = []
    gallery_embeds = []
    
    # Listar imágenes válidas
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(args.gallery) 
                  if f.lower().endswith(valid_extensions)]
    
    if len(image_files) == 0:
        print("❌ No se encontraron imágenes en la galería")
        return
    
    print(f"Encontradas {len(image_files)} imágenes en la galería")
    
    # Extraer embeddings de todas las imágenes
    for fname in tqdm(image_files, desc="Extrayendo embeddings FastSAM"):
        path = os.path.join(args.gallery, fname)
        
        try:
            img = load_image(path).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, _, emb, _ = student(img)
            
            gallery_paths.append(path)
            gallery_embeds.append(emb.cpu().numpy()[0])
        except Exception as e:
            print(f"⚠️  Error procesando {fname}: {e}")
            continue
    
    if len(gallery_embeds) == 0:
        print("❌ No se pudieron extraer embeddings de la galería")
        return
    
    gallery_embeds = np.stack(gallery_embeds)
    print(f"✅ {len(gallery_embeds)} embeddings extraídos exitosamente")
    
    # 4. Calcular similitudes
    print("\n📊 Calculando similitudes coseno...")
    sims = gallery_embeds @ fastsam_query.cpu().numpy()[0]
    
    # 5. Obtener top-K resultados
    top_idx = np.argsort(-sims)[:args.k]
    top_paths = [gallery_paths[i] for i in top_idx]
    top_scores = sims[top_idx]
    
    # 6. Mostrar resultados
    print(f"\n{'='*60}")
    print("🎯 TOP-K RESULTADOS (Componente FastSAM)")
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
                 title=f"Retrieval FastSAM - Top {args.k} resultados")
    
    # 8. Mostrar matriz de similitud si se solicita
    if args.matrix and len(gallery_embeds) <= 20:  # Limitar para visualización clara
        print("\n📐 Calculando matriz de similitud completa...")
        all_paths = [args.query] + gallery_paths[:10]  # Limitar a 10 imágenes para visualización
        all_embeds = np.vstack([fastsam_query.cpu().numpy()[0], gallery_embeds[:10]])
        
        similarity_matrix = compute_similarity_matrix(all_embeds)
        visualize_similarity_matrix(all_paths, similarity_matrix, 
                                    "Matriz de Similitud FastSAM (Query + 10 imágenes)")
    
    # 9. Guardar resultados si se solicita
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Consulta: {args.query}\n")
            f.write(f"Galería: {args.gallery}\n")
            f.write(f"Total imágenes: {len(gallery_paths)}\n")
            f.write(f"K: {args.k}\n")
            f.write(f"Componente: FastSAM (256-dim)\n\n")
            f.write("Rank,Similaridad,Imagen\n")
            for i, (path, score) in enumerate(zip(top_paths, top_scores)):
                img_name = os.path.basename(path)
                f.write(f"{i+1},{score:.4f},{img_name}\n")
        print(f"✅ Resultados guardados en: {args.output}")
    
    print(f"\n{'='*60}")
    print("✅ Evaluación del componente FastSAM completada")
    print(f"{'='*60}")
    
    # 10. Estadísticas adicionales
    print("\n📈 Estadísticas de similitud FastSAM:")
    print(f"   • Similaridad máxima: {np.max(sims):.4f}")
    print(f"   • Similaridad mínima: {np.min(sims):.4f}")
    print(f"   • Similaridad promedio: {np.mean(sims):.4f}")
    print(f"   • Desviación estándar: {np.std(sims):.4f}")
    print(f"   • Similaridad mediana: {np.median(sims):.4f}")

if __name__ == "__main__":
    main()