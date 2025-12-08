# eval_clip_component.py
import torch
import clip
from PIL import Image
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = "runs/theia_triple/theia_triple_best.pt"

# Imágenes de prueba
IMAGE_PATHS = [
    "perro.jpg",
    "gato.jpg"
]

TEXT_PROMPTS = [
    "un perro corriendo en el césped",
    "un gato mirando a la cámara"
]

# -----------------------------
# Definición del modelo (ajustada para triple)
# -----------------------------
import torch.nn as nn
import timm

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
# CARGAR MODELOS
# -----------------------------
print("Cargando CLIP teacher...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)

print("Cargando Mini-Theia (CLIP+DINO+FastSAM)...")
student = TheiaStudent(
    backbone_name='vit_tiny_patch16_224',
    proj_clip=512,
    proj_dino=384,
    proj_fastsam=256
).to(DEVICE)

student.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
student.eval()

# Transformaciones
student_transform = T.Compose([
    T.Resize(int(224 * 1.14)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# EXTRACCIÓN
# -----------------------------
def get_student_clip_emb(path):
    img = Image.open(path).convert("RGB")
    tensor = student_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        clip_emb, _, _, _ = student(tensor)  # Solo usamos salida CLIP
    return clip_emb

def get_clip_teacher_image_emb(path):
    img = Image.open(path).convert("RGB")
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

def get_clip_text_emb(text):
    with torch.no_grad():
        text_tokens = clip.tokenize(text).to(DEVICE)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# -----------------------------
# SIMILITUD
# -----------------------------
def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

# -----------------------------
# EVALUACIÓN
# -----------------------------
print("\n=== Evaluación del componente CLIP (texto ↔ imagen) ===")
print(f"Comparando: {len(IMAGE_PATHS)} imágenes vs {len(TEXT_PROMPTS)} textos\n")

for text in TEXT_PROMPTS:
    print(f"⚫ Texto: '{text}'")
    
    # Embedding del texto (teacher CLIP)
    teacher_text = get_clip_text_emb(text)
    
    best_img = None
    best_teacher_score = -1
    best_student_score = -1
    
    for img_path in IMAGE_PATHS:
        # Embeddings de imagen
        teacher_img = get_clip_teacher_image_emb(img_path)
        student_clip_emb = get_student_clip_emb(img_path)
        
        # Calcular similitudes
        s_t = cosine(student_clip_emb, teacher_text)
        t_t = cosine(teacher_img, teacher_text)
        
        print(f"  📷 Imagen: {img_path}")
        print(f"    ✅ Teacher CLIP score: {t_t:.4f}")
        print(f"    🎓 Student (CLIP head) score: {s_t:.4f}")
        print(f"    📊 Diferencia: {abs(t_t - s_t):.4f}")
        
        if s_t > best_student_score:
            best_student_score = s_t
            best_img = img_path
        if t_t > best_teacher_score:
            best_teacher_score = t_t
    
    print(f"➡ El estudiante (componente CLIP) considera más similar: **{best_img}** (score: {best_student_score:.4f})")
    print(f"➡ El teacher CLIP considera más similar: **{[p for p in IMAGE_PATHS if p != best_img][0] if best_teacher_score > best_student_score else best_img}** (score: {best_teacher_score:.4f})")
    print("-" * 50)

# -----------------------------
# MATRIZ DE SIMILITUD
# -----------------------------
print("\n=== Matriz de similitud CLIP ===")
print("Texto ↔ Imagen (valores más altos = más similar)")
print("-" * 60)

# Encabezado
header = " " * 20
for img_path in IMAGE_PATHS:
    header += f"{img_path[:15]:>15}"
print(header)

# Filas
for i, text in enumerate(TEXT_PROMPTS):
    teacher_text = get_clip_text_emb(text)
    row = f"'{text[:18]:<18}'"
    for img_path in IMAGE_PATHS:
        student_clip_emb = get_student_clip_emb(img_path)
        teacher_img = get_clip_teacher_image_emb(img_path)
        
        s_t = cosine(student_clip_emb, teacher_text)
        t_t = cosine(teacher_img, teacher_text)
        
        # Mostrar ambos (estudiante / teacher)
        row += f"{s_t:7.3f}/{t_t:5.3f}"
    print(row)

print("\n" + "=" * 60)
print("✅ Evaluación del componente CLIP completada")