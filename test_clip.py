import torch
import clip
from PIL import Image
import torchvision.transforms as T

from train_minitheia import TheiaStudent  # Ajusta si tu archivo tiene otro nombre
import warnings
warnings.filterwarnings('ignore')
# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = "runs/theia_mini/theia_mini_best.pt"

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
# CARGAR MODELOS
# -----------------------------
print("Cargando CLIP teacher...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)

print("Cargando Mini-Theia...")
student = TheiaStudent().to(DEVICE)
student.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
student.eval()

to_tensor = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

# -----------------------------
# EXTRACCIÓN
# -----------------------------
def get_student_image_emb(path):
    img = Image.open(path).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            clip_emb, dino_emb, fused = student(tensor)  # fused = 192 dims
    return clip_emb, fused

def get_clip_teacher_image_emb(path):
    img = Image.open(path).convert("RGB")
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(tensor)
    return image_features

def get_clip_text_emb(text):
    with torch.no_grad():
        text_tokens = clip.tokenize(text).to(DEVICE)
        text_features = clip_model.encode_text(text_tokens)
    return text_features


# -----------------------------
# SIMILITUD
# -----------------------------
def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()


# -----------------------------
# EVALUACIÓN
# -----------------------------
print("\n=== Evaluación CLIP texto ↔ imagen ===")

for text in TEXT_PROMPTS:
    print(f"\n⚫ Texto: {text}")

    teacher_text = get_clip_text_emb(text)

    best_img = None
    best_teacher_score = -1
    best_student_score = -1

    for img_path in IMAGE_PATHS:
        teacher_img = get_clip_teacher_image_emb(img_path)
        student_img, fused = get_student_image_emb(img_path)

        s_t = cosine(student_img, teacher_text)
        t_t = cosine(teacher_img, teacher_text)

        print(f"  Imagen: {img_path}")
        print(f"    Teacher CLIP score: {t_t:.4f}")
        print(f"    Student Mini-Theia score: {s_t:.4f}")

        if s_t > best_student_score:
            best_student_score = s_t
            best_img = img_path

    print(f"➡ Mini-Theia considera más similar: **{best_img}**")
