import scipy.io as sio
import os

META_PATH = "./devkit/data/meta_det.mat"
GT_PATH = "./devkit/data/ILSVRC2013_clsloc_validation_ground_truth.txt"

OUTPUT_TXT = "./val_wnids.txt"
OUTPUT_CSV = "./val_mapping.csv"


print("Cargando meta_clsloc.mat...")
mat = sio.loadmat(META_PATH)
synsets = mat['synsets']

id_to_wnid = {}

for entry in synsets[0]:    
    s = entry
    ilsvrc_id = int(s[0][0][0])     # ID oficial interno
    wnid = s[1][0]            # string tipo "nxxxxx"
    id_to_wnid[ilsvrc_id] = wnid
print(id_to_wnid)
print(f"Total synsets cargados: {len(id_to_wnid)}")

print("Cargando ground truth...")
with open(GT_PATH, "r") as f:
    gt_ids = [int(x.strip()) for x in f.readlines()]

wnids_val = []

for i, cls_id in enumerate(gt_ids):
    if cls_id not in id_to_wnid:
        raise ValueError(f"ERROR: ID {cls_id} no existe en meta_clsloc.mat (línea {i})")
    wnids_val.append(id_to_wnid[cls_id])

print("Guardando val_wnids.txt...")
with open(OUTPUT_TXT, "w") as f:
    for wnid in wnids_val:
        f.write(wnid + "\n")

print("Guardando val_mapping.csv...")
with open(OUTPUT_CSV, "w") as f:
    f.write("image,wnid\n")
    for i, wnid in enumerate(wnids_val, start=1):
        img_name = f"ILSVRC2012_val_{i:08d}.JPEG"
        f.write(f"{img_name},{wnid}\n")

print(f"Archivos guardados:\n - {OUTPUT_TXT}\n - {OUTPUT_CSV}")
print("Terminado.")