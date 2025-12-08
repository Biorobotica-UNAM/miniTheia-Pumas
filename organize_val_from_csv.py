import os
import csv
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default="val", help="Carpeta con las imágenes de validación sueltas")
    parser.add_argument("--csv", type=str, default="val_mapping.csv", help="val_mapping.csv con columnas: image, wnid")
    parser.add_argument("--out-dir", required=False, help="Carpeta destino (default: <parent>/val_sorted)")
    parser.add_argument("--copy", action="store_true", help="Copiar en vez de mover")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime sin mover/copiar")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    csv_path = Path(args.csv)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = val_dir.parent / "val_sorted"

    # Cargar mapping desde CSV
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image"]
            wnid = row["wnid"]
            mapping[img] = wnid

    print(f"Cargadas {len(mapping)} entradas desde {csv_path}")

    # Crear carpetas destino
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Procesar imágenes
    files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(".jpeg")])

    for f in tqdm(files, desc="Organizando VAL"):
        if f not in mapping:
            print(f"[WARN] No se encontró mapeo para {f}, se omite.")
            continue

        wnid = mapping[f]
        class_dir = out_dir / wnid

        if not args.dry_run:
            class_dir.mkdir(parents=True, exist_ok=True)

        src = val_dir / f
        dst = class_dir / f

        if args.dry_run:
            print(f"{src} -> {dst}")
        else:
            if args.copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))

    print("✔ Terminado. Carpeta final:", out_dir)

if __name__ == "__main__":
    main()
