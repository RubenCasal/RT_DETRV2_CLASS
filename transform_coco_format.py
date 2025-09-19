#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
import shutil, random, json, os

# =====================
# CONFIG
# =====================
SRC  = Path("/media/rcasal/PortableSSD/ptfue_v2").resolve()       # contiene images/ y labels/
DEST = Path("/media/rcasal/PortableSSD/ptfue_v2_(COCO_FORMAT)").resolve()

TRAIN_RATIO = 0.85
SEED = 42

# Si conoces el mapeo de clases, rellénalo; si es None, se infiere como class_0..class_N
CLASS_NAMES: Optional[List[str]] = None
# CLASS_NAMES = ["person", "vehicles", "boat"]  # ejemplo si lo quieres fijo

# Política (pareo estricto imagen-label)
SKIP_IF_NO_LABEL = True

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =====================
# Utils
# =====================
def ensure_dirs(root: Path) -> None:
    for p in [
        root/"images/train", root/"images/val",
        root/"labels/train", root/"labels/val",
        root/"annotations"
    ]:
        p.mkdir(parents=True, exist_ok=True)

def iter_images(img_dir: Path) -> List[Path]:
    return [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def lbl_for(lbl_root: Path, stem: str) -> Optional[Path]:
    p = lbl_root / f"{stem}.txt"
    return p if p.exists() else None

def transfer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        c = int(float(parts[0])); cx, cy, w, h = map(float, parts[1:5])
        return c, cx, cy, w, h
    except Exception:
        return None

def read_yolo(txt: Path) -> List[Tuple[int,float,float,float,float]]:
    if not txt.exists(): return []
    rows=[]
    for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip(): continue
        r = parse_yolo_line(ln)
        if r: rows.append(r)
    return rows

def image_size_pillow(img_path: Path) -> Tuple[int,int]:
    # Pillow solo se usa para width/height para evitar dependencias pesadas
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size  # (W, H)

def yolo_bbox_to_coco_abs(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[float,float,float,float]:
    # De YOLO [0,1] a absolutos
    x = (cx - w/2.0) * W
    y = (cy - h/2.0) * H
    bw = w * W
    bh = h * H
    # Clip seguro al frame
    x = max(0.0, min(x, W - 1.0))
    y = max(0.0, min(y, H - 1.0))
    bw = max(0.0, min(bw, W - x))
    bh = max(0.0, min(bh, H - y))
    return x, y, bw, bh

# =====================
# COCO builder
# =====================
def build_categories(class_names: Optional[List[str]], classes_seen: Set[int]) -> List[Dict]:
    if class_names is None:
        max_id = max(classes_seen) if classes_seen else 0
        class_names = [f"class_{i}" for i in range(max_id + 1)]
    cats = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    return cats

def yolo_split_to_coco(split_pairs: List[Tuple[Path, Path]],
                       dimg: Path, dlbl: Path,
                       out_json: Path,
                       cats: List[Dict]) -> None:
    images, annotations = [], []
    ann_id = 1
    img_id = 1
    cat_ids = {c["id"] for c in cats}
    for im_src, _ in split_pairs:
        im = dimg / im_src.name
        lb = dlbl / (im_src.stem + ".txt")
        W, H = image_size_pillow(im)
        images.append({"id": img_id, "file_name": im.name, "width": W, "height": H})

        for ln in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = parse_yolo_line(ln)
            if not parsed:
                continue
            cid, cx, cy, bw, bh = parsed
            # salta clases no declaradas (por si hay huecos y CLASS_NAMES fijo)
            if cid not in cat_ids:
                continue
            x, y, w, h = yolo_bbox_to_coco_abs(cx, cy, bw, bh, W, H)
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w*h),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    out = {"images": images, "annotations": annotations, "categories": cats}
    out_json.write_text(json.dumps(out), encoding="utf-8")

# =====================
# MAIN
# =====================
def main():
    random.seed(SEED)
    img_root = SRC / "images"
    lbl_root = SRC / "labels"
    if not img_root.exists() or not lbl_root.exists():
        raise SystemExit("[ERR] SRC debe contener 'images/' y 'labels/'")

    ensure_dirs(DEST)

    # 1) recolecta pares válidos
    pairs: List[Tuple[Path, Path]] = []
    for im in iter_images(img_root):
        lb = lbl_for(lbl_root, im.stem)
        if lb is None and SKIP_IF_NO_LABEL:
            continue
        if lb is None:  # si no estrictos, puedes crear un .txt vacío aquí si lo deseas
            continue
        pairs.append((im, lb))

    if not pairs:
        raise SystemExit("[ERR] No hay pares imagen+label válidos.")

    # 2) split 85/15 (determinista)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    # 3) copia pares
    def copy_pairs(pairs: List[Tuple[Path, Path]], dimg: Path, dlbl: Path):
        for im, lb in pairs:
            transfer(im, dimg / im.name)
            transfer(lb, dlbl / (im.stem + ".txt"))

    copy_pairs(train_pairs, DEST/"images/train", DEST/"labels/train")
    copy_pairs(val_pairs,   DEST/"images/val",   DEST/"labels/val")

    # 4) inferir clases vistas para COCO (si no se pasó CLASS_NAMES)
    classes_seen: Set[int] = set()
    for _, lb in pairs:
        for r in read_yolo(lb):
            classes_seen.add(r[0])

    cats = build_categories(CLASS_NAMES, classes_seen)

    # 5) generar COCO JSONs
    yolo_split_to_coco(train_pairs, DEST/"images/train", DEST/"labels/train",
                       DEST/"annotations"/"train.json", cats)
    yolo_split_to_coco(val_pairs,   DEST/"images/val",   DEST/"labels/val",
                       DEST/"annotations"/"val.json", cats)

    # 6) resumen
    print("== DONE ==")
    print(f"Destino: {DEST}")
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")
    print(f"COCO: {DEST/'annotations/train.json'}  |  {DEST/'annotations/val.json'}")
    print(f"Clases vistas: {sorted(classes_seen)}")
    if CLASS_NAMES:
        print(f"Categories (fijas): {CLASS_NAMES}")
    else:
        print(f"Categories (inferidas): {[c['name'] for c in cats]}")

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    main()
