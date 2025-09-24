# Coco_DS_Augmentation.py
from pathlib import Path
from typing import Dict, List, Any
import albumentations as A
import numpy as np
from PIL import Image

from Coco_DS import CocoDS


class CocoDSAug(CocoDS):
    """
    Dataset COCO con data augmentation on-the-fly usando Albumentations.
    Mantiene el mismo esquema de salida que CocoDS y rellena claves requeridas
    por RTDetrImageProcessor (area, iscrowd).
    """

    def __init__(self, root: Path, split: str, ann_path: Path, old2new: Dict[int, int], augmentations):
        super().__init__(root, split, ann_path, old2new)
        self.tf = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(
                format="coco",               # [x, y, w, h] en píxeles
                label_fields=["category_ids"],
                min_visibility=0.2
            ),
        )

    def __getitem__(self, i: int) -> Dict[str, Any]:
        info = self.images[i]

        # Resuelve ruta como en CocoDS
        img_path = self.root / "images" / self.split / info["file_name"]
        if not img_path.exists():
            img_path = self.root / "images" / info["file_name"]

        img = Image.open(img_path).convert("RGB")
        anns = self.ann_by_img.get(info["id"], [])

        # Extrae bboxes/labels (ids ya contiguos por CocoDS)
        bboxes = [a["bbox"] for a in anns]
        labels = [a["category_id"] for a in anns]

        if len(bboxes) > 0:
            out = self.tf(image=np.array(img), bboxes=bboxes, category_ids=labels)

            aug_img = out["image"]
            aug_bboxes = out["bboxes"]
            aug_labels = out["category_ids"]

            # Reconstruye anotaciones con area/iscrowd y filtra degeneradas
            anns_aug: List[Dict[str, Any]] = []
            for b, c in zip(aug_bboxes, aug_labels):
                x, y, w, h = map(float, b)
                if w > 0.0 and h > 0.0:
                    anns_aug.append({
                        "bbox": [x, y, w, h],
                        "category_id": int(c),
                        "area": float(w * h),
                        "iscrowd": 0
                    })

            # Si quedan cajas válidas, usa imagen/a.n.n.s aumentados
            if len(anns_aug) > 0:
                img = Image.fromarray(aug_img)
                anns = anns_aug
            else:
                # Si se pierden todas las cajas por la aug, conserva el ejemplo original
                # pero asegúrate de que tenga area/iscrowd
                anns = [
                    {
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "category_id": int(c),
                        "area": float(w * h),
                        "iscrowd": 0
                    }
                    for (x, y, w, h), c in zip(bboxes, labels)
                    if w > 0.0 and h > 0.0
                ]
        else:
            # Imagen sin anotaciones: devuelve tal cual (sin aug de bboxes)
            # (RTDetrImageProcessor soporta lista vacía de anotaciones)
            anns = []

        return {
            "image": img,
            "image_id": info["id"],
            "width": info.get("width", img.width),
            "height": info.get("height", img.height),
            "annotations": anns,
        }
