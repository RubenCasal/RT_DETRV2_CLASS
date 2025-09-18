from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import json
from PIL import Image
from torch.utils.data import Dataset


class CocoDS(Dataset):
        def __init__(self, root: Path, split: str, ann_path: Path, old2new: Dict[int, int]):
            self.root = Path(root)
            self.split = split
            self.ann_path = Path(ann_path)
            self.old2new = old2new
            js = json.loads(self.ann_path.read_text())
            self.images: List[Dict[str, Any]] = js["images"]
            self.ann_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for a in js["annotations"]:
                self.ann_by_img[a["image_id"]].append({
                    "bbox": a["bbox"],
                    "category_id": self.old2new[int(a["category_id"])],
                    "iscrowd": a.get("iscrowd", 0),
                    "area": a.get("area", a["bbox"][2]*a["bbox"][3]),
                })
        def __len__(self): return len(self.images)
        def __getitem__(self, i: int) -> Dict[str, Any]:
            info = self.images[i]
            img_path = self.root / "images" / self.split / info["file_name"]
            if not img_path.exists():
                img_path = self.root / "images" / info["file_name"]
            img = Image.open(img_path).convert("RGB")
            anns = self.ann_by_img.get(info["id"], [])
            return {
                "image": img,
                "image_id": info["id"],
                "width": info.get("width", img.width),
                "height": info.get("height", img.height),
                "annotations": anns,
            }

  