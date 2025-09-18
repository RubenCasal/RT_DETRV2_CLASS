from pathlib import Path
import torch
import inspect, json, random
import numpy as np


def extract_dataset_labels(root_dataset):
    seed = 42

    root = Path(root_dataset).expanduser().resolve()
    ann_dir = root / "annotations"
    train_ann = find_ann(ann_dir, "train")
    val_ann   = find_ann(ann_dir, "val")
    train_ann_path, val_ann_path = train_ann, val_ann

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    cats = json.loads(train_ann.read_text())["categories"]
    old2new = {int(c["id"]): i for i, c in enumerate(cats)}
    id2label = {i: c["name"] for i, c in enumerate(cats)}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    
    return num_labels, old2new, id2label,label2id, train_ann, val_ann, train_ann_path, val_ann_path


@staticmethod
def sig_params(obj) -> set[str]:
    try:
        return set(inspect.signature(obj).parameters.keys())
    except (ValueError, TypeError):
        return set()

@staticmethod
def find_ann(ann_dir: Path, split: str) -> Path:
    cand = ann_dir / f"{split}.json"
    if cand.exists(): return cand
    for p in list(ann_dir.glob(f"instances_*{split}*.json")) + list(ann_dir.glob(f"*{split}*.json")):
        if p.exists(): return p
    raise FileNotFoundError(f"No se encontró {split}.json en {ann_dir}")