# compute_MAP.py
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import copy
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class ComputeCOCOEval:
    """
    COCOeval oficial (bbox) con saneado del GT:
      - Convierte (logits, pred_boxes) NumPy -> Torch.
      - post_process_object_detection -> (xyxy @ HxW orig).
      - Convierte a COCO results (xywh) + score + category_id (ids ORIGINALES).
      - Sanea GT: añade 'info'/ 'licenses' si faltan para evitar KeyError en loadRes().
    """
    def __init__(self,
                 processor,
                 ann_file: str,
                 val_images: List[Dict[str, Any]],
                 new2old: Optional[Dict[int, int]] = None,
                 score_thr: float = 0.001,
                 id2label: Optional[Dict[int, str]] = None):
        self.p = processor
        self.ann_file = ann_file
        self.val_images = val_images
        self.thr = float(score_thr)
        # mapping contiguo->original; si no viene, identidad
        self.new2old = {int(k): int(v) for k, v in (new2old or {}).items()}

    def _sanitize_coco_gt(self, coco_gt: COCO) -> None:
        # pycocotools espera 'info' y 'licenses' en coco_gt.dataset
        ds = coco_gt.dataset
        if "info" not in ds:
            ds["info"] = {"description": "auto-added by ComputeCOCOEval"}
        if "licenses" not in ds:
            ds["licenses"] = []

    def __call__(self, eval_pred):
        (logits, pred_boxes), _ = eval_pred.predictions, eval_pred.label_ids
        results: List[Dict[str, Any]] = []

        B = logits.shape[0]  # pueden ser np.ndarray; convertimos por-imagen
        for i in range(B):
            H = int(self.val_images[i]["height"])
            W = int(self.val_images[i]["width"])
            img_id = int(self.val_images[i]["id"])

            li = torch.as_tensor(logits[i], dtype=torch.float32)      # (Q, C+1)
            bi = torch.as_tensor(pred_boxes[i], dtype=torch.float32)  # (Q, 4) cxcywh in [0,1]

            det = self.p.post_process_object_detection(
                SimpleNamespace(logits=li.unsqueeze(0), pred_boxes=bi.unsqueeze(0)),
                target_sizes=torch.tensor([(H, W)], dtype=torch.long),
                threshold=self.thr
            )[0]  # dict: boxes(xyxy), scores, labels(new ids)

            bxyxy = det["boxes"].cpu()
            # xyxy -> xywh
            xywh = bxyxy.clone()
            xywh[:, 2:] = bxyxy[:, 2:] - bxyxy[:, :2]

            scores = det["scores"].cpu().tolist()
            labels_new = det["labels"].cpu().tolist()

            for bb, sc, cid_new in zip(xywh.tolist(), scores, labels_new):
                cid_old = self.new2old.get(int(cid_new), int(cid_new))
                results.append({
                    "image_id": img_id,
                    "category_id": cid_old,   # ids originales del JSON
                    "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    "score": float(sc),
                })

        # COCOeval
        coco_gt = COCO(self.ann_file)
        self._sanitize_coco_gt(coco_gt)  # <-- evita KeyError: 'info'

        # loadRes necesita un dataset estilo detecciones (lista de dicts)
        # si no hay preds, pasa lista vacía
        try:
            coco_dt = coco_gt.loadRes(results if results else [])
        except KeyError:
            # por si otra clave faltase en builds antiguas
            self._sanitize_coco_gt(coco_gt)
            coco_dt = coco_gt.loadRes(results if results else [])

        E = COCOeval(coco_gt, coco_dt, iouType='bbox')
        E.evaluate(); E.accumulate(); E.summarize()

        return {
            "coco/AP": float(E.stats[0]),
            "coco/AP50": float(E.stats[1]),
            "coco/AP75": float(E.stats[2]),
            "coco/AR100": float(E.stats[8]),
        }
