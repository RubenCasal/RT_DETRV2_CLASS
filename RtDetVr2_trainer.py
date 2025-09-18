# rtdetrv2_trainer.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, Any
import inspect, json, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    RTDetrV2Config, RTDetrV2ForObjectDetection,
    RTDetrImageProcessor, TrainingArguments, Trainer
)
from compute_MAP import ComputeCOCOEval
from Coco_DS import CocoDS
from utils import sig_params

class RtDetvr2Trainer:
    def __init__(
           
        self,
        num_labels,
        old2new,
        id2label,
        label2id,
        train_ann,
        val_ann,
        train_ann_path,
        val_ann_path,
        model_config,
        training_config,
        root: str | Path,
   
        image_size: Tuple[int, int] = (256, 256),

    ):
        self.root = root
        self.num_labels = num_labels
        self.old2new = old2new
        self.label2id = label2id
        self.id2label = id2label
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.train_ann_path = train_ann_path
        self.val_ann_path = val_ann_path
        self.cfg = model_config
        self.training_config = training_config
    

        self.processor = RTDetrImageProcessor()
       
        self.model = RTDetrV2ForObjectDetection(self.cfg)



        # Fuerza salidas tipo ModelOutput (evita tuples raros)
        if hasattr(self.model, "config"):
            self.model.config.return_dict = True

        self.train_ds = CocoDS(self.root, "train", self.train_ann, self.old2new)
        self.val_ds   = CocoDS(self.root, "val",   self.val_ann,   self.old2new)

        Ht, Wt = image_size
        def _collate(batch: List[Dict[str, Any]]):
            images = [b["image"] for b in batch]
            anns   = [{"image_id": b["image_id"], "annotations": b["annotations"]} for b in batch]
            bf = self.processor(
                images=images,
                annotations=anns,
                return_tensors="pt",
                do_resize=True,
                size={"height": int(Ht), "width": int(Wt)},
            )
            return {"pixel_values": bf["pixel_values"], "labels": bf["labels"]}
        self.collate_fn = _collate

        ta_params = sig_params(TrainingArguments.__init__)
        ta_kwargs = self.training_config


        if "evaluation_strategy" in ta_params:
            ta_kwargs["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in ta_params:
            ta_kwargs["eval_strategy"] = "epoch"
        if "save_strategy" in ta_params:
            ta_kwargs["save_strategy"] = "epoch"
        if "label_names" in ta_params:
            ta_kwargs["label_names"] = ["labels"]
        self.args = TrainingArguments(**ta_kwargs)

        tr_params = sig_params(Trainer.__init__)
        tr_kwargs = dict(
            model=self.model,
            args=self.args,
            data_collator=self.collate_fn,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
        )
        if "preprocess_logits_for_metrics" in tr_params:
            tr_kwargs["preprocess_logits_for_metrics"] = self.preprocess_logits_for_metrics
        if "compute_metrics" in tr_params:
            tr_kwargs["compute_metrics"] = self._noop_metrics

        self.trainer = Trainer(**tr_kwargs)

    # ---------- extractor robusto ----------
    def _extract_logits_boxes(self, obj: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (logits, boxes) en forma torch.Tensor (CPU).
        - Soporta ModelOutput/dict/list/tuple anidados.
        - Empareja por (B,Q,*) con *!=4 frente a (B,Q,4).
        - Prioriza logits con última dim == num_labels o num_labels+1.
        """
        def to_t(x):
            if isinstance(x, np.ndarray): return torch.from_numpy(x)
            return x

        # ModelOutput / dict
        if hasattr(obj, "logits") and hasattr(obj, "pred_boxes"):
            return obj.logits.detach().cpu(), obj.pred_boxes.detach().cpu()
        if isinstance(obj, dict):
            if "logits" in obj and "pred_boxes" in obj:
                return to_t(obj["logits"]).detach().cpu(), to_t(obj["pred_boxes"]).detach().cpu()
            # buscar en valores
            for v in obj.values():
                try:
                    L, B = self._extract_logits_boxes(v)
                    return L, B
                except Exception:
                    pass

        # list / tuple
        if isinstance(obj, (list, tuple)):
            # Recolecta tensores candidatos
            arrays: List[torch.Tensor] = []
            for x in obj:
                # recursion primero (si alguno ya es dict/ModelOutput)
                try:
                    L, B = self._extract_logits_boxes(x)
                    return L, B
                except Exception:
                    pass
                x = to_t(x)
                if torch.is_tensor(x) and x.ndim >= 3:
                    arrays.append(x)

            # Emparejado por forma (B,Q,*)
            boxes_cands = [a for a in arrays if a.size(-1) == 4]
            if boxes_cands:
                # elige el mayor Q (más consultas)
                boxes = max(boxes_cands, key=lambda a: (a.shape[1], a.shape[-1]))
                B_, Q_ = boxes.shape[0], boxes.shape[1]
                # logits con misma (B,Q,*) y * != 4
                logits_cands = [a for a in arrays if a.shape[:2] == (B_, Q_) and a.size(-1) != 4]
                # prioridad: última dim == num_labels o num_labels+1
                logits_pref = [a for a in logits_cands if a.size(-1) in (self.num_labels, self.num_labels + 1)]
                logits = (logits_pref[0] if logits_pref else (logits_cands[0] if logits_cands else None))
                if logits is not None:
                    return logits.detach().cpu(), boxes.detach().cpu()

        raise RuntimeError("No pude extraer (logits, pred_boxes) de la salida del modelo.")

    # ---------- Hooks métricas ----------
    def preprocess_logits_for_metrics(self, outputs, labels):
        logits, boxes = self._extract_logits_boxes(outputs)
        return (logits, boxes)

    @staticmethod
    def _noop_metrics(eval_pred) -> Dict[str, float]:
        return {}

    def attach_map_evaluator(self, compute_metrics_fn: Callable):
        self.trainer.compute_metrics = compute_metrics_fn

    # ---------- API ----------
    def freeze_backbone(self):
        if hasattr(self.model, "backbone"):
            for p in self.model.backbone.parameters():
                p.requires_grad = False
    def unfreeze_backbone(self):
        if hasattr(self.model, "backbone"):
            for p in self.model.backbone.parameters():
                p.requires_grad = True
    def train(self):
        return self.trainer.train()
    def evaluate(self):
        return self.trainer.evaluate()
    @torch.no_grad()
    def predict_image(self, image_path: str | Path, threshold: float = 0.5, device: str = "cpu"):
        self.model.to(device).eval()
        img = Image.open(image_path).convert("RGB")
        H, W = img.height, img.width
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        det = self.processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(H, W)], device=device), threshold=threshold
        )[0]
        det["labels"] = [self.id2label[int(i)] for i in det["labels"]]
        return det
    def save(self, path: str | Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(path)
