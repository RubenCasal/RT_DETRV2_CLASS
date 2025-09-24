# ==============================
# trainer_finetuned.py
# ==============================

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    RTDetrV2Config, RTDetrV2ForObjectDetection,
    AutoImageProcessor, TrainingArguments, Trainer
)
from Coco_DS import CocoDS
from Coco_DS_Augmentation import CocoDSAug
from utils import sig_params


class RtDetvr2Trainer:
    """
    Trainer ligero para RT-DETRv2 que permite:
      - Entrenar desde config (scratch o sólo backbone pretrain)
      - O cargar un checkpoint completo COCO-pretrained con `from_checkpoint`
    """

    def __init__(
        self,
        num_labels: int,
        old2new: Dict[int, int],
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        train_ann: Dict[str, Any],
        val_ann: Dict[str, Any],
        train_ann_path: str | Path,
        val_ann_path: str | Path,
        model_config: Optional[RTDetrV2Config],
        training_config: Dict[str, Any],
        augmentations: Any,
        root: str | Path,
        image_size: Tuple[int, int] = (256, 256),
        from_checkpoint: Optional[str] = None,   # si se pasa, carga modelo COCO-pretrained
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
        self.augmentations = augmentations

        # Processor consistente con el checkpoint (resize/normalize/pixel_mask)
        Ht, Wt = image_size
        ckpt_for_processor = from_checkpoint or "PekingU/rtdetr_v2_r50vd"
        self.processor = AutoImageProcessor.from_pretrained(
            ckpt_for_processor,
            do_resize=True,
            size={"height": int(Ht), "width": int(Wt)},
        )

        # Modelo: desde checkpoint completo o desde config
        if from_checkpoint is not None:
            self.model = RTDetrV2ForObjectDetection.from_pretrained(
                from_checkpoint,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,  # rehace la head para tus clases
            )
        else:
            assert self.cfg is not None, "model_config debe proporcionarse si no usas from_checkpoint"
            self.model = RTDetrV2ForObjectDetection(self.cfg)

        if hasattr(self.model, "config"):
            self.model.config.return_dict = True

        # Datasets
        self.train_ds = CocoDSAug(
            self.root, "train", self.train_ann, self.old2new, augmentations=self.augmentations
        )
        self.val_ds = CocoDS(self.root, "val", self.val_ann, self.old2new)

        # Collate: genera pixel_values/labels (y pixel_mask si aplica)
        def _collate(batch: List[Dict[str, Any]]):
            images = [b["image"] for b in batch]  # PIL RGB tras Albumentations
            anns = [{"image_id": b["image_id"], "annotations": b["annotations"]} for b in batch]

            bf = self.processor(
                images=images,
                annotations=anns,
                return_tensors="pt",
            )
            out = {"pixel_values": bf["pixel_values"], "labels": bf["labels"]}
            if "pixel_mask" in bf:
                out["pixel_mask"] = bf["pixel_mask"]
            return out

        self.collate_fn = _collate

        # TrainingArguments
        ta_params = sig_params(TrainingArguments.__init__)
        ta_kwargs = dict(self.training_config)

        if "evaluation_strategy" in ta_params and "evaluation_strategy" not in ta_kwargs:
            ta_kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in ta_params and "eval_strategy" not in ta_kwargs:
            ta_kwargs["eval_strategy"] = "steps"
        if "save_strategy" in ta_params and "save_strategy" not in ta_kwargs:
            ta_kwargs["save_strategy"] = "steps"
        if "label_names" in ta_params:
            ta_kwargs["label_names"] = ["labels"]

        self.args = TrainingArguments(**ta_kwargs)

        # Trainer
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
        def to_t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return x

        if hasattr(obj, "logits") and hasattr(obj, "pred_boxes"):
            return obj.logits.detach().cpu(), obj.pred_boxes.detach().cpu()

        if isinstance(obj, dict):
            if "logits" in obj and "pred_boxes" in obj:
                return to_t(obj["logits"]).detach().cpu(), to_t(obj["pred_boxes"]).detach().cpu()
            for v in obj.values():
                try:
                    L, B = self._extract_logits_boxes(v)
                    return L, B
                except Exception:
                    pass

        if isinstance(obj, (list, tuple)):
            arrays: List[torch.Tensor] = []
            for x in obj:
                try:
                    L, B = self._extract_logits_boxes(x)
                    return L, B
                except Exception:
                    pass
                x = to_t(x)
                if torch.is_tensor(x) and x.ndim >= 3:
                    arrays.append(x)

            boxes_cands = [a for a in arrays if a.size(-1) == 4]
            if boxes_cands:
                boxes = max(boxes_cands, key=lambda a: (a.shape[1], a.shape[-1]))
                B_, Q_ = boxes.shape[0], boxes.shape[1]
                logits_cands = [a for a in arrays if a.shape[:2] == (B_, Q_) and a.size(-1) != 4]
                logits_pref = [a for a in logits_cands if a.size(-1) in (self.num_labels, self.num_labels + 1)]
                logits = (
                    logits_pref[0]
                    if logits_pref
                    else (logits_cands[0] if logits_cands else None)
                )
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
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(path)
