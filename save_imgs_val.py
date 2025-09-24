from transformers import TrainerCallback
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch, random


class SaveValImgsInCkpt(TrainerCallback):
    """
    Callback que, cada vez que Trainer guarda un checkpoint, escribe K imágenes
    de validación con:
      - GT (verde): val{j}_label.jpg
      - Pred (rojo): val{j}_pred.jpg
    en: {output_dir}/checkpoint-{global_step}/val_vis/

    Opcionalmente registra las imágenes en Weights & Biases.
    """

    def __init__(
        self,
        processor,
        val_ds,
        id2label,
        k: int = 4,
        thr: float = 0.01,
        infer_size: tuple | None = None,  # (H, W). Si None, usa tamaño nativo
        log_to_wandb: bool = False,
        wandb_prefix: str = "val_vis",
        seed: int = 123,
    ):
        self.p = processor
        self.ds = val_ds
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.k = int(k)
        self.thr = float(thr)
        self.infer_size = infer_size
        self.log_to_wandb = log_to_wandb
        self.wandb_prefix = wandb_prefix.rstrip("/")
        self.rng = random.Random(seed)

        # Fuente para texto (fallback a default si no está disponible)
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

    def _draw_gt(self, img: Image.Image, anns):
        im = img.copy().convert("RGB")
        dr = ImageDraw.Draw(im)
        for a in anns:
            x, y, w, h = a["bbox"]
            cls = int(a.get("category_id", -1))
            name = self.id2label.get(cls, str(cls))
            dr.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
            dr.text((x, max(0, y - 12)), name, fill=(0, 255, 0), font=self.font)
        return im

    def _draw_pred(self, img: Image.Image, det):
        im = img.copy().convert("RGB")
        dr = ImageDraw.Draw(im)
        for b, sc, lb in zip(det["boxes"], det["scores"], det["labels"]):
            x1, y1, x2, y2 = [float(v) for v in b]
            name = self.id2label.get(int(lb), str(int(lb)))
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            dr.text((x1, max(0, y1 - 12)), f"{name}:{float(sc):.2f}", fill=(255, 0, 0), font=self.font)
        return im

    def on_save(self, args, state, control, **kwargs):
        # Carpeta de salida para las imágenes del checkpoint actual
        out_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "val_vis"
        out_dir.mkdir(parents=True, exist_ok=True)

        model = kwargs["model"].eval()
        device = next(model.parameters()).device

        # Muestreamos K índices de validación
        k = min(self.k, len(self.ds))
        idxs = self.rng.sample(range(len(self.ds)), k=k)

        # W&B (lazy import para no forzar dependencia)
        wb = None
        if self.log_to_wandb:
            try:
                import wandb as _wandb
                wb = _wandb
            except Exception:
                wb = None  # si no está, simplemente no logueamos

        with torch.no_grad():
            images_to_log = []  # batch de wandb.Image
            for j, idx in enumerate(idxs, 1):
                sample = self.ds[idx]
                img = sample["image"].convert("RGB")
                H, W = img.height, img.width

                # --- GT ---
                im_gt = self._draw_gt(img, sample.get("annotations", []))
                gt_path = out_dir / f"val{j}_label.jpg"
                im_gt.save(gt_path, quality=95)

                # --- Pred ---
                # Procesado con tamaño consistente si se indica
                proc_kwargs = {"images": img, "return_tensors": "pt", "do_resize": False}
                if self.infer_size is not None:
                    ih, iw = int(self.infer_size[0]), int(self.infer_size[1])
                    proc_kwargs.update({"do_resize": True, "size": {"height": ih, "width": iw}})

                inputs = self.p(**proc_kwargs).to(device)
                outputs = model(**inputs)
                det = self.p.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([(H, W)], device=device),
                    threshold=self.thr,
                )[0]

                im_pd = self._draw_pred(img, det)
                pd_path = out_dir / f"val{j}_pred.jpg"
                im_pd.save(pd_path, quality=95)

                # Añadir a W&B si está habilitado
                if wb is not None:
                    images_to_log.append(
                        wb.Image(im_gt, caption=f"GT idx={idx}")
                    )
                    images_to_log.append(
                        wb.Image(im_pd, caption=f"Pred idx={idx} thr={self.thr}")
                    )

            if wb is not None and images_to_log:
                wb.log({self.wandb_prefix: images_to_log, "global_step": state.global_step})

        return control
