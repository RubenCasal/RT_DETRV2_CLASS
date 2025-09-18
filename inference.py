# inference_cv2.py
from pathlib import Path
import math
import random
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_ID = "PekingU/rtdetr_v2_r50vd"  # o ruta local a tu checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.5

def _color_from_label(lbl: str) -> tuple[int, int, int]:
    """Color BGR estable basado en el nombre de la clase."""
    rnd = random.Random(hash(lbl) & 0xFFFFFFFF)
    return (rnd.randint(64, 255), rnd.randint(64, 255), rnd.randint(64, 255))

@torch.no_grad()
def predict_image(img_path: str):
    # Modelo + processor (usa el del mismo repo para consistencia de preprocess)
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)

    # Carga imagen
    img_pil = Image.open(img_path).convert("RGB")
    H, W = img_pil.height, img_pil.width

    # Preprocess
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)

    # Forward (mixed precision si hay GPU)
    autocast_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.autocast(device_type=DEVICE, dtype=autocast_dtype):
        outputs = model(**inputs)

    # Post-proceso a escala original
    det = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(H, W)], device=DEVICE),
        threshold=CONF_THRESHOLD
    )[0]

    # Mapear ids a nombres legibles
    id2label = model.config.id2label
    det["labels"] = [id2label[int(i)] for i in det["labels"]]

    return img_pil, det

def draw_yolo_cv2(img_pil: Image.Image, det, save_path: str | None = None, show: bool = True):
    """
    Dibuja con OpenCV: caja gruesa + etiqueta (label + score).
    Devuelve la imagen PIL con anotaciones.
    """
    # PIL->OpenCV (RGB->BGR)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    # Grosor y fuente adaptativos
    t = max(2, int(min(W, H) / 200))                         # thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, t / 4.0)

    boxes = det["boxes"].cpu().numpy() if hasattr(det["boxes"], "cpu") else np.asarray(det["boxes"])
    scores = det["scores"]
    labels = det["labels"]

    for (x0, y0, x1, y1), s, lbl in zip(boxes, scores, labels):
        p1 = (int(x0), int(y0)); p2 = (int(x1), int(y1))
        color = _color_from_label(str(lbl))  # BGR

        # Caja
        cv2.rectangle(img, p1, p2, color, thickness=t)

        # Texto
        text = f"{lbl} {float(s):.2f}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness=max(1, t//2))
        # Fondo de la etiqueta
        cv2.rectangle(img, (p1[0], p1[1] - th - baseline - 4), (p1[0] + tw + 4, p1[1]), color, -1)
        # Texto negro encima
        cv2.putText(img, text, (p1[0] + 2, p1[1] - 2), font, font_scale, (0, 0, 0),
                    thickness=max(1, t//2), lineType=cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, img)

    # OpenCV->PIL (BGR->RGB) para mostrar con matplotlib si quieres
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(img_rgb)

    if show:
        plt.imshow(out_pil); plt.axis("off"); plt.show()

    return out_pil

if __name__ == "__main__":
    img_pil, det = predict_image("../image.png")
    print(det)  # dict con boxes/scores/labels
    _ = draw_yolo_cv2(img_pil, det, save_path="out_cv2.jpg", show=True)
