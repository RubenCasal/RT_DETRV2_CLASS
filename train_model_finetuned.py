# ==============================
# train_model_pretrained.py
# ==============================

from pathlib import Path
from math import ceil
import os
import json

from transformers.integrations import WandbCallback
from transformers import TrainingArguments

from trainer_finetuned import RtDetvr2Trainer
from compute_MAP import ComputeCOCOEval
from save_imgs_val import SaveValImgsInCkpt
import wandb
from utils import extract_dataset_labels

# -----------------------
# Configuración general
# -----------------------
USE_WANDB = True
WANDB_PROJECT = "Pt-Fue"
WANDB_RUN_NAME = "rtdetrv2-coco-pretrained"
CKPT = "PekingU/rtdetr_v2_r50vd"  # <-- Checkpoint COCO-pretrained (modelo completo)

# Dataset
# ROOT = "/home/rcasal/Desktop/projects/dataset fuerteventura/datasets_internet/pt_fue_v4_full(COCO FORMAT)"
ROOT = "/home/rcasal/Desktop/projects/PtFue/detvr2_training/dataset_prueba2"
OUTPUT_DIR = "rtdetrv2_pretrained_out"

# Hiperparámetros (ajustados para finetune)
TRAIN_BATCH = 2
VAL_BATCH = 2
EPOCHS = 40
LEARNING_RATE = 5e-5        # <- más bajo para finetune desde COCO
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.15         # <- un poco más largo para estabilizar
USE_FP16 = False
IMAGE_SIZE = (640, 640)     # puedes subir a (512,512) o (640,640) si hay VRAM
EVAL_RATE_EPOCHS = 1      # eval/save cada media época

# -----------------------
# W&B (opcional)
# -----------------------
if USE_WANDB:
    WANDB_API_KEY = "78d292744788af62441ee14891bd488ef500e3b6"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(project="Pt-Fue", name="rtdetrv2-scratch")

# -----------------------
# Etiquetas del dataset
# -----------------------
num_labels, old2new, id2label, label2id, train_ann, val_ann, train_ann_path, val_ann_path = extract_dataset_labels(ROOT)

with open(Path(train_ann_path), "r") as f:
    num_train_samples = len(json.load(f)["images"])

steps_per_epoch = ceil(num_train_samples / TRAIN_BATCH)
eval_save_steps = max(1, int(steps_per_epoch * EVAL_RATE_EPOCHS))  # asegurar int

training_arguments = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=VAL_BATCH,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    fp16=USE_FP16,
    remove_unused_columns=False,
    report_to=("wandb" if USE_WANDB and os.environ.get("WANDB_API_KEY") else "none"),
    eval_steps=eval_save_steps,
    save_steps=eval_save_steps,
    # Rendimiento DataLoader
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
)

# -----------------------
# Augmentations (ligeras para dominio dron)
# -----------------------
import albumentations as A
AUGMENTATIONS = [
    # Affine un poco más agresivo
    A.Affine(
        scale=(0.90, 1.10),
        translate_percent=(0.0, 0.05),
        rotate=(-10, 10),
        shear=(-4, 4),
        p=0.5
    ),

    # Ruido gaussiano (igual que antes)
    A.AdditiveNoise(
        noise_type="gaussian",
        spatial_mode="shared",
        noise_params={"mean_range":[0,0], "std_range":[0.01,0.03]},
        approximation=1
    ),

    # Brillo/contraste leve (igual)
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.1,
        p=0.3
    ),

    # Motion blur ligero (igual)
    A.MotionBlur(
        blur_limit=2,
        p=0.01
    ),

    # Sombra con menos probabilidad e intensidad más contenida
    A.RandomShadow(
        shadow_roi=[0, 0.5, 1, 1],
        num_shadows_limit=[2, 3],
        shadow_dimension=4,
        shadow_intensity_range=[0.15, 0.5],
        p=0.3
    ),

    # Sun flare menos agresivo y menos frecuente
    A.RandomSunFlare(
        flare_roi=[0, 0, 1, 0.5],
        src_radius=250,
        src_color=[255, 255, 255],
        angle_range=[0, 1],
        num_flare_circles_range=[3, 6],
        method="overlay",
        p=0.1
    ),

    # Neblina ligera (nuevo)
    A.RandomFog(
        fog_coef_lower=0.05,
        fog_coef_upper=0.2,
        alpha_coef=0.05,
        p=0.01
    ),

    # Lluvia muy ocasional (nuevo)
    A.RandomRain(
        slant_lower=-5,
        slant_upper=5,
        drop_length=10,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.95,
        p=0.05
    ),
]


# -----------------------
# Crear trainer con checkpoint COCO-pretrained
# -----------------------
trainer = RtDetvr2Trainer(
    root=ROOT,
    num_labels=num_labels,
    old2new=old2new,
    label2id=label2id,
    id2label=id2label,
    train_ann=train_ann,
    val_ann=val_ann,
    train_ann_path=train_ann_path,
    val_ann_path=val_ann_path,
    model_config=None,                   # no se usa si from_checkpoint no es None
    training_config=training_arguments,
    augmentations=AUGMENTATIONS,
    image_size=IMAGE_SIZE,
    from_checkpoint=CKPT,               # <-- carga modelo completo COCO-pretrained
)

# Callbacks
trainer.trainer.add_callback(WandbCallback)

# mAP COCO-like
new2old = {v: k for k, v in trainer.old2new.items()}
coco_eval = ComputeCOCOEval(
    processor=trainer.processor,
    ann_file=str(trainer.val_ann_path),
    val_images=trainer.val_ds.images,
    new2old=new2old,
    score_thr=0.001,
)
trainer.attach_map_evaluator(coco_eval)

# Guardado de imágenes de validación
trainer.trainer.add_callback(
    SaveValImgsInCkpt(
        trainer.processor,
        trainer.val_ds,
        trainer.id2label,
        infer_size=IMAGE_SIZE,
        k=4,
        thr=0.1,
        log_to_wandb=True,
    )
)

# Entrenamiento + evaluación
trainer.train()
print(trainer.evaluate())


