# train_model.py
from RtDetVr2_trainer import RtDetvr2Trainer
from compute_MAP import ComputeCOCOEval
import wandb
import os
from transformers.integrations import WandbCallback
from pathlib import Path
from transformers import (
    RTDetrV2Config, RTDetrV2ForObjectDetection,
    RTDetrImageProcessor, TrainingArguments, Trainer
)
from utils import extract_dataset_labels

######### WeightAndBiases Configuration ########
USE_WANDB = True
if USE_WANDB:
    WANDB_API_KEY = "c29555602f2911880862550bb2d3ec2f25b61019"  
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(project="Pt-Fue", name="rtdetrv2-scratch")



#####   ROOT AND DST #######

#root = Path.home() / "ws/dataset_prueba"  #Docker location
ROOT = "/home/rcasal/Desktop/projects/PtFue/detvr2_training/dataset_prueba" #Local location
OUTPUT_DIR = "rtdetrv2_scratch"


##########  MODEL CONFIGURATION  #########

num_labels, old2new, id2label,label2id, train_ann, val_ann, train_ann_path, val_ann_path = extract_dataset_labels(ROOT)

BACKBONE = "microsoft/resnet-50"
ENCONDER_IN_CHANNELS = [512, 1024, 2048]
FEAT_STRIDES = [8, 16, 32]
USE_PRETRAINED_BACKBONE = False

model_cfg = RTDetrV2Config(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            backbone= BACKBONE,
            use_pretrained_backbone=USE_PRETRAINED_BACKBONE,
            freeze_backbone_batch_norms=False,
            backbone_kwargs={"out_indices": (1, 2, 3)},
            encoder_in_channels=ENCONDER_IN_CHANNELS,
            feat_strides=FEAT_STRIDES,
            decoder_method="discrete",
        )

#######  TRAINING ARGUMENTS ##########

TRAIN_BATCH = 2
VAL_BATCH = 2
EPOCHS = 5
LEARNING_RATE =  5e-4
USE_FP16= True
IMAGE_SIZE = (256,256)

training_arguments = dict(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=TRAIN_BATCH,
            per_device_eval_batch_size=VAL_BATCH,
            gradient_accumulation_steps=1,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            fp16=USE_FP16,
            logging_steps=50,
            remove_unused_columns=False,
            report_to="none",
        )


#######  CLASS INITIALISATION ###########

trainer = RtDetvr2Trainer(
    root=ROOT,
    num_labels = num_labels,
    old2new = old2new,
    label2id = label2id,
    id2label = id2label,
    train_ann = train_ann,
    val_ann = val_ann,
    train_ann_path = train_ann_path,
    val_ann_path = val_ann_path,
    model_config = model_cfg,
    training_config = training_arguments,
    image_size=IMAGE_SIZE, 

)

trainer.trainer.add_callback(WandbCallback)
# mapa new(contiguo 0..K-1) -> old(id original del JSON COCO)
new2old = {v: k for k, v in trainer.old2new.items()}

coco_eval = ComputeCOCOEval(
    processor=trainer.processor,
    ann_file=str(trainer.val_ann_path),
    val_images=trainer.val_ds.images,
    new2old=new2old,               # <- CRÍTICO para COCOeval
    score_thr=0.001
)

trainer.attach_map_evaluator(coco_eval)  # <-- conectar métricas
trainer.train()
print(trainer.evaluate())
