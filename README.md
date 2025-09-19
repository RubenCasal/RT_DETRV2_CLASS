# RT\_DETRV2 CLASS

This repository provides a **complete pipeline for training and evaluating RT-DETRv2** (Real-Time DEtection TRansformer v2), a state-of-the-art object detection model.
It is designed to be **easy to use**, covering everything from dataset preparation to model configuration, training, evaluation with COCO metrics, and inference visualization.

---

## 📖 Documentation

* [📂 Dataset Transformation](documentation/dataset_transformation.md)
* [⚙️ Model Configuration](documentation/model_configuration.md)
* [🎛️ Training Arguments](documentation/training_arguments.md)
* [📊 Evaluation Metrics](documentation/evaluation_metrics.md)
* [🔍 Inference](documentation/inference.md)

---

## 🚀 Quickstart

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your_user>/RT_DETRV2_CLASS.git
cd RT_DETRV2_CLASS
bash install_deps.sh
```

Now you are ready to train and run inference.

---

## 🧩 Training Script Overview (`train_model.py`)

The main training entry point is `train_model.py`.
It integrates **Weights & Biases (wandb)**, model configuration, training arguments, and COCO evaluation metrics.

### 🔹 Weights & Biases (wandb)

* Used for experiment tracking, logging metrics, and visualizing training progress.
* You can enable/disable it with the variable `USE_WANDB`.
* If enabled, the script automatically authenticates with your API key and initializes a project.

### 🔹 Paths

* `ROOT` → directory where your dataset is located (annotations + images).
* `OUTPUT_DIR` → folder where checkpoints and logs will be saved.

### 🔹 Running training

Once everything is set up, simply run:

```bash
python3 train_model.py
```

This will start training the RT-DETRv2 model with your dataset and log results to **wandb** (if enabled).

---
