# 📊 Training Metrics — COCO Evaluation

This repository integrates **COCO-style evaluation** using `pycocotools`. The evaluation is performed via the `ComputeCOCOEval` class, which processes predictions and compares them against ground-truth annotations in COCO format.

---

## ⚙️ Metrics Returned

The `COCOeval` object computes a series of standard object detection metrics. From these, we expose the most relevant ones:

| Metric         | Description                                                                                                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **coco/AP**    | Mean Average Precision (mAP) across **IoU thresholds from 0.5 to 0.95** in steps of 0.05. This is the main COCO evaluation metric, offering a balanced view of model performance. |
| **coco/AP50**  | Average Precision at **IoU = 0.50** (similar to Pascal VOC metric). Easier and usually higher than the overall mAP.                                                               |
| **coco/AP75**  | Average Precision at a stricter threshold (**IoU = 0.75**). Highlights the model’s ability to produce accurate, well-localized bounding boxes.                                    |
| **coco/AR100** | Average Recall considering up to **100 detections per image**. Measures how many ground-truth objects are successfully detected, independent of classification confidence.        |

---



