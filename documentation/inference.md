# 🔍 Inference with OpenCV — RT-DETRv2

This document explains how the `inference.py` script works for running object detection with **RT-DETRv2** and visualizing results using **OpenCV**.

---

## Model Loading

* The model is loaded from Hugging Face Hub (or a local checkpoint):

  ```python
  MODEL_ID = "PekingU/rtdetr_v2_r50vd"
  model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
  processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
  ```
* Replace `MODEL_ID` with a local path (e.g., `rtdetrv2_scratch/checkpoint-250`) if you trained your own model.

---

##  Confidence Threshold

* Controlled by:

  ```python
  CONF_THRESHOLD = 0.5
  ```
* Predictions with a confidence score below this value are **filtered out**.
---

## Prediction Output

The function `predict_image(img_path)` returns:

1. **Original Image** (as a PIL object).
2. **Detection Dictionary** with:

   * `boxes`: List of bounding boxes in `(x_min, y_min, x_max, y_max)` format.
   * `scores`: Confidence score for each prediction.
   * `labels`: Class names for detected objects.

**Example Output (printed in console):**

```python
{
  "boxes": tensor([[100.3,  50.6, 220.1, 300.4], ...]),
  "scores": [0.95, 0.78, ...],
  "labels": ["person", "dog", ...]
}
```

---

##  Visual Result

* The detections are drawn on the image using **OpenCV**:

  * Bounding boxes are colored per class.
  * Each box includes the **label name** and **confidence score**.
  * The drawing style mimics **YOLO** visualization (thick lines + colored label background).

**Result Example:**

<p align="center">
  <img src="../out_cv2.jpg" alt="Detection Example" width="500"/>
</p>


---


