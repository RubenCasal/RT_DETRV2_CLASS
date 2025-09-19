# YOLO → COCO Conversion Script

This script converts datasets annotated in **YOLO format** into the **COCO format**, which is required for training with **RT-DETRv2**. Since many datasets are originally available in YOLO format, this utility was created to streamline the preparation process.

---

## Why Convert to COCO?

RT-DETRv2 expects datasets in **COCO format** (JSON annotations with image and bounding box metadata). Most open-source datasets are distributed in YOLO format, which uses simple text files for bounding boxes. This script bridges that gap by automatically converting YOLO datasets into the required COCO structure.

---

## 📂 YOLO Dataset Structure

A typical YOLO dataset contains:

```
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

* **images/**: Contains the training and validation images.
* **labels/**: Each image has a corresponding `.txt` file with annotations.
* Each line in a YOLO label file has the format:

  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```

  where all coordinates are normalized to `[0,1]`.

---

## 📂 COCO Dataset Structure

A COCO dataset includes:

```
└── dataset_coco/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── annotations/
        ├── train.json
        └── val.json
```

* **annotations/**: Contains COCO-style `.json` files with metadata.
* A COCO JSON includes:

  * `images`: list of image entries with `id`, `file_name`, `width`, `height`.
  * `annotations`: bounding boxes in **\[x, y, width, height]** absolute pixel format.
  * `categories`: list of categories with `id` and `name`.

---

## How to Use

1. **Set the paths in the script:**

   * `SRC`: Path to your YOLO dataset (must contain `images/` and `labels/`).
   * `DEST`: Path where the converted COCO dataset will be stored.
   * `TRAIN_RATIO`: Ratio of training/validation split (default = `0.85`).

   Example:

   ```python
   SRC  = Path("/media/user/datasets/my_yolo_dataset").resolve()
   DEST = Path("/media/user/datasets/my_dataset_coco").resolve()
   TRAIN_RATIO = 0.85
   ```

2. **Run the conversion:**

   ```bash
   python transform_coco_format.py
   ```

3. **Resulting structure:**

   * Training/validation images and labels copied to `DEST/images/` and `DEST/labels/`.
   * COCO-style annotation files generated in `DEST/annotations/train.json` and `DEST/annotations/val.json`.

---


