# 🎛️ Training Arguments — RT-DETRv2  

This section defines the **training setup** for RT-DETRv2. These parameters directly control **performance, stability, and convergence speed**.  

---

| Parameter                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| **per_device_train_batch_size**    | Number of images processed per GPU during training (mini-batch size).        |
| **per_device_eval_batch_size**     | Batch size used during validation.                                           |
| **gradient_accumulation_steps**    | Accumulates gradients over multiple steps to simulate a larger batch size.   |
| **num_train_epochs**               | Total number of passes over the dataset.                                     |
| **learning_rate**                  | Initial learning rate for the optimizer. Controls how fast the model updates weights. |
| **weight_decay**                   | L2 regularization to reduce overfitting by penalizing large weights.          |
| **lr_scheduler_type**              | Strategy for adapting the learning rate during training (e.g., `"cosine"`).   |
| **warmup_ratio**                   | Percentage of training steps used for LR warm-up to stabilize early training. |
| **fp16**                           | Enables mixed-precision training (faster + less memory on GPUs with Tensor Cores). |
| **logging_steps**                  | Frequency (in steps) at which logs are reported.                             |
| **remove_unused_columns**          | Prevents dropping unused dataset columns, important for custom labels.        |
| **report_to**                      | Defines logging integration (`"none"`, `"wandb"`, etc.).                      |
| **output_dir**                     | Path where checkpoints and logs will be saved.                               |
| **image_size**                     | Input image resolution (affects accuracy, memory, and training time).         |

---

