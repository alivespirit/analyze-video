# Fine-Tuning a YOLOv11 Model

This guide provides a step-by-step walkthrough for fine-tuning a YOLOv11 object detection model for your specific use case. Following these instructions will help you prepare a dataset, train the model, and export it for use in the main application.

---

### Step 1: Install Dependencies

First, ensure you have the necessary Python libraries installed. The core components for training are `ultralytics` and `opencv-python`.

```bash
pip install ultralytics opencv-python
```

---

### Step 2: Test the Default YOLOv11 Model

Before diving into training, it's a good idea to test the default pre-trained `yolo11n` model. This will give you a baseline and help you determine if fine-tuning is necessary.

1.  Open the `yolo11n.py` script.
2.  Modify the script to point to one of your own videos.
3.  Run the script: `python yolo11n.py`

Observe the output. If the default model already detects objects with sufficient accuracy for your needs, you may not need to proceed with custom training.

---

### Step 3: Prepare Your Training Dataset

If the default model is not accurate enough, you'll need to create a custom dataset. We recommend using [CVAT.io](https://cvat.io), a powerful open-source annotation tool.

After annotating your images, organize your dataset into the following folder structure. This structure is required by the `ultralytics` training framework.

```
/datasets
    /images
        /train      <-- Place 80% of your images here
        /val        <-- Place 20% of your images here
    /labels
        /train      <-- Place corresponding .txt label files here
        /val        <-- Place corresponding .txt label files here
config.yaml
```

Your `config.yaml` file should define the paths to the training/validation sets and specify the class names.

**Example `config.yaml`:**
```yaml
path: ../datasets  # Path to the root datasets folder
train: images/train
val: images/val

# Class names
names:
  0: person
  1: car
```

---

### Step 4: Run the Training

Once your dataset is prepared, you can start the training process using the `yolo` command-line interface.

```bash
yolo detect train data=config.yaml model=yolo11n.pt epochs=100 imgsz=640
```

- `data`: Points to your dataset configuration file.
- `model`: Specifies the base model to start from (`yolo11n.pt`).
- `epochs`: The number of training cycles. `100` is a good starting point, but more may be needed.
- `imgsz`: The image size for training. `640` is standard.

The training process will save its results, including model weights and performance graphs, into a new directory, typically `runs/detect/train/`.

---

### Step 5: Analyze Training Results

After training is complete, navigate to the output directory (e.g., `runs/detect/train/`). Inside, you will find several result graphs, such as `results.png` and `confusion_matrix.png`.

For a deeper understanding of these graphs and to determine the optimal confidence threshold for your model, you can **upload them to an AI assistant like Google's Gemini** and ask for an explanation. This can help you interpret the model's performance and choose a confidence value that balances precision and recall for your specific use case.

---

### Step 6: Export the Model to OpenVINO Format

The main application uses the OpenVINO format for optimized inference on Intel hardware. Use the provided `export_to_openvino.py` script to convert your best-trained model.

1.  The best model weights are usually saved as `best.pt` in the training output directory.
2.  Open `export_to_openvino.py` and adjust the path to point to your `best.pt` file.
3.  Run the script: `python export_to_openvino.py`

This will create an `openvino_model` directory containing the `.xml` and `.bin` files required by the application.

---

### Step 7: Validate the Trained Model

Before deploying the model, validate its real-world tracking performance using the `batch_tracking.py` script. This script runs the tracker on a folder of videos and helps you see how it behaves.

1.  Open `batch_tracking.py`.
2.  Adjust the parameters inside the script, such as the path to your validation videos and the path to your newly exported OpenVINO model.
3.  Run the script: `python batch_tracking.py`

Review the output videos to ensure the tracking is stable and accurate.

---

### Step 8: Iterative Improvement

You may find that the model still fails in certain scenarios (e.g., flickering detections, misidentifications). The `extract_flickering_frames.py` script is designed to help you find these problematic frames.

1.  Run the script on videos where tracking performance was poor.
2.  It will extract frames where the object detection count changes erratically.
3.  Collect these difficult frames, upload them to CVAT.io, and add them to your dataset.
4.  Re-run the training (Step 4) to further improve the model's robustness.

This iterative process of identifying failures, adding them to the dataset, and retraining is key to achieving a highly accurate and reliable object detection model.
