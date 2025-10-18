# Melodious: Optical Music Recognition with YOLO

## Overview
This project implements a custom YOLO-style object detector for Optical Music Recognition (OMR) on the DeepScores v2 Dense dataset. The model is trained entirely from scratch (no pretrained weights) to detect and classify musical notation symbols such as noteheads, clefs, rests, accidentals, beams, and stems.

## Main Components
- **notebooks/model_evaluation.ipynb**: The main notebook for model evaluation, visualization, and reporting. This is the primary file to run and review.
- **melodious/**: Contains all core Python code for the model, dataset loading, training, and inference.
- **outputs/**: Stores training history, model checkpoints, and generated visualizations.
- **dataset_ds2_dense/**: Contains the DeepScores v2 Dense dataset (images and annotations).

## How to Use This Project

### 1. Environment Setup
- Make sure you have Python 3.8+ and PyTorch installed.
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

### 2. Main Workflow
- **The main workflow is in the Jupyter notebook:**
  - Open `notebooks/model_evaluation.ipynb` in Jupyter or VS Code.
  - Run the notebook cells in order. This will:
    - Load the trained YOLO model
    - Analyze training and validation loss curves
    - Evaluate detection performance on the test set
    - Visualize detection results and confidence statistics
    - Summarize key findings

- **You do NOT need to run `main.py` for evaluation.**
  - `main.py` is for custom scripts or additional experiments, but all core results and analysis are in the notebook.

### 3. Training (Optional)
- If you want to retrain the model from scratch:
  - Use the scripts in `melodious/train.py` 
  - Make sure the dataset is available in `dataset_ds2_dense/` (I did not upload it since it's large, you can download it from https://zenodo.org/records/4012193/files/ds2_dense.tar.gz?download=1).

### 4. Outputs
- Visualizations and results are saved in `outputs/visualizations/`.
- Model checkpoints are in `outputs/` (e.g., `yolo_epoch_8.pth`).

## What the Code Trains
- A YOLO-style detector for 15 classes of musical symbols
- Trained on dense, real-world sheet music images
- Handles tiny, overlapping, and visually similar symbols
