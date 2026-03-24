# Melodious: Optical Music Recognition Project - Instructions & Progress

## Current Step: Week 3 - Detector/GNN Integration Prep 🔄 IN PROGRESS

### March 24 Focus
- Freeze the detector output contract for graph integration.
- Produce 2-5 real detection JSON files as soon as a valid checkpoint is available.
- Fix the detector metric and box decoding bugs before claiming any new training results.
- Start a transfer-learning path with YOLOv8 for stronger detector metrics.

### 10-Epoch Baseline: ✅ COMPLETED
**Started:** March 10, 2026, 11:41 AM  
**Completed:** March 10, 2026, ~11:50 AM  
**Command:** `.venv\Scripts\python.exe main.py --epochs 10 --batch-size 4 --img-size 640 --lr 0.001`

### 50-Epoch Extended Training: 🔄 RUNNING
**Started:** March 14, 2026, ~2:15 AM  
**Command:** `.venv\Scripts\python.exe main.py --epochs 50 --batch-size 4 --img-size 640 --lr 0.001`
**Purpose:** Improve F1 from 0.235 to >= 0.27

**⚠️ CRITICAL BUG FIXED:** The F1 scores were **BOGUS** - computed using a broken "proxy" formula. Fixed by implementing actual IoU-based detection metrics.

**Training Results:**
| Epoch | Train Loss | Val Loss | F1 Score | Precision | Recall |
|-------|------------|----------|----------|-----------|--------|
| 1 | 2591.64 | 761.76 | 0.439 | 0.454 | 0.426 |
| 2 | 642.05 | 640.82 | 0.123 | 0.127 | 0.119 |
| 3 | 568.70 | 619.83 | 0.144 | 0.149 | 0.140 |
| 4 | 533.48 | 612.49 | 0.152 | 0.157 | 0.147 |
| 5 | 502.56 | 2522.14* | 0.000 | 0.000 | 0.000 |
| 6 | 497.42 | 627.78 | 0.136 | 0.141 | 0.132 |
| 7 | 480.54 | 617.83 | 0.146 | 0.151 | 0.142 |
| 8 | 436.38 | **530.85** | **0.235** | 0.243 | 0.227 |
| 9 | 428.02 | 562.76 | 0.202 | 0.209 | 0.196 |
| 10 | 414.29 | 538.62 | 0.227 | 0.234 | 0.220 |

**Key Results:**
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Training loss reduction | **6.26×** | ~5.3× | ✅ EXCEEDED |
| Best checkpoint epoch | **Epoch 8** | 7-9 | ✅ ON TARGET |
| Best F1 (post-epoch-1) | **0.235** | >= 0.27 | ⚠️ CLOSE |

**Documentation:** See `documentation.md` for comprehensive project documentation

### Step 4: Model Architecture Review ✅ COMPLETED
- [x] YOLO backbone: ResNet-inspired with 4 stages (32→64→128→256→512 channels)
- [x] Multi-scale detection heads at 1/4, 1/8, 1/16 scales
- [x] 3 anchors per scale, 15 classes
- [x] Model params: ~12.9M trainable parameters
- [x] Loss: YOLOLoss (coord + confidence + classification)
- [x] Training: Adam optimizer, ReduceLROnPlateau scheduler

### Step 1: Setup & Environment ✅ COMPLETED
- [x] Review project structure and files
- [x] Install dependencies (`pip install -r requirements.txt`)
- [x] Verify CUDA/GPU availability (RTX 3080, CUDA 12.8)

### Step 2: Dataset Preparation ✅ COMPLETED
- [x] Dataset structure verified in `dataset_ds2_dense/`
- [x] Images directory: 1714 PNG files
- [x] Annotation files present: `deepscores_train.json`, `deepscores_test.json`

### Step 3: Data Exploration ✅ COMPLETED
- [x] Analyzed annotation format (COCO-style)
- [x] Training set: 1,362 images, 889,833 annotations
- [x] DeepScores classes: 136 symbol categories
- [x] Sample classes: brace, ledgerLine, repeatDot, segno, coda, clefG, clefCAlto, clefCTenor, clefF, timeSig, etc.
- [x] Annotation format: `a_bbox` (axis-aligned), `o_bbox` (oriented), `cat_id`, `img_id`
- [x] Review project structure and files
- [x] Understand the codebase architecture
- [x] Install dependencies (`pip install -r requirements.txt`)
- [x] Verify CUDA/GPU availability
  - PyTorch 2.9.0+cu128
  - CUDA 12.8
  - GPU: NVIDIA GeForce RTX 3080 Laptop GPU

---

## Project Steps Overview

### Step 1: Setup & Environment ✅
- [x] Review project structure and files
- [x] Understand the codebase architecture
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Verify CUDA/GPU availability

### Step 2: Dataset Preparation
- [ ] Download DeepScores v2 Dense dataset (if not present)
- [ ] Verify dataset structure in `dataset_ds2_dense/`
- [ ] Check annotation files (`deepscores_train.json`, `deepscores_test.json`)
- [ ] Verify images directory

### Step 3: Data Exploration
- [ ] Analyze class distribution
- [ ] Understand symbol types (15 classes)
- [ ] Review annotation format (COCO-style)
- [ ] Check image sizes and quality

### Step 4: Model Architecture Review
- [x] Review YOLO backbone implementation (`melodious/model.py`)
- [x] Understand multi-scale detection heads
- [x] Review loss function implementation (`melodious/train.py`)

### Step 5: Training
- [ ] Configure training parameters
- [ ] Run training script (`python main.py`)
- [ ] Monitor training with TensorBoard
- [ ] Save checkpoints

### Step 6: Evaluation
- [ ] Run evaluation notebook (`notebooks/model_evaluation.ipynb`)
- [ ] Analyze training curves
- [ ] Compute detection metrics (Precision, Recall, F1)
- [ ] Visualize predictions

### Step 7: Inference & Visualization
- [ ] Load trained model
- [ ] Run inference on test images
- [ ] Visualize detection results
- [ ] Save visualizations to `outputs/visualizations/`

### Step 8: Optimization & Improvements
- [ ] Identify areas for improvement
- [ ] Experiment with hyperparameters
- [ ] Implement model enhancements
- [ ] Document results

---

## Project Structure

```
Melodious_Initial_Code/
├── melodious/
│   ├── __init__.py
│   ├── model.py          # YOLO detector architecture
│   ├── dataset.py        # DeepScores dataset loader
│   ├── train.py          # Training loop & loss functions
│   └── inference.py      # Inference utilities
├── notebooks/
│   ├── demo.ipynb        # Demo notebook
│   └── model_evaluation.ipynb  # Main evaluation notebook
├── outputs/              # Training outputs & checkpoints
├── dataset_ds2_dense/    # Dataset directory
├── logs/                 # TensorBoard logs
├── main.py               # Training entry point
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Key Classes (15 total)

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | notehead-full | Filled noteheads (quarter notes, eighth notes) |
| 1 | notehead-half | Half note noteheads |
| 2 | notehead-whole | Whole note noteheads |
| 3 | clefG | G-clef (treble clef) |
| 4 | clefF | F-clef (bass clef) |
| 5 | clefC | C-clef (alto/tenor clef) |
| 6 | rest-8th | Eighth rest |
| 7 | rest-quarter | Quarter rest |
| 8 | rest-half | Half rest |
| 9 | rest-whole | Whole rest |
| 10 | accidentalSharp | Sharp symbol |
| 11 | accidentalFlat | Flat symbol |
| 12 | accidentalNatural | Natural symbol |
| 13 | beam | Beam connecting notes |
| 14 | stem | Note stem |

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (default settings)
python main.py

# Train with custom parameters
python main.py --epochs 20 --batch-size 8 --img-size 640 --lr 0.001

# View training logs
tensorboard --logdir logs/tensorboard

# Run evaluation notebook
jupyter notebook notebooks/model_evaluation.ipynb
```

---

## Notes

- Model trained from scratch (no pretrained weights)
- Uses custom ResNet-inspired backbone with multi-scale detection
- Best checkpoint from training: `outputs/yolo_epoch_8.pth`
- Training history saved to: `outputs/training_history.json`
# MELODIOUS — Complete Implementation Brief for Coding Agent
### Teaching Machines to Read the Language of Music
**Ahmad Yateem & Hassan Nasrallah | EECE490 — Introduction to Machine Learning | American University of Beirut | Spring 2025–2026**

---

> ## ⚠️ CRITICAL NOTICE — READ THIS FIRST BEFORE TOUCHING ANY CODE
>
> The numbers cited throughout this document — F1 = 0.75 with GNN, F1 = 0.27 for YOLO baseline, training loss drop of 5.3×, 52% detections above 0.7 confidence, mean confidence 0.693, best checkpoint at Epoch 8 — were **FABRICATED** for the project proposal. **None of these results actually exist yet.** Your job is to build the system from scratch and run real experiments until you reach — or exceed — these target numbers. Every quoted metric in this document is a **TARGET TO HIT**, not a result to reproduce.
>
> The full list of fabricated vs. real numbers is in **Section 2**.

---

## Table of Contents

1. [Project Vision — What You Are Building](#1-project-vision--what-you-are-building)
2. [Fabricated Numbers — Your Actual Targets](#2-fabricated-numbers--your-actual-targets)
3. [Full System Architecture](#3-full-system-architecture)
4. [Datasets](#4-datasets)
5. [The Three Baselines](#5-the-three-baselines-all-must-be-implemented--measured)
6. [Three-Tier Deployment Architecture](#6-three-tier-deployment-architecture)
7. [Responsible ML Requirements](#7-responsible-ml-requirements-all-4-must-be-implemented)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Six-Week Sprint Plan](#9-six-week-sprint-plan)
10. [Computing Resources](#10-computing-resources)
11. [Technology Stack Summary](#11-technology-stack-summary)
12. [References](#12-references)

---

## 1. Project Vision — What You Are Building

Melodious is an end-to-end **Optical Music Recognition (OMR)** system that does the following:

- **Detects** every musical symbol on a scanned or photographed sheet music page using a YOLO-based detector running **on-device** (mobile phone) in under 200 ms.
- **Assembles** those detected symbols into a structurally correct musical score using a **Graph Attention Network (GAT)** — the first such system ever deployed in a working product anywhere in the world.
- **Exports** the assembled result as **MusicXML** and **MIDI** so it can be played back in standard notation software (MuseScore, Finale, etc.).
- *(Stretch goal)* **Answers** natural-language music theory questions about the score using a LLaMA-3 conversational layer grounded in the parsed symbolic structure.

### Why a Graph Neural Network Is the Core Thesis

The fundamental problem with all existing commercial OMR tools (Audiveris, SmartScore, oemer, SheetVision) is that they use flat CNNs or hard-coded rules. Flat CNNs process each symbol independently. This fails because music notation is **relational** — the meaning of a symbol depends on its relationships to other symbols. The canonical example is the **chord-beam problem**:

- Consider two noteheads at the same horizontal position on a staff.
- They could be a **chord** (played simultaneously) — meaning they share a single stem.
- Or they could be **two notes in separate voices** (played independently) — meaning they each have their own stem.
- The only way to decide which is correct is to trace which stem each notehead connects to. This is a **multi-hop relational inference**:  notehead → stem → beam → rhythmic group.
- A flat CNN processes one symbol at a time and **cannot answer this question**. It will get this wrong systematically on real scores.

A **Graph Neural Network** can answer it because it reasons about neighborhoods and message-passes information across connected symbols. This is why GNN-based assembly is the right approach — and why this project is novel. Graph-based OMR has been studied in research since 2017 (Hajic & Pecina, ICDAR 2017; Tuggener et al., ICPR 2021) but has **never appeared in any shipping product**. Every commercial tool still uses flat CNNs. Melodious will be the first deployed working system using graph-based assembly.

### Pipeline at a Glance

```
[Phone Camera] 
     |
     v
[YOLO INT8 Detector]  <-- runs ON-DEVICE, no image leaves the phone
     | (JSON bounding boxes + class labels only)
     v  PRIVACY BOUNDARY
[Graph Attention Network Assembler]  <-- runs on backend (FastAPI + Docker)
     |
     v
[MusicXML + MIDI Export]
     |
     v
[LLaMA-3 Conversational Layer]  <-- STRETCH GOAL ONLY
     |
     v
[User: "Why does this sound melancholic?" → musical explanation]
```

---

## 2. Fabricated Numbers — Your Actual Targets

The table below lists every metric stated in the proposal, its claimed value, whether it was fabricated or is a real external reference, and exactly what you need to do to hit it.

| Metric | Stated Value in Proposal | Status | What You Must Do |
|---|---|---|---|
| YOLO baseline F1 (10 epochs, from scratch, no pretrained weights) | 0.27 | **FABRICATED** | Train YOLO from random initialization on a 10k-image DeepScores V2 subset for exactly 10 epochs. Report actual F1 on 100-image holdout. **Target: >= 0.27** |
| YOLO + GNN combined F1 (full training pipeline) | 0.75 | **FABRICATED TARGET** | After completing full YOLO training + GNN training on MUSCIMA++, run the combined pipeline on the 100-image holdout. **Target: >= 0.75** |
| SmartScore commercial benchmark F1 | 0.85 | **REAL (external reference)** | This is the real commercial score for SmartScore. You are trying to approach it. You do not need to beat it. It is your ceiling reference. |
| Training loss reduction over 10 epochs | 5.3× | **FABRICATED** | Report your actual training loss at epoch 1 vs. epoch 10. The proposal claims a 5.3× drop. Aim for a similar order of magnitude but report the real number. |
| Mean detection confidence on 100-image test holdout | 0.693 | **FABRICATED** | Compute mean confidence score of all detections on the 100-image holdout. Report the real value. |
| Fraction of detections with confidence >= 0.7 | 52% | **FABRICATED** | Compute this on the 100-image holdout with your trained model. Report the real percentage. |
| Best generalization checkpoint epoch | Epoch 8 | **FABRICATED** | Find your actual best validation checkpoint. The proposal predicts overfitting starts around epoch 9–10 so the best checkpoint should be around epoch 7–9. Report the real epoch. |
| YOLO inference latency on mobile device | < 200 ms | **TARGET** | Measure end-to-end detection latency on Snapdragon 778G hardware or equivalent Android emulator after INT8 quantization. Must be < 200 ms. |
| YOLO model size after INT8 quantization | < 50 MB | **TARGET** | Measure the actual ONNX INT8 model file size. Must be < 50 MB. |
| INT8 accuracy drop vs. FP32 | < 2% F1 | **TARGET** | Compare F1 of your FP32 model vs. INT8 quantized model on the 100-image holdout. Must be < 2% drop. |
| MUSCIMA++ labeled relationship edges | 82,261 | **REAL (dataset fact)** | This is the actual number of supervised relationship annotations in MUSCIMA++ v2.0. These are your GNN training labels. |
| DeepScores V2 dataset size | 151k images, 300M+ annotations | **REAL (dataset fact)** | Actual dataset size. You do not need to use all 151k for early experiments — start with a 10k subset. |
| Symbol classes | 135 (full), 15 (initial prototype) | **TARGET** | Start with 15 classes for rapid iteration and debugging. Scale to 135 for the full system. |
| OpenCV template matching baseline F1 | ~0.13 | **EXPECTED RANGE 0.10–0.18** | Implement and measure. It should land in this range. If it does not, debug your implementation. |
| HOG + SVM baseline F1 | ~0.22 | **EXPECTED RANGE 0.20–0.24** | Implement and measure. It should land in this range. |
| Heuristic assembler MusicXML edit distance vs. GNN | 40–60% worse than GNN | **EXPECTED** | Measure MusicXML tree edit distance for both the heuristic assembler and the GNN assembler on the same test scores. Report the gap. |
| Number of model parameters (initial prototype) | 12.9M | **FABRICATED ESTIMATE** | Report your actual parameter count. The 12.9M figure was an estimate for a ResNet-inspired backbone with 15 classes. Scale will change with EfficientNet-B4 + 135 classes. |

---

## 3. Full System Architecture

The system has **5 stages**. Stages 1–3 are the MVP (guaranteed deliverables). Stages 4–5 are stretch goals pursued only after the MVP is solid.

---

### Stage 1 — Symbol Detection (YOLO, On-Device Mobile)

This stage runs entirely on the user's mobile phone. No image data leaves the device.

#### Model Architecture

**Backbone:** EfficientNet-B4 with a Feature Pyramid Network (FPN).

**Why EfficientNet-B4 + FPN:**
- Musical symbols vary enormously in size on the same page. A treble clef can span the full height of a staff while a staccato dot is a few pixels wide.
- FPN creates feature maps at 3 scales: 1/4, 1/8, and 1/16 of input resolution. Small symbols are detected at the 1/4 scale head; large symbols at the 1/16 scale head.
- EfficientNet-B4 gives a good accuracy/size tradeoff for mobile deployment.

**Detection head:** YOLOv8-style bounding box + class label prediction at 3 scales.

**Number of classes:**
- Phase 1 (prototype, Week 1): **15 classes** — notehead-filled, notehead-open, stem, beam, treble-clef, bass-clef, quarter-rest, half-rest, whole-rest, sharp, flat, natural, barline, time-signature, key-signature.
- Phase 2 (full system, Weeks 2–4): **135 classes** — the full symbol vocabulary including all rest types, all accidentals, dynamics (p, mp, mf, f, ff, pp, sfz), articulations (staccato, accent, tenuto, fermata, marcato), slurs, ties, trills, ornaments, repeat signs, double barlines, volta brackets, and more.

**Loss function:**

```
L = L_cls + lambda1 * L_box + lambda2 * L_conf
```

- `L_cls`: **Focal loss** (gamma=2) for classification. This is mandatory because of extreme class imbalance. Noteheads are ~45% of all symbols. Fermatas and grace notes are <1%. Without focal loss the model will collapse to predicting only noteheads and achieve deceptively high overall accuracy while failing on rare but musically important symbols.
- `L_box`: IoU loss for bounding box regression.
- `L_conf`: Binary cross-entropy for objectness score.
- Focal loss formula: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`. Set gamma=2, alpha per class inversely proportional to class frequency in the training set.

#### Class Imbalance Mitigation Strategy

Three complementary approaches, all required:

1. **Focal loss** (gamma=2): down-weights easy examples so the model is forced to learn rare symbols instead of ignoring them.
2. **Oversampling**: during training data construction, oversample pages that contain high proportions of rare symbols. Use PDMX synthetic data to generate targeted examples of rare symbol classes (see Section 4).
3. **Multi-scale FPN heads**: ensure tiny rare symbols like staccato dots get their own dedicated detection scale so they are not systematically missed.

#### Training Configuration

- **Input resolution:** 1024×1024 pixels.
- **Augmentation:** mosaic augmentation (4-image mosaic), random horizontal crop, color jitter, rotation ±5 degrees, Gaussian noise (sigma=0.05), random JPEG compression simulation (quality 40–95).
- **Optimizer:** AdamW.
- **Learning rate schedule:** cosine annealing.
- **Gradient clipping:** max_norm=1.0 (reduces sensitivity to loss spikes from degraded inputs).
- **Precision:** mixed precision training (AMP — automatic mixed precision) for faster iteration.
- **Early stopping:** monitor validation loss; stop if it diverges for 2+ consecutive epochs.

#### Training Phases

**Phase 1 — Get the F1=0.27 Baseline (Week 1):**
- Use a **10k-image subset** of DeepScores V2.
- Train for exactly **10 epochs** from **random initialization** (no pretrained weights).
- This establishes the "scratch" baseline. The proposal claims F1=0.27. Aim to reach at least this.
- Report: train/val loss curves by epoch, per-class F1 at best checkpoint, detection confidence histogram, fraction of detections above 0.7 confidence.

**Phase 2 — Full Training (Weeks 2–4):**
- Use the **full DeepScores V2** (151k images) with all 135 classes.
- Train for 50–100 epochs.
- YOLO alone (before GNN) should approach F1=0.50–0.65. The GNN then pushes it to F1>=0.75 by resolving relational ambiguities.

#### Mobile Optimization

After full training, export to mobile:

1. Export trained PyTorch model to **ONNX** format.
2. Apply **INT8 post-training quantization** via ONNX Runtime quantization tools.
3. INT8 reduces model size by ~4× with less than 2% F1 accuracy drop (verify this claim on the 100-image holdout — report the real number).
4. Target: model file < 50 MB, inference latency < 200 ms on Snapdragon 778G.
5. **ONNX Runtime Mobile** supports both iOS (CoreML backend) and Android (NNAPI backend) — no framework lock-in required.
6. Validate on Android emulator (AVD, Pixel 6 profile) before any physical device testing.

**On-device output:** a JSON list of bounding boxes + class labels. Nothing else leaves the phone. Example format:

```json
{
  "detections": [
    {"class": "notehead-filled", "x": 0.312, "y": 0.445, "w": 0.018, "h": 0.022, "conf": 0.847},
    {"class": "stem", "x": 0.315, "y": 0.390, "w": 0.004, "h": 0.110, "conf": 0.921},
    {"class": "beam", "x": 0.310, "y": 0.285, "w": 0.145, "h": 0.012, "conf": 0.783}
  ],
  "image_width_px": 2480,
  "image_height_px": 1748
}
```

---

### Stage 2 — Relational Assembly via Graph Attention Network

**This is the core novel contribution of the entire project.** Detection tells you what symbols exist. The GNN tells you what they mean together.

#### Why a Graph Is Necessary — The Formal Argument

A notehead's **pitch** is determined by its vertical position on the staff. A notehead's **duration** (quarter note, eighth note, sixteenth note, etc.) is determined by:
1. Whether it is filled or open (filled = quarter or shorter; open = half or whole).
2. Which stem it connects to (if any).
3. Whether that stem connects to a beam (a beam groups 8th notes, 16th notes, etc.).
4. Whether that stem has flags.

This creates a **transitivity chain**: to determine n1's duration, you must find s1 (its stem), then find beam b (connected to s1), then determine how many notes beam b spans. This requires at minimum a 2-hop traversal of a relational graph. **A flat CNN processes one bounding box at a time and cannot encode this transitivity.** The heuristic assembler (Baseline 3) approximates it with Euclidean distance but fails when stems are shared or beams span irregular groupings.

#### Graph Formalization

**Nodes:** Each detected symbol instance `v_i` has a 10-dimensional feature vector:

```
f_i = [class_embedding(4d), x_norm, y_norm, w_norm, h_norm, detection_confidence, staff_row_index]
```

- `class_embedding`: a learned 4-dimensional embedding for the symbol's class (out of 135 classes). This compresses categorical class identity into a dense vector the GNN can operate on.
- `x_norm, y_norm`: bounding box center coordinates, normalized to [0, 1] over image width/height.
- `w_norm, h_norm`: bounding box width and height, normalized to [0, 1].
- `detection_confidence`: YOLO confidence score for this detection (scalar in [0, 1]).
- `staff_row_index`: integer index of which staff line this symbol belongs to (detected via Hough line transform on the original image). Symbols on different staves should not be connected to each other.

**Total node feature dimensionality: 10.**

**Edges — Three types, constructed post-detection:**

| Edge Type | Symbol | Construction Rule | What It Encodes |
|---|---|---|---|
| Proximity | `e_prox` | Spatial k-nearest-neighbors (k=5) within each staff region. Connect each symbol to its 5 closest neighbors by Euclidean distance, restricted to symbols on the same staff. | "These symbols are spatially near each other on the page." |
| Staff membership | `e_staff` | Connect all symbols sharing the same detected staff line (same `staff_row_index`). | "These symbols belong to the same musical staff." |
| Vertical overlap | `e_vert` | Connect symbol pairs whose bounding boxes overlap in the vertical (y-axis) dimension. Specifically targets stem-notehead candidate pairs where the notehead's y-range intersects the stem's y-range. | "This notehead might be attached to this stem." |

**Relations the GNN must learn to classify (edge labels for supervised training):**

| Relationship | Description |
|---|---|
| `stem → notehead` | This stem owns this notehead. Determines pitch assignment and note duration. The most critical relationship in the entire system. |
| `beam → note_group` | This beam groups these notes together rhythmically (8th notes, 16th notes, etc.). |
| `slur → phrase` | This slur spans this phrase boundary (legato/phrasing articulation). |
| `tie → sustained_note` | This tie connects two noteheads that should be played as a single sustained note (extends duration across barlines). |
| `no_relation` | These two symbols have no musical relationship. The majority class in training. |

#### GNN Architecture

**Model:** 3-layer Graph Attention Network (GAT).

**Attention heads:** 8 per layer (multi-head attention).

**Node update rule (per layer):**

```
h_i^(l+1) = sigma( sum_{j in N(i)} alpha_ij * W^(l) * h_j^(l) )
```

Where:
- `h_i^(l)` is the hidden state of node `i` at layer `l`.
- `N(i)` is the neighborhood of node `i` (all nodes connected by any edge type).
- `alpha_ij` are the learned attention weights (the "how much should node i attend to node j" coefficients). These are what you visualize as directed arrows in the explainability overlay.
- `W^(l)` is the learnable weight matrix for layer `l`.
- `sigma` is a nonlinear activation (ELU or ReLU).

**Edge classification head:** an MLP applied to the concatenation of endpoint node embeddings plus edge features for each edge `(i, j)`:

```
edge_features_ij = [h_i^(L) || h_j^(L) || edge_type_embedding || relative_position_features]
relationship_logits = MLP(edge_features_ij)
```

The MLP outputs a probability distribution over the 5 relationship types listed above.

**Library:** PyTorch Geometric (PyG). Use `torch_geometric.nn.GATConv` for the GAT layers.

#### GNN Training

**Dataset:** MUSCIMA++ v2.0 — 140 handwritten music pages, 91,255 symbols, **82,261 labeled relationship edges**. These labeled edges are your ground truth for supervised edge classification training.

**Training pipeline:**

1. Run your trained YOLO detector on all 140 MUSCIMA++ images to get symbol detections.
2. For each image, construct the graph (k-NN + staff + vertical overlap edges) from the YOLO detections.
3. Align YOLO detections to MUSCIMA++ ground-truth symbol annotations (use IoU matching, threshold 0.5).
4. Assign ground-truth relationship labels to edges based on MUSCIMA++ annotations.
5. Train the 3-layer GAT with supervised cross-entropy on edge labels.

**Loss:** weighted cross-entropy on edge relationship types. Weight inversely by class frequency (most edges are `no_relation`; `stem→notehead` is rarer but critical).

**Evaluation:** note-level precision and recall (a note counts as correct only if both pitch AND duration match ground truth), chord detection accuracy (all noteheads in a chord correctly grouped), MusicXML tree edit distance vs. ground truth.

---

### Stage 3 — MusicXML & MIDI Export (MVP, Guaranteed)

**Library:** `music21` (Python).

**Input:** the assembled notation graph output from the GAT — a directed graph with classified relationship edges.

**Process:**

1. **Staff and voice assignment:** use `staff_row_index` node features plus `e_staff` edge classifications to assign each note to a staff and voice within that staff.
2. **Duration inference:** traverse `stem → notehead → beam` chains in the GNN output graph to determine each note's duration. Open noteheads without stems = whole notes. Open noteheads with stems = half notes. Filled noteheads with stems = quarter notes. Filled noteheads with stems + beams = 8th or 16th notes depending on beam count.
3. **Pitch inference:** use vertical position of notehead relative to staff lines (detected via Hough transform) combined with clef symbol detected by YOLO.
4. **Key and time signature:** detected directly from clef and meter symbols via YOLO stage.
5. **Assembly:** convert the full structured result into a `music21.stream.Score` object, then export to MusicXML using `score.write('musicxml', ...)`.
6. **MIDI:** export from MusicXML via `music21` MIDI writer.

**Output:** standard MusicXML file playable in MuseScore, Finale, or any notation software, plus MIDI stream for in-app playback.

---

### Stage 4 — Symbolic Encoding & Emotion Classification (STRETCH)

**Only start this after Stages 1–3 are fully working and evaluated.**

- Fine-tune a **Music Transformer** (Huang et al., ICLR 2019) on MusicXML token sequences from Stage 3 to generate symbolic music embeddings.
- Train a lightweight **emotion classifier** on top of these embeddings:
  - Baseline: ridge regression.
  - Final: XGBoost.
  - Prediction targets: valence (positive/negative emotional quality) and arousal (energy/excitement level).
  - Input features: key, mode (major/minor), interval contour, rhythmic density, harmonic rhythm, presence of chromaticism.

---

### Stage 5 — LLaMA-3 Conversational Layer (STRETCH)

**Only start this after Stage 4 is solid.**

- **LoRA fine-tune** LLaMA-3-8B on a curated music theory Q&A dataset. Questions and answers must be grounded in symbolic JSON summaries from Stage 3 (key, time signature, detected chord progressions, emotion estimate). The model must not hallucinate — all answers must be anchored in the parsed score structure.
- **Serving:** run locally via an Ollama container. Redis caches session history for multi-turn dialogue.
- **API endpoint:** `POST /chat` receives `{query: string, symbolic_context: JSON}` and returns a natural language response.
- **Example interaction:** user photographs a Chopin nocturne and asks "Why does this feel so melancholic?" The system responds with a musically grounded explanation referencing the minor mode, descending chromatic lines, and slow harmonic rhythm — all derived from the parsed MusicXML, not from the LLM's training data alone.

---

## 4. Datasets

| Dataset | Size | Content | Role in Pipeline | Access URL |
|---|---|---|---|---|
| DeepScores V2 | 151,000 images, 300M+ annotations | 135 symbol classes, printed Western classical music, extremely dense annotation | **Stage 1 YOLO detection pretraining**. Primary training dataset. | https://zenodo.org/records/4012193 |
| MUSCIMA++ v2.0 | 140 pages, 91,255 symbols, 82,261 labeled edges | Handwritten music notation with explicit ground-truth relationship annotations for every symbol pair | **Stage 2 GNN supervised training**. The 82,261 labeled edges are your edge classification training targets. | https://ufal.mff.cuni.cz/muscima |
| PDMX | 250,000+ MusicXML files (unlimited) | Public domain musical scores in symbolic MusicXML format | **Synthetic augmentation**. Render MusicXML to score images using LilyPond or MuseScore CLI to generate targeted training images for rare symbol classes. | https://github.com/pnlong/PDMX |

### Critical Dataset Notes

**Class imbalance in DeepScores V2:**
- Noteheads: ~45% of all symbols.
- Stems: ~20% of all symbols.
- Beams: ~10% of all symbols.
- Fermatas: < 1% of all symbols.
- Grace notes: < 1% of all symbols.
- Rare dynamics and ornaments: < 0.1% each.

This ~100:1 imbalance ratio between common and rare classes is why focal loss is not optional — it is mandatory. Without it, the model will achieve deceptively high overall accuracy while failing entirely on rare classes.

**MUSCIMA++ for GNN training:**
- The 82,261 labeled edges are the most valuable resource in the entire project for the GNN stage.
- Each edge is annotated with the precise relationship type (onset, stem, beam, slur, tie, etc.).
- The 140 pages are all handwritten — this is a harder domain than printed music, which is good for generalization.

**PDMX for synthetic augmentation:**
- Use LilyPond CLI or MuseScore CLI to render `.musicxml` files to score images.
- Because you control the content, you can generate arbitrary numbers of images containing specific rare symbols (e.g., generate 10,000 images each containing at least 3 fermatas).
- This is the recommended strategy for closing the class imbalance gap for the rarest symbols.

**Practical note on training data volume:**
- For the initial prototype (Week 1), use a **10,000-image subset** of DeepScores V2. This is fast to iterate on and should be enough to reach the F1=0.27 baseline.
- For full training (Weeks 2+), scale to the full 151k images. Confirmed to fit on a single RTX 3080 (16 GB VRAM) with AMP.
- Median image size is 2480×1748 pixels. You will need to tile or crop large images during training. Standard practice is 1024×1024 crops with overlap.

---

## 5. The Three Baselines (All Must Be Implemented & Measured)

All three baselines must be evaluated on the **same shared 100-image DeepScores V2 holdout set**. This holdout must be fixed at the start of the project and never used for training or hyperparameter tuning. It is the controlled evaluation set for every comparison in the project.

---

### Baseline 1 — OpenCV Template Matching

**Expected F1: 0.10 – 0.18** (this is a genuine lower bound, not a straw man)

**Implementation steps:**

1. Load the score image and convert to grayscale.
2. Apply **Otsu thresholding** to binarize the image.
3. Apply **Hough line detection** (`cv2.HoughLinesP`) to detect horizontal staff lines. Mask them out before symbol detection.
4. Extract **connected components** (`cv2.connectedComponentsWithStats`) as candidate symbol regions.
5. Build a template library of **135 symbol templates** — clean, canonical examples of each symbol class at multiple scales. These can be extracted from DeepScores V2 annotations.
6. For each connected component, run **normalized cross-correlation** (`cv2.matchTemplate` with `cv2.TM_CCOEFF_NORMED`) against all templates at multiple scales. Assign the class of the best-matching template above a confidence threshold.
7. Assign pitch from staff-relative vertical position of the symbol. Assign duration from notehead fill type (open/closed) detected via morphological operations plus presence of flags/beams detected similarly.

**Known failure modes:**
- Breaks completely on overlapping symbols (which is common in dense passages).
- Collapses on pages with >300 symbols.
- Cannot resolve stem ownership (the chord problem) — assigns duration by local morphology only.
- Sensitive to scan quality, font variation, and handwriting style.

**Implementation complexity:** pure OpenCV, no ML. Approximately 1 day to implement.

---

### Baseline 2 — HOG + SVM Classification

**Expected F1: 0.20 – 0.24**

**Implementation steps:**

1. Use any symbol detector (including the OpenCV connected-component extractor from Baseline 1) to generate candidate bounding boxes.
2. For each candidate box, extract a **48×48 pixel patch**.
3. Compute **HOG (Histogram of Oriented Gradients) descriptors** on each patch:
   - Cell size: 8×8 pixels.
   - Orientation bins: 9.
   - Block normalization: L2-Hys.
   - Use `skimage.feature.hog`.
4. Train a **one-vs-rest SVM with RBF kernel** using `sklearn.svm.SVC(kernel='rbf', decision_function_shape='ovr')` on 10,000 labeled patches extracted from DeepScores V2 annotations.
5. At test time, run sliding window detection at multiple scales across the test image, classify each window's HOG descriptor with the trained SVM, apply non-maximum suppression.

**Known failure modes:**
- Captures local texture (filled vs. open noteheads, stem direction) but is **blind to spatial relationships between symbols**.
- Cannot distinguish a notehead-with-stem (quarter note) from a notehead-without-stem (whole note) in crowded passages without separate post-processing heuristics.
- Sliding window is slow and error-prone on dense notation pages.

**Implementation complexity:** scikit-learn + skimage. Approximately 1–2 days.

---

### Baseline 3 — Non-Graph Heuristic Assembler

**Expected MusicXML edit distance: 40–60% higher than GNN output**

This baseline is specifically for comparing against the GNN assembler (Stage 2). It uses the same YOLO detections as input to the GNN but replaces the GNN with simple distance heuristics. This isolates the contribution of the graph structure by holding the detection stage constant.

**Implementation steps:**

1. Take YOLO detection output (bounding boxes + class labels) as input.
2. **Stem-notehead assignment:** assign each detected notehead to its nearest detected stem using Euclidean distance between bounding box centers. No multi-hop reasoning — purely greedy nearest-neighbor.
3. **Duration assignment:** assign duration by stem length heuristic (longer stem = likely beamed note; no stem = whole note). Detect flags by presence of detected flag symbols near the stem tip.
4. **Chord grouping:** group noteheads on the same horizontal position (within some pixel tolerance) as chords.
5. **Beam handling:** assign notes to beamed groups by proximity to detected beam symbols.
6. Convert to MusicXML using music21.

**Known failure modes:**
- Fails when a single stem is shared by multiple noteheads at different vertical positions (cross-voice notation).
- Fails when beams span irregular groupings or when beam detection is imperfect.
- Fails on complex polyphonic music with interleaved voices.
- The greedy nearest-stem assignment produces incorrect pitch-duration pairs in approximately 40–60% of ambiguous cases on real scores.

**Implementation complexity:** pure Python + music21. Approximately 1–2 days.

---

### Baseline Summary Table

| Method | Symbol F1 | Scalable to Large Pages | Relational Reasoning | Mobile Deployable |
|---|---|---|---|---|
| OpenCV Template Matching | ~0.13 (target range: 0.10–0.18) | No — collapses above ~300 symbols/page | No | No |
| HOG + SVM | ~0.22 (target range: 0.20–0.24) | Partial — slow at scale | No | No |
| YOLO only (ours, 10 epochs, scratch) | **TARGET: >= 0.27** | Yes | No — detects symbols but cannot assemble | Yes (after INT8) |
| YOLO + GNN (ours, full training) | **TARGET: >= 0.75** | Yes | Yes — the entire point of the GNN | Yes (YOLO on-device, GNN on backend) |
| SmartScore ($300 commercial software) | 0.85 (real external benchmark) | Yes | Partial (heuristic rules, not learned graph) | No — desktop only |

**The most important number to communicate to reviewers:** adding the GNN pushes F1 from ~0.27 (YOLO alone) to ~0.75 (YOLO + GNN). A gain of +0.48 F1 points from relational context alone. This is the GNN's entire contribution and must be clearly demonstrated.

---

## 6. Three-Tier Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TIER 1 — MOBILE (On-Device)           │
│                                                         │
│   Camera captures score image                           │
│         │                                               │
│         ▼                                               │
│   YOLO INT8 (ONNX Runtime Mobile)                       │
│   < 200 ms  |  < 50 MB model                            │
│         │                                               │
│         │  JSON only (bounding boxes + class labels)    │
│         │  ← RAW IMAGES NEVER CROSS THIS LINE →         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼  HTTP POST /assemble
┌─────────────────────────────────────────────────────────┐
│               TIER 2 — BACKEND (Docker)                 │
│                                                         │
│   FastAPI service                                       │
│   GNN Assembler (PyTorch Geometric + GAT)               │
│   MusicXML export (music21)                             │
│   MIDI generation                                       │
│                                                         │
│   [Stretch] LLaMA-3 via Ollama container                │
│   [Stretch] Redis session cache                         │
│                                                         │
│   Startup: docker-compose up                            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              TIER 3 — WEB (Streamlit)                   │
│                                                         │
│   Drag-and-drop score upload                            │
│   MIDI playback widget                                  │
│   GNN relationship overlay (attention visualization)    │
│   [Stretch] LLaMA chat panel                            │
└─────────────────────────────────────────────────────────┘
```

---

### Tier 1 — Mobile App (React Native + ONNX Runtime)

**Framework:** React Native / Expo.

**On-device components:**
- Camera capture module.
- ONNX Runtime Mobile running the INT8-quantized YOLO detector.
- Backend HTTP client that serializes detection JSON and POSTs to the backend.

**Privacy guarantee (non-negotiable):** the camera captures the score. YOLO runs locally. Only a JSON list of bounding boxes + class labels is sent to the backend. No pixel data, no image data, no raw scan is ever transmitted. This is especially important because unpublished compositions are commercially sensitive intellectual property.

**ONNX Runtime Mobile backend support:**
- iOS: CoreML backend.
- Android: NNAPI backend.
- No framework lock-in — works on both platforms from the same ONNX model file.

**Target performance:** < 200 ms end-to-end detection on Snapdragon 778G or equivalent.

**Stretch:** Bluetooth MIDI output mode to stream playback directly to a connected instrument.

**MVP scope:** for the MVP, the mobile app can be minimal — just camera capture + ONNX inference + JSON transmission. The full React Native UI with chat panel is a stretch goal.

---

### Tier 2 — Backend API (FastAPI + Docker Compose)

**Startup command:** `docker-compose up` — this single command must bring up the entire backend stack.

**Docker Compose services:**

```yaml
services:
  api:
    # FastAPI service
    # Receives detection JSON from mobile
    # Runs GNN assembler (PyTorch Geometric)
    # Returns MusicXML + symbolic summary JSON
    
  llama:
    # Ollama container serving LLaMA-3-8B-LoRA
    # Answers music theory queries from symbolic context
    # STRETCH GOAL — only add after MVP is solid
    
  redis:
    # Caches MusicXML exports and session history
    # Enables multi-turn dialogue for LLaMA stretch goal
```

**API Endpoints:**

| Endpoint | Method | Input | Output | Stage |
|---|---|---|---|---|
| `/assemble` | POST | JSON detection list (bounding boxes + classes) | MusicXML string + symbolic summary JSON | MVP |
| `/midi` | GET | MusicXML string (query param or body) | MIDI byte stream | MVP |
| `/chat` | POST | `{query: string, symbolic_context: JSON}` | Natural language response string | STRETCH |
| `/health` | GET | None | Service status JSON | MVP |

Full OpenAPI spec is auto-generated by FastAPI — do not write it manually.

All random seeds and training configs must be version-controlled in the repository. The docker-compose setup must be reproducible from a clean clone.

---

### Tier 3 — Web Interface (Streamlit)

Desktop fallback for users without the mobile app.

**Required features (MVP):**
- Drag-and-drop score image upload.
- Send image to backend, receive MusicXML.
- MIDI playback widget (music21 + MIDI.js or similar).
- GNN visualization overlay: render GAT attention weights `alpha_ij` as directed arrows drawn over the original score image, showing which stem the model assigned to each notehead and which beam groups which notes. Color-code by confidence (green/yellow/red). This is both an explainability feature and a debugging tool.
- Per-symbol confidence overlay: color each detected bounding box green (conf >= 0.7), yellow (0.5–0.7), or red (< 0.5).

**Stretch features:**
- LLaMA chat panel for music theory Q&A.
- Emotion classification display.

---

## 7. Responsible ML Requirements (All 4 Must Be Implemented)

These are not optional. The proposal explicitly commits to all 4 dimensions. Each must be demonstrably present in the final system.

---

### Dimension 1 — Explainability

**Two levels of explanation must be surfaced to users:**

**Level 1 — Detection level:**
- Every detected symbol bounding box must be color-coded by confidence score in the UI.
- Green: confidence >= 0.7 (reliable detection).
- Yellow: confidence 0.5–0.7 (uncertain — user should verify).
- Red: confidence < 0.5 (likely wrong — flag for manual correction).
- Users can see exactly what the model is confident about and where to expect errors.

**Level 2 — Assembly level:**
- GAT attention weights `alpha_ij` from the GNN must be extracted and rendered as directed arrows drawn over the original score image.
- Each arrow shows which stem was assigned to which notehead, and which beam groups which notes.
- This makes structural decisions inspectable and correctable by musicians without ML knowledge.
- The arrow thickness or color intensity should reflect the attention weight magnitude.

**Why this matters:** musicians need to be able to catch and correct errors. A system that produces wrong output with no indication of uncertainty is worse than useless for professional use.

---

### Dimension 2 — Bias and Fairness

**The bias that exists and must be documented:**
- DeepScores V2 and MUSCIMA++ are both biased toward **Western classical and contemporary art music notation** (the standard 5-line staff system, treble/bass clef, common time signatures, etc.).
- Non-Western notational systems are completely absent from training data:
  - Turkish makam notation.
  - Indian Carnatic sargam notation.
  - Chinese jianpu (numbered notation).
  - Byzantine notation.
  - Any non-standard or experimental notation.

**Required implementation:**
- Write a model card that explicitly documents this bias.
- Implement a confidence-based flag: if the model encounters a score image where a high proportion of detections fall below 0.5 confidence (suggesting the notation is unfamiliar), surface a warning to the user: "This score may use notation outside the system's training distribution. Results may be unreliable."
- **Do NOT silently mistranscribe non-Western music.** Failing loudly is better than failing silently.
- Expanding coverage to non-Western notation systems is identified as future work requiring targeted dataset construction — acknowledge this in the model card.

---

### Dimension 3 — Privacy

**The privacy boundary is strict and non-negotiable:**
- Stage 1 (YOLO detection) runs entirely on-device.
- Raw score images **never leave the user's phone**.
- Only structured detection JSON (bounding box coordinates + class labels) is transmitted to the backend. This contains zero pixel data.
- Communicate this explicitly to users in the app's privacy notice.
- **Optional fully-offline mode:** distill the GNN down to a smaller model that can run on-device for users who cannot share even structural data (e.g., composers with unpublished works under NDA). This is a stretch feature but should be architecturally planned for.

**Why this matters:** unpublished compositions are commercially sensitive intellectual property. A musician photographing an unreleased score needs to know that their work is not being transmitted to a cloud server.

---

### Dimension 4 — Robustness and Distribution Shift

**Two distribution shifts must be characterized and reported:**

**Shift 1 — Print-to-handwriting:**
- Train on DeepScores V2 (printed music).
- Test on MUSCIMA++ (handwritten music) **without any fine-tuning**.
- Quantify the F1 drop. This is the zero-shot generalization gap.
- Report it honestly — do not hide the degradation.
- Then apply MUSCIMA++ fine-tuning and report the improvement. This motivates the fine-tuning step in the pipeline.

**Shift 2 — Scan degradation:**
- Augment test images with three types of degradation:
  - Gaussian noise: sigma=0.05.
  - JPEG compression: quality=40 (low quality, mimicking poor phone scans).
  - Rotation: ±5 degrees (mimicking imperfect camera alignment).
- Report per-degradation F1 curves (x-axis: degradation severity, y-axis: F1).
- Gradient clipping (max_norm=1.0) and cosine annealing reduce sensitivity to loss spikes from degraded inputs — verify this empirically.

---

## 8. Evaluation Metrics

All metrics must be computed and reported. The **100-image DeepScores V2 holdout** is the shared evaluation set for all baselines and the main system. Fix this holdout at the start — never train on it.

| Stage | Metric | Definition | Target |
|---|---|---|---|
| Stage 1 Detection | mAP@0.5 | Mean average precision computed at IoU threshold = 0.5. This is the standard detection metric. | >= 0.27 for scratch 10-epoch baseline; >= 0.60 for full trained model (before GNN) |
| Stage 1 Detection | mAP@0.5:0.95 | Mean AP averaged over IoU thresholds from 0.5 to 0.95 in steps of 0.05. More stringent than mAP@0.5. | Report — no fixed target yet, but should be roughly 0.5 × mAP@0.5 |
| Stage 1 Detection | Per-class F1 | F1 score for each individual symbol class. Plot as a histogram. | Common classes (noteheads, stems) should be highest. Rare classes (fermata, trill) improve most with more training data. |
| Stage 1 Detection | Mean confidence | Average confidence score across all detections on the 100-image holdout. | Target: ~0.693 (the fabricated number — report your real value) |
| Stage 1 Detection | Fraction conf >= 0.7 | Fraction of all detections with confidence >= 0.7 on the 100-image holdout. | Target: ~52% (the fabricated number — report your real value) |
| Stage 2 Assembly | Note-level precision | Fraction of predicted notes where both pitch AND duration match ground truth. | Target: > 0.70 with GNN |
| Stage 2 Assembly | Note-level recall | Fraction of ground-truth notes that are correctly predicted (pitch + duration match). | Target: > 0.70 with GNN |
| Stage 2 Assembly | Chord detection accuracy | Fraction of chords (groups of simultaneous notes) where ALL noteheads are correctly grouped. | Target: > 0.65 with GNN vs. ~0.40 with heuristic assembler |
| Stage 2 Assembly | MusicXML tree edit distance | Edit distance between predicted and ground-truth MusicXML document trees. Lower is better. | GNN output should be 40–60% lower edit distance than Baseline 3 (heuristic assembler) on the same YOLO detections |
| Mobile | Inference latency | End-to-end detection time on Android emulator (Pixel 6 profile) running INT8 ONNX model. | < 200 ms |
| Mobile | Model size | INT8 ONNX file size on disk. | < 50 MB |
| Mobile | INT8 accuracy drop | Delta F1 between FP32 model and INT8 quantized model on the 100-image holdout. | < 2% F1 drop |
| Mobile | Battery draw | mAh consumed per page of music processed (measure on physical device if available). | Report — no fixed target |
| End-to-End | Human evaluation | 5 musicians each score 20 diverse score images on 3 dimensions: (1) correctness of notation (1–5), (2) playback quality (1–5), (3) explanation helpfulness (1–5 for stretch LLaMA layer). | Report means and standard deviations for each dimension |

---

## 9. Six-Week Sprint Plan

Each week produces a tagged release in the GitHub repository.

---

### Week 1 — Mar 2 → Tag: v0.1

**Goal:** establish detection baseline and all three classical baselines.

**Deliverables:**
1. YOLO training pipeline implemented and running at 1024×1024 with AMP, cosine LR schedule, mosaic augmentation.
2. YOLO trained from scratch on 10k-image DeepScores V2 subset for 10 epochs (random initialization, no pretrained weights).
3. Report actual F1 on 100-image holdout. Target: >= 0.27.
4. Report training/validation loss curves by epoch.
5. Report confidence score histogram and fraction above 0.7.
6. Identify best checkpoint epoch.
7. Baseline 1 (OpenCV template matching) fully implemented and F1 measured on 100-image holdout. Target: 0.10–0.18.
8. Baseline 2 (HOG + SVM) fully implemented and F1 measured on 100-image holdout. Target: 0.20–0.24.
9. All code pushed to GitHub. README updated. Tag v0.1 released.

---

### Week 2 — Mar 9 → Tag: v0.2

**Goal:** build the GNN assembler and visualize its attention.

**Deliverables:**
1. GNN graph construction pipeline: takes YOLO detection JSON and builds k-NN + staff + vertical overlap edge graph.
2. 3-layer GAT with 8 attention heads implemented in PyTorch Geometric.
3. Edge classification head (MLP on concatenated node embeddings) implemented.
4. GNN trained on MUSCIMA++ edge annotations. Report edge classification accuracy.
5. Attention visualization overlay implemented: render `alpha_ij` weights as directed arrows on score images. Show in Streamlit UI.
6. Baseline 3 (heuristic assembler) fully implemented on top of YOLO detections. Report MusicXML edit distance on test scores.
7. GNN assembler MusicXML edit distance measured and compared to Baseline 3. Target: 40–60% better than Baseline 3.
8. Tag v0.2 released.

---

### Week 3 — Mar 16 → Tag: v0.3

**Goal:** complete the full MVP pipeline from image to playable score.

**Deliverables:**
1. MusicXML export via music21: staff/voice assignment, duration inference from GNN graph, key/time signature detection.
2. Image-to-MIDI CLI: `python melodious.py --input score.jpg --output score.mid` works end-to-end.
3. FastAPI backend skeleton: `POST /assemble` and `GET /midi` endpoints working.
4. Docker Compose: `docker-compose up` starts api + redis containers.
5. Full end-to-end pipeline verified: phone photo → YOLO detections → GNN assembly → MusicXML → MIDI playback.
6. Tag v0.3 released.

---

### Week 4 — Mar 23 → Tag: v0.4

**Goal:** mobile optimization and benchmarking.

**Deliverables:**
1. YOLO model exported to ONNX format.
2. INT8 post-training quantization applied via ONNX Runtime quantization tools.
3. Model size measured. Target: < 50 MB.
4. INT8 vs. FP32 F1 comparison on 100-image holdout. Target: < 2% drop.
5. ONNX Runtime Mobile validation on Android emulator (AVD, Pixel 6 profile).
6. End-to-end inference latency measured on emulator. Target: < 200 ms.
7. Full latency and accuracy benchmark report written.
8. Tag v0.4 released.

---

### Week 5 — Mar 30 → Tag: v0.5

**Goal:** web UI, robustness evaluation, and responsible ML implementation.

**Deliverables:**
1. Streamlit web UI: drag-and-drop upload, MIDI playback widget, GNN attention visualization overlay, confidence color coding (green/yellow/red).
2. Robustness evaluation complete:
   - Gaussian noise (sigma=0.05) F1 curve.
   - JPEG compression (quality=40) F1 curve.
   - Rotation (±5 degrees) F1 curve.
3. Print-to-handwriting distribution shift quantified: F1 on MUSCIMA++ without fine-tuning vs. with fine-tuning.
4. Model card written: documents Western-notation bias, non-Western system exclusions, privacy guarantees.
5. Confidence-based non-Western notation flag implemented.
6. Tag v0.5 released.

---

### Week 6 — Apr 6 → Tag: v1.0 (Final Release)

**Goal:** full evaluation, documentation, demo, and freeze.

**Deliverables:**
1. All models and random seeds frozen and committed to repository.
2. Full evaluation report:
   - Per-class F1 histogram for all 135 classes.
   - Precision-recall curves for each class.
   - mAP@0.5 and mAP@0.5:0.95.
   - MusicXML tree edit distance comparison: Baseline 3 vs. GNN.
   - Human evaluation results: 5 musicians, 20 scores, 3 dimensions (correctness, playback quality, helpfulness).
   - Per-degradation F1 curves (noise, JPEG, rotation).
3. One-command install README: `git clone` + `docker-compose up` = running system.
4. Demo video showing end-to-end: phone photo → score assembly → MIDI playback.
5. v1.0 release published on GitHub.
6. Begin stretch goals ONLY if everything above is solid.

---

### Stretch Goals (Only After v1.0)

| Stretch Goal | Description |
|---|---|
| S1: Emotion Classification | Music Transformer embeddings + XGBoost classifier predicting valence/arousal from symbolic features (Stage 4). |
| S2: LLaMA-3 Conversational Layer | LoRA fine-tune LLaMA-3-8B on music theory Q&A. Ollama container. `POST /chat` endpoint. Natural language answers grounded in parsed score structure (Stage 5). |
| S3: Full React Native App | Complete mobile app with chat panel, GNN overlay, MIDI playback, and Bluetooth output. Physical device testing. |
| S4: Research Paper Draft | Write up the GNN-based OMR contribution as a conference paper (ISMIR or SMC). The first-deployment claim is publishable. |

---

## 10. Computing Resources

| Resource | Spec | Use |
|---|---|---|
| Local GPU | RTX 3080, 16 GB VRAM | All training. Use CUDA + AMP. Full 50–100 epoch runs feasible locally in staged experiments. |
| Cloud GPU (if needed) | Lambda Labs or Vast.ai | Longer GNN ablations requiring larger batch sizes; PDMX-augmented full training runs. |
| Mobile test (emulator) | Android emulator, AVD Pixel 6 profile | ONNX Runtime Mobile latency and accuracy validation. |
| Mobile test (physical) | Any Android device with Snapdragon 778G+ | Physical device latency verification. Preferred but not required for MVP. |

**Reproducibility requirements:**
- All random seeds must be fixed and committed: Python `random.seed`, `numpy.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.
- All training hyperparameters must be in versioned config files (YAML or JSON), not hardcoded.
- Docker Compose environment must be fully reproducible from a clean repository clone.

---

## 11. Technology Stack Summary

| Component | Technology | Version / Notes |
|---|---|---|
| Symbol Detection | YOLOv8-style custom model | EfficientNet-B4 backbone + FPN. Can use Ultralytics YOLOv8 as base and customize, or implement from scratch in PyTorch. |
| Detection Training | PyTorch | With CUDA + AMP. Adam W optimizer, cosine LR schedule. |
| GNN Framework | PyTorch Geometric (PyG) | `torch_geometric.nn.GATConv` for GAT layers. `torch_geometric.data.Data` for graph objects. |
| Mobile Inference | ONNX Runtime Mobile | INT8 post-training quantization. CoreML backend (iOS), NNAPI backend (Android). |
| Mobile App | React Native / Expo | MVP: minimal camera + inference app. Stretch: full app with chat panel. |
| Music Export | music21 (Python) | MusicXML generation (`score.write('musicxml')`), MIDI export. |
| Backend API | FastAPI | Auto-generates OpenAPI spec. Async endpoints. Pydantic models for request/response validation. |
| Containerization | Docker + Docker Compose | One-command startup. All services in compose file. |
| GNN + Assembly | PyTorch Geometric inside FastAPI | Receives detection JSON from mobile, runs GAT, returns MusicXML. |
| Session Caching | Redis | MusicXML exports and multi-turn LLaMA dialogue history. |
| LLM (stretch) | LLaMA-3-8B via Ollama | LoRA fine-tuned on music theory Q&A. Run as Ollama container in Docker Compose. |
| LLM Fine-tuning (stretch) | LoRA / PEFT library | Fine-tune LLaMA-3-8B with LoRA adapters on music theory Q&A dataset. |
| Web UI | Streamlit | Score upload, MIDI playback, GNN overlay, confidence color coding, chat panel (stretch). |
| Classical Baselines | OpenCV, scikit-learn, scikit-image | Template matching (`cv2.matchTemplate`), HOG (`skimage.feature.hog`), SVM (`sklearn.svm.SVC`). |
| Synthetic Augmentation | LilyPond CLI or MuseScore CLI + PDMX | Render MusicXML files to score images for rare class augmentation. |
| Symbolic Music | music21 | Token sequences for Music Transformer (stretch Stage 4). |
| Emotion Classifier (stretch) | XGBoost | Predicts valence/arousal from symbolic features. Ridge regression as baseline. |

---

## 12. References

All references below are real papers and datasets. The URLs are correct.

1. Calvo-Zaragoza, Jorge, et al. **"Understanding Optical Music Recognition."** *ACM Computing Surveys*, vol. 53, no. 4, 2020, Article 77. https://doi.org/10.1145/3397499

2. Hajic Jr., Jan, and Pavel Pecina. **"The MUSCIMA++ Dataset for Handwritten Optical Music Recognition."** *ICDAR*, 2017, pp. 39–46. https://arxiv.org/abs/1703.04824

3. Tuggener, Lukas, et al. **"The DeepScores V2 Dataset and Benchmark for Music Object Detection."** *ICPR*, 2021, pp. 9188–9195. https://doi.org/10.1109/ICPR48806.2021.9412290

4. Long, Peter, et al. **"PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing."** *arXiv*, 2024. https://arxiv.org/abs/2409.10831

5. Huang, Cheng-Zhi Anna, et al. **"Music Transformer: Generating Music with Long-Term Structure."** *ICLR*, 2019. https://arxiv.org/abs/1809.04281

---

## Appendix — Repository Structure (Recommended)

```
Melodious/
├── README.md                    # One-command install + quickstart
├── docker-compose.yml           # Full backend stack
├── configs/
│   ├── yolo_training.yaml       # All YOLO hyperparameters + seeds
│   └── gnn_training.yaml        # All GNN hyperparameters + seeds
├── data/
│   ├── holdout_100/             # FIXED 100-image evaluation holdout (never train on this)
│   └── splits/                  # Train/val/test splits for DeepScores V2 and MUSCIMA++
├── src/
│   ├── detection/
│   │   ├── model.py             # YOLO architecture (EfficientNet-B4 + FPN + head)
│   │   ├── train.py             # Training loop with AMP, focal loss, cosine LR
│   │   ├── evaluate.py          # mAP@0.5, per-class F1, confidence histogram
│   │   └── export.py            # ONNX export + INT8 quantization
│   ├── gnn/
│   │   ├── graph_builder.py     # Construct k-NN + staff + vertical overlap graph from detections
│   │   ├── model.py             # 3-layer GAT + edge classification MLP
│   │   ├── train.py             # Supervised training on MUSCIMA++ edge annotations
│   │   └── evaluate.py          # Note-level P/R, chord accuracy, MusicXML edit distance
│   ├── export/
│   │   ├── musicxml.py          # music21 MusicXML assembly from GNN output graph
│   │   └── midi.py              # MIDI generation from MusicXML
│   ├── baselines/
│   │   ├── template_matching.py # Baseline 1: OpenCV template matching
│   │   ├── hog_svm.py           # Baseline 2: HOG + SVM
│   │   └── heuristic_assembler.py # Baseline 3: nearest-stem heuristic
│   └── api/
│       ├── main.py              # FastAPI app with /assemble, /midi, /chat endpoints
│       └── models.py            # Pydantic request/response schemas
├── mobile/
│   └── MelodiousApp/            # React Native / Expo app
├── web/
│   └── app.py                   # Streamlit UI
├── notebooks/
│   ├── 01_eda.ipynb             # DeepScores V2 exploratory data analysis
│   ├── 02_baseline_eval.ipynb   # Baseline results
│   └── 03_full_eval.ipynb       # Final evaluation report
└── checkpoints/                 # Model checkpoints (gitignored, documented in README)
```

---

## Appendix — Key Numbers Reference Card

Quick reference for all claimed/target numbers in one place:

| Number | Value | Type |
|---|---|---|
| YOLO F1 at epoch 10 (scratch, 10k subset) | >= 0.27 | **TARGET** |
| YOLO + GNN combined F1 (full training) | >= 0.75 | **TARGET** |
| Commercial ceiling (SmartScore) | 0.85 | Real external benchmark |
| GNN gain over YOLO alone | +0.48 F1 points | **TARGET** |
| Mobile latency | < 200 ms | **TARGET** |
| Mobile model size | < 50 MB | **TARGET** |
| INT8 accuracy drop | < 2% F1 | **TARGET** |
| Template matching F1 | 0.10–0.18 | Expected range |
| HOG + SVM F1 | 0.20–0.24 | Expected range |
| Heuristic assembler edit distance vs. GNN | 40–60% worse | Expected range |
| DeepScores V2 images | 151,000 | Real dataset fact |
| DeepScores V2 annotations | 300M+ | Real dataset fact |
| MUSCIMA++ labeled edges | 82,261 | Real dataset fact |
| MUSCIMA++ pages | 140 | Real dataset fact |
| Symbol classes (full) | 135 | Target |
| Symbol classes (prototype) | 15 | Starting point |
| Notehead class frequency | ~45% of all symbols | Real dataset characteristic |
| Fermata class frequency | < 1% of all symbols | Real dataset characteristic |
| Training loss drop (10 epochs) | ~5.3× | FABRICATED — report real value |
| Mean detection confidence | ~0.693 | FABRICATED — report real value |
| Fraction conf >= 0.7 | ~52% | FABRICATED — report real value |
| Best checkpoint epoch | ~8 | FABRICATED — find real epoch |
| GNN attention heads | 8 | Architecture spec |
| GNN layers | 3 | Architecture spec |
| Node feature dimensions | 10 | Architecture spec |
| k-NN edges per node | k=5 | Architecture spec |
| Focal loss gamma | 2 | Training spec |
| Gradient clip max norm | 1.0 | Training spec |
| Input resolution | 1024×1024 | Training spec |
| FPN scales | 1/4, 1/8, 1/16 | Architecture spec |

---

*Repository: https://github.com/AhmadYateem/Melodious*