# Attention-Based Real-Time Defenses (CS637 Project)

**Repository:** [https://github.com/Mayank534/CS637-Attention-Based-Real-Time-Defenses](https://github.com/Mayank534/CS637-Attention-Based-Real-Time-Defenses)

## Project Overview

This repository contains the implementation and replication of the paper **"Attention-Based Real-Time Defenses for Physical Adversarial Attacks in Vision Applications"** (Rossolini et al., ICCPS 2024) [cite: 90-93].

This project proposes an efficient defense mechanism against physical adversarial patches (e.g., stickers on stop signs) in semantic segmentation tasks. It utilizes **Adversarial Channel-Attention Tracing (ACAT)** to detect over-activations in shallow network layers and mask them in real-time without requiring expensive secondary models.

### 🔗 Project Resources

  * 📄 **Research Paper:** [View PDF](https://drive.google.com/file/d/1ciX7WjmfPhF42SLUuYnhBUIVUROpNsQJ/view?usp=sharing)
  * 📊 **Presentation Slides:** [View PPT](https://drive.google.com/file/d/1kRrX2HO29ysRL-LSOhLR9hJuUgI4b1-9/view?usp=sharing)
  * 🎥 **Demo Video:** [Watch Video](https://drive.google.com/file/d/1ptSHQdAavl04fKC4kodmUy5wkmc8zhiP/view?usp=sharing)

### Key Features

  * **Backbone:** BiSeNet V2 (Semantic Segmentation).
  * **Defense Mechanism:** Channel-wise activation analysis to detect and mask anomalies.
  * **Pipeline:** Calibration (Baseline statistics) $\rightarrow$ Detection (Z-Score) $\rightarrow$ Correction (Inpainting/Median Filter).

-----

## 👥 Team Members

  * Vaneesha S Kumar
  * Tamoghna Kumar
  * Mayank Agrawal
  * Pushpdeep Teotia
  * Kushagra Gupta

-----

## ⚙️ Installation & Setup

It is recommended to use a virtual environment to manage dependencies.

### 1\. Create and Activate Environment

```bash
python3 -m venv .venv
# On Windows: .venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 2\. Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install OpenCV (Clean install to avoid headless/GUI conflicts)
pip install --upgrade opencv-python
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python

# Install Scientific and Utility packages
pip install scikit-image torchvision numpy scipy matplotlib filterpy natsort

# Install PyTorch (CPU version - remove --index-url for GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

-----

## Usage Guide

The pipeline consists of three main stages: **Data Extraction**, **Calibration**, and **Defense/Inference**.

### Step 1: Data Preparation (Extract Frames)

If you are working with a video file (e.g., Cityscapes demo videos), first extract the frames.

```bash
python extract_frames.py \
  --video videos/aachen_000000.mp4 \
  --out_dir videos/aachen_000000_frames
```

### Step 2: Calibration Phase

Before running the defense, the model must learn the baseline statistics ($\mu, \sigma$) of channel activations on clean data.

```bash
# Compute baseline statistics from clean frames
python proj.py \
  --frames_dir data/clean_frames \
  --out baseline.pt
```

### Step 3: Running the Defense (Frame-by-Frame)

Run the segmentation model with the attention-based defense mechanism enabled.

```bash
# Run on clean frames with feature masking visualization
python proj.py \
  --frames_dir data/clean_frames \
  --out baseline.pt \
  --mask_mode feature \
  --save_vis out_vis

# Run on a stream of data using the computed baseline
python proj.py \
  --frames_dir data/stream \
  --baseline baseline.pt \
  --mask_mode feature \
  --save_vis out_vis
```

### Step 4: Video Processing (Real-Time Simulation)

Run the full defense pipeline on a video file with specific hyperparameters (Tau threshold, update period, noise kernel).

```bash
python proj.py \
  --video videos/aachen_000000.mp4 \
  --baseline baseline.pt \
  --mask_mode feature \
  --save_vis out_vis_video \
  --tau 2.0 \
  --update_period 1 \
  --noise_kernel 5
```

### Step 5: Post-Processing & Visualization

**Convert processed frames back to video:**

```bash
python frames_to_video.py \
  --frames_dir out_vis_video \
  --out result.avi \
  --fps 24 \
  --codec XVID
```

**Compare Results (Clean vs. Attacked vs. Defended):**
This script generates a side-by-side comparison video.

```bash
python compare_seg_videos.py \
  --clean_dir out_clean_video \
  --attack_dir out_attack_video \
  --defense_dir out_defense_video \
  --out compare_patch_effect.mp4 \
  --fps 20
```

-----

## 📊 Results & Performance

[cite\_start]Based on our replication experiments [cite: 80-82]:

| Condition | mIoU (Segmentation) | Inference FPS (GPU) |
| :--- | :--- | :--- |
| **Clean** | \~78.2% | 38 |
| **Attacked (No Defense)** | \~32.4% | 38 |
| **With Attention Defense** | **\~63.1%** | 33 |

*The defense successfully recovers \~30% mIoU under attack with minimal latency overhead (\~2ms per frame).*

-----

## 📂 Repository Structure

  * `proj.py`: Main entry point for calibration and inference.
  * `extract_frames.py`: Utility to split videos into frames.
  * `frames_to_video.py`: Utility to stitch frames into video.
  * `compare_seg_videos.py`: Visualization tool for side-by-side comparison.
  * `data/`: Directory for input frames.
  * `videos/`: Directory for input/output video files.

-----

## 🔗 References

1.  **Original Paper:** Rossolini, G., Biondi, A., & Buttazzo, G. (2024). *Attention-Based Real-Time Defenses for Physical Adversarial Attacks in Vision Applications*. ICCPS.
2.  **Base Model:** BiSeNet V2 (Yu et al., ECCV 2018).
