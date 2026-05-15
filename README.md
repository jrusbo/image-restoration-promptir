# PromptIR for Multi-Degradation Image Restoration (Rain & Snow)

## Introduction
This repository contains the implementation of an advanced Prompt-Learning Multi-Degradation Image Restoration network based on the PromptIR architecture. The model is built entirely from scratch to restore images degraded by both rain streaks and snow particles. 

Key improvements to the baseline PromptIR include:
*   **Prompt Generation Module (PGM):** Constrained to precisely 5 learnable prompt components.
*   **Degradation-Guided Perturbation Blocks (DGPB):** Relocated into the skip connections to filter high-resolution encoder features effectively.
*   **Dual-Domain Loss:** A composite loss function utilizing Charbonnier Loss (spatial) and Fast Fourier Transform (FFT) Focal Frequency Loss (spectral).
*   **Test-Time Adaptations:** Integration of a Test-Time Local Converter (TLC) to bridge train-test gap and an 8-fold geometric self-ensemble for hallucination suppression.

## Environment Setup
This project uses `uv` for lightning-fast dependency management.

1. Install `uv` if you haven't already:
   ```bash
   pip install uv
    ```
   
## Usage

### Training (Kaggle 2x T4 Distributed Setup)
The training script is designed for Distributed Data Parallel (DDP) execution to fully utilize 2 GPUs. Ensure your dataset is located in `dataset/train` with `clean` and `degraded` subfolders.

```bash
# Execute on 2 GPUs via torchrun
torchrun --nproc_per_node=2 src/train.py --data_dir dataset/train --epochs 100 --batch_size 8 --lr 2e-4
```

---

### How to Execute This Repository

1.  **Directory Setup:**
    Create the root folder `image-restoration-promptir`. Place `pyproject.toml`, `.gitignore`, and `README.md` at the root. Create a `src` directory and copy `dataset.py`, `train.py`, `predict.py`, and `metrics.py` into it. Create `src/model` and place `__init__.py` (empty), `blocks.py`, and `promptir.py` inside. Place your `dataset` folder directly at the root.

2.  **Install Libraries:**
    Inside the root folder, run:
    
```bash
    pip install uv
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

3.  **Run Training:**
    On Kaggle (assuming 2 GPUs are active), run:
    ```bash
    torchrun --nproc_per_node=2 src/train.py --data_dir dataset/train
    ```

4.  **Generate Submission:**
    Once training is finished, grab the latest weights from the `checkpoints` folder and run:
    
    ```bash
    python src/predict.py --data_dir dataset/test --weights checkpoints/promptir_epoch_100.pth
    ```
    This will produce the `pred.npz` file, ready to submit to CodaBench.
