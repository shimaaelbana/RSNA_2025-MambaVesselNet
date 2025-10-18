# RSNA_2025-MambaVesselNet
RSNA Intracranial Aneurysm Detection using MambaVesselNet++ architecture.


## 📘 Overview

Medical image segmentation is essential for accurate computer-aided diagnosis. While Vision Transformers have achieved great success, their quadratic self-attention mechanism leads to high computational costs. The recently proposed Mamba state space model provides an efficient alternative for capturing long-range dependencies with reduced memory usage.

In this work, we apply the MambaVesselNet++ architecture — a hybrid CNN-Mamba framework — to the RSNA Intracranial Aneurysm Detection 2025
dataset using MRA scans. We adapt the original model to perform binary vessel segmentation, converting the original multi-class labels into vessel (1) and background (0).

Our customized RSNA-MambaVesselNet++ effectively models both local texture details and long-range spatial dependencies, achieving robust vessel segmentation performance suitable for clinical aneurysm.

✨ Highlights

🔹 Adapted MambaVesselNet++ for RSNA Intracranial Aneurysm Detection (MRA) data.

🔹 Converted multi-class segmentation masks into binary vessel vs. background for improved clarity.

🔹 Combined CNN-based texture extraction with Mamba-based long-range modeling.

🔹 Achieved accurate and efficient vessel segmentation tailored for aneurysm detection tasks.

🔹 Trained and tested on **A100 GPU** for optimal performance
This study uses the publicly available datasets related to cerebrovascular segmentation. Accessible at [MRA RSNA 2025 Competition Page](https://www.kaggle.com/datasets/shymaaelbana/vessel-mamba-mra)

🚀## Installation

✅ Step 1 — Clone the Repository
Please make sure your environment supports **CUDA ≥ 12.4**

```bash
git clone https://github.com/shimaaelbana/RSNA_2025-MambaVesselNet.git
cd RSNA_2025-MambaVesselNet
```


✅ Step 2 — Install PyTorch (CUDA 12.4 support)

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

✅ Step 3 — Fix Ninja installation (optional but recommended)

```bash
pip uninstall -y ninja
pip install ninja --no-cache-dir --force-reinstall
```

✅ Step 4 — Install causal-conv1d

```bash
cd causal-conv1d
python setup.py install
cd ..
```

✅ Step 5 — Install Mamba

```bash
cd mamba
python setup.py install
cd ..
```

✅ Step 6 — Install Required Libraries

```bash
pip install loguru monai nibabel tqdm scikit-image SimpleITK torchprofile
```

🧠 Training & Evaluation

🔹 Train the Model

```bash
!python /MambaVesselNet/train.py \
    --dataset /dataset.json \
    --checkpoint_dir /Mamba_Results \
    --pretrain_weights /Mamba_Results/MRA_BS8_5000_warmstartbest_model.ckpt \
    --batch_size 8 \
    --max_iter 5000 \
    --num_classes 2 \
    --valid_iter 200 \
    --lr 1e-4 \
    --num_workers 0 \
    --switch_mode smooth \
    --switch_ratio 0.4
```

🔹 Test the Model

```bash
python /content/MambaVesselNet/test.py
```

🔹 Evaluate the Model
```bash
python /content/MambaVesselNet/evaluation.py
```
