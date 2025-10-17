# RSNA_2025-MambaVesselNet
RSNA Intracranial Aneurysm Detection using MambaVesselNet++ architecture.

✨ Description

Medical image segmentation is essential for accurate computer-aided diagnosis. While Vision Transformers have achieved great success, their quadratic self-attention mechanism leads to high computational costs. The recently proposed Mamba state space model provides an efficient alternative for capturing long-range dependencies with reduced memory usage.

In this work, we apply the MambaVesselNet++ architecture — a hybrid CNN-Mamba framework — to the RSNA Intracranial Aneurysm Detection 2025
dataset using MRA scans. We adapt the original model to perform binary vessel segmentation, converting the original multi-class labels into vessel (1) and background (0).

Our customized RSNA-MambaVesselNet++ effectively models both local texture details and long-range spatial dependencies, achieving robust vessel segmentation performance suitable for clinical aneurysm

✨ Highlights

🚀 Adapted MambaVesselNet++ for RSNA Intracranial Aneurysm Detection (MRA) data.

🧩 Converted multi-class segmentation masks into binary vessel vs. background for improved clarity.

⚙️ Combined CNN-based texture extraction with Mamba-based long-range modeling.

🩺 Achieved accurate and efficient vessel segmentation tailored for aneurysm detection tasks.
%%%%%

📊 Ready-to-use training and inference scripts for easy experimentation.

# 1️⃣ Clone your repository
!git clone https://github.com/shimaaelbana/RSNA_2025-MambaVesselNet.git
%cd RSNA_2025-MambaVesselNet

# 2️⃣ Install the correct PyTorch version (CUDA 12.4 support)
!pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3️⃣ Fix Ninja installation (for faster CUDA ops)
!pip uninstall -y ninja
!pip install ninja --no-cache-dir --force-reinstall

# 4️⃣ Install dependencies for Mamba model
%cd causal-conv1d
!python setup.py install
%cd ../mamba
!python setup.py install

# 5️⃣ Install required libraries
!pip install loguru monai nibabel tqdm scikit-image

# 6️⃣ (Optional) Go back to main project folder
%cd ..

