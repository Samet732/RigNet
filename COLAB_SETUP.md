Colab setup notes for RigNet

This document explains how to set up a Colab runtime for this repository. Colab environment details (CUDA and default Python) change over time, so the recommended approach is:

1) Set the Colab runtime to "GPU" (Runtime -> Change runtime type -> GPU).
2) Run the following cells to install system and Python packages. The key idea: install a compatible torch first, then install PyTorch Geometric (PyG) wheels that match your torch+CUDA.

---
# Minimal Colab install (example, run in a notebook cell)

# 1) Install lightweight system libs required by some Python packages
!apt-get update -y && apt-get install -y libspatialindex-dev

# 2) Install basic Python packages from requirements.txt (excluding torch & pyg)
!pip install -r requirements.txt

# 3) Install torch + torchvision. Choose the right CUDA wheel for the runtime.
#    If you want to use the prebuilt Colab CUDA, a common recent option is cu118/cu121.
#    Example (may need to be adapted depending on the Colab runtime):
!pip install --quiet --upgrade "torch" "torchvision" --index-url https://download.pytorch.org/whl/cu118

# 4) Verify torch and CUDA
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available(), 'cuda version', torch.version.cuda)
PY

# 5) Install PyTorch Geometric (PyG) and its dependency wheels matching your installed torch
#    Replace <TORCH_VERSION> and +cuXXX with the printed torch version and CUDA tag.
#    Example command (adjust the "torch-<TORCH_VERSION>+cuXXX.html" to the correct one):
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<TORCH_VERSION>+cuXXX.html
!pip install torch-geometric

# If you cannot find matching wheels, you can install CPU-only torch and the CPU PyG wheels or use a different runtime.

---

Notes and tips

- open3d: The package can be large. If you run out of disk space, consider running a CPU runtime or skipping visualization parts.
- rtree: requires libspatialindex (we install it above). If you hit wheel issues on Colab, try building from source or skip functionality that depends on rtree.
- If you prefer, create a small Colab cell with helper logic that detects torch.__version__ and suggests the exact data.pyg.org wheel URL.

Example helper snippet to print the exact PyG wheel URL to use:

python - <<'PY'
import torch
v = torch.__version__.split('+')[0]
cuda = torch.version.cuda or 'cpu'
print('Detected torch', v, 'CUDA', cuda)
print('Use wheels from https://data.pyg.org/whl/torch-{}+cu{}.html'.format(v, cuda.replace('.', '')))
PY

If you want, I can add a small Colab notebook that automates these steps (detects torch version and runs the proper pip commands). Ask and I'll create it.
