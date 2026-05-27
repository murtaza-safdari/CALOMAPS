"""CLI entry point: train 4 Deep Quantile Ensembles on GPU.

This is the headless / batch version. For interactive notebook use, prefer
notebooks/03_ml_training_and_eval.ipynb — it imports `train_one_ensemble`
from `quantilenet.py` and shows the training loop in the notebook UI.

This script does:
  reads $CALOMAPS_HOME/models/decal_extracted_data.npz
  trains 20 networks per readout (80 total)
  saves to $CALOMAPS_HOME/models/saved_ensembles_gpu_v2/

GPU prerequisite: see handbook.md §11.2 for the cu121 torch install.
The sys.path shim below picks up /tmp/cu_torch_env when it exists (Path B).

Usage from a terminal:
    /tmp/cu_torch_env/bin/python $CALOMAPS_HOME/analysis/train_ensembles.py
"""
import sys, os

# Path B sys.path shim: prefer cu121 torch in /tmp/cu_torch_env over CVMFS
_VENV = "/tmp/cu_torch_env/lib/python3.13/site-packages"
if os.path.isdir(_VENV):
    sys.path = [p for p in sys.path if "py-torch" not in p]
    sys.path.insert(0, _VENV)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from quantilenet import train_one_ensemble, save_ensemble


CALOMAPS_HOME = os.environ.get("CALOMAPS_HOME", os.path.expanduser("~/CALOMAPS"))
NPZ = os.path.join(CALOMAPS_HOME, "models", "decal_extracted_data.npz")
OUT_DIR = os.path.join(CALOMAPS_HOME, "models", "saved_ensembles_gpu_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# ---- env print ---------------------------------------------------------------
print(f"torch:          {torch.__file__}")
print(f"version:        {torch.__version__}")
print(f"cuda built:     {torch.backends.cuda.is_built()}")
print(f"cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:         {device}")
if torch.cuda.is_available():
    print(f"device name:    {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU memory:     {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")
print()


# ---- load data ---------------------------------------------------------------
print(f"loading {NPZ}")
data = np.load(NPZ)
all_truth   = data["all_truth"]
all_visible = data["all_visible"]
all_mip     = data["all_mip"]
all_hits    = data["all_hits"]
all_cluster = data["all_cluster"]
valid = (all_hits > 0) & (all_truth > 0) & (all_visible > 0) & (all_mip > 0) & (all_cluster > 0)
x_train = all_truth[valid]
print(f"valid events: {valid.sum()} of {len(valid)}\n")


# ---- train all 4 readouts ----------------------------------------------------
import time
t_grand = time.time()
for label, y_arr, seed, fname in [
    ("True Analog",      all_visible, 1000, "ens_analog.pth"),
    ("MIP Proxy",        all_mip,     2000, "ens_mip.pth"),
    ("Raw Hits",         all_hits,    3000, "ens_hits.pth"),
    ("Naive Clustering", all_cluster, 4000, "ens_cluster.pth"),
]:
    ens, xm, ym = train_one_ensemble(x_train, y_arr[valid], device,
                                     name=label, seed_base=seed)
    fp = os.path.join(OUT_DIR, fname)
    save_ensemble(ens, xm, ym, fp)
    print(f"  saved -> {fp} ({os.path.getsize(fp)/1024:.0f} KB)\n")

print(f"=== ALL DONE in {time.time()-t_grand:.1f}s total ===")
print(f"models in: {OUT_DIR}")
