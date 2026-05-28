#!/bin/bash
# One-shot GPU kernel setup for notebook 03 (the DECAL ML / dashboard notebook).
#
# The CVMFS Key4hep stack ships a CPU-only PyTorch, so notebook 03 needs a venv
# with a CUDA build. This script builds a *self-contained* venv (CUDA torch +
# the few packages nb03 imports) and registers a "Key4hep + GPU" Jupyter kernel
# whose launcher uses a clean PYTHONPATH, so the venv's cu121 torch always wins
# (no CVMFS shadowing). nb00 and nb02 don't need this — they use "Python (Key4hep)".
#
# Usage (from a JupyterLab terminal, after `source ~/setup_calomaps.sh`):
#     bash setup/setup_gpu_kernel.sh
#
# By default the venv goes in /tmp (large, fast, but wiped when the EAF container
# restarts — just re-run this script then). For a persistent install set
# CALOMAPS_GPU_ENV to a home path with ~5 GB free, e.g.:
#     CALOMAPS_GPU_ENV=$HOME/calomaps_gpu_env bash setup/setup_gpu_kernel.sh
set -e

VENV="${CALOMAPS_GPU_ENV:-/tmp/calomaps_gpu_env}"
PYBIN=/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/python/3.13.8-z2dydk/bin/python3.13
KDIR="$HOME/.local/share/jupyter/kernels/calomaps_gpu"

echo "[1/4] Creating venv at $VENV"
"$PYBIN" -m venv --system-site-packages "$VENV"

# Install with a CLEAN PYTHONPATH so packages land in the venv (not shadowed by
# CVMFS) and torch's deps resolve to the venv copies.
echo "[2/4] Installing numpy / scipy / matplotlib / ipykernel"
TMPDIR=/tmp env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir --ignore-installed \
    numpy scipy matplotlib ipykernel
echo "[3/4] Installing CUDA (cu121) PyTorch (~4.4 GB)"
TMPDIR=/tmp env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

echo "[4/4] Registering the 'Key4hep + GPU' kernel"
mkdir -p "$KDIR"
cat > "$KDIR/wrapper.sh" <<EOF
#!/bin/bash
# clean PYTHONPATH -> the venv's cu121 torch wins over CVMFS's CPU torch
unset PYTHONPATH PYTHONHOME
exec "$VENV/bin/python" -m ipykernel_launcher "\$@"
EOF
chmod +x "$KDIR/wrapper.sh"
cat > "$KDIR/kernel.json" <<EOF
{
 "argv": ["$KDIR/wrapper.sh", "-f", "{connection_file}"],
 "display_name": "Key4hep + GPU",
 "language": "python"
}
EOF

echo
echo "Verifying:"
env -u PYTHONPATH "$VENV/bin/python" -c \
    "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo
echo "Done. In JupyterLab, open notebooks/03_ml_training_and_eval.ipynb and pick"
echo "the 'Key4hep + GPU' kernel. (nb00 / nb02 use 'Python (Key4hep)'.)"
