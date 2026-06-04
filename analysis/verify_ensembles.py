"""Load trained ensembles and produce the 3-panel reconstruction dashboard.

Reads:
    $CALOMAPS_HOME/models/decal_extracted_data.npz
    $CALOMAPS_HOME/models/<ensemble_dir>/ens_{analog,mip,hits,cluster}.pth

Writes:
    $CALOMAPS_HOME/docs/figures/dashboard_linearity.png
    $CALOMAPS_HOME/docs/figures/dashboard_resolution.png

`ensemble_dir` defaults to "saved_ensembles_gpu_v2" but can be overridden
via the --ensemble-dir CLI arg.
"""
import sys, os, argparse

# Pick up cu121 torch if available (see analysis/train_ensembles.py)
_VENV = "/tmp/cu_torch_env/lib/python3.13/site-packages"
if os.path.isdir(_VENV):
    sys.path = [p for p in sys.path if "py-torch" not in p]
    sys.path.insert(0, _VENV)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from quantilenet import load_ensemble
from dashboard import (
    get_ensemble_metrics, get_interpolators, reco_metrics_over_grid,
    plot_dashboard,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", default="saved_ensembles_gpu_v2",
                        help="Subdir under $CALOMAPS_HOME/models/ holding ens_*.pth files")
    parser.add_argument("--show", action="store_true",
                        help="Also call plt.show() (useful when running inside a notebook).")
    args = parser.parse_args()

    calomaps_home = os.environ.get("CALOMAPS_HOME", os.path.expanduser("~/CALOMAPS"))
    npz_path = os.path.join(calomaps_home, "models", "decal_extracted_data.npz")
    ens_dir = os.path.join(calomaps_home, "models", args.ensemble_dir)
    fig_dir = os.path.join(calomaps_home, "docs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"torch:  {torch.__file__}  v{torch.__version__}  cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"data:   {npz_path}")
    print(f"models: {ens_dir}")
    print(f"figures: {fig_dir}")
    print()

    # ---- load data + valid mask ----------------------------------------------
    data = np.load(npz_path)
    all_truth   = data["all_truth"]
    all_visible = data["all_visible"]
    all_mip     = data["all_mip"]
    all_hits    = data["all_hits"]
    all_cluster = data["all_cluster"]
    valid = (all_hits > 0) & (all_truth > 0) & (all_visible > 0) & (all_mip > 0) & (all_cluster > 0)
    print(f"valid events: {valid.sum()}")

    # ---- load all four ensembles ---------------------------------------------
    print("loading ensembles...")
    ens_a, xa, ya = load_ensemble(os.path.join(ens_dir, "ens_analog.pth"), device)
    ens_m, xm, ym = load_ensemble(os.path.join(ens_dir, "ens_mip.pth"),     device)
    ens_h, xh, yh = load_ensemble(os.path.join(ens_dir, "ens_hits.pth"),    device)
    ens_c, xc, yc = load_ensemble(os.path.join(ens_dir, "ens_cluster.pth"), device)
    print(f"  loaded 4 ensembles, {len(ens_a)} models each")

    # ---- build interpolators + Neyman reconstruction over a grid -------------
    print("computing dashboard metrics...")
    fla, fma, fha = get_interpolators(ens_a, xa, ya, device)
    flm, fmm, fhm = get_interpolators(ens_m, xm, ym, device)
    flh, fmh, fhh = get_interpolators(ens_h, xh, yh, device)
    flc, fmc, fhc = get_interpolators(ens_c, xc, yc, device)

    reco = {
        "Analog":  reco_metrics_over_grid(fla, fma, fha),
        "MIP":     reco_metrics_over_grid(flm, fmm, fhm),
        "Hits":    reco_metrics_over_grid(flh, fmh, fhh),
        "Cluster": reco_metrics_over_grid(flc, fmc, fhc),
    }

    # ---- dashboard plots -----------------------------------------------------
    out_prefix = os.path.join(fig_dir, "dashboard")
    plot_dashboard(reco, out_path_prefix=out_prefix, show=args.show)
    print(f"saved {out_prefix}_linearity.png and {out_prefix}_resolution.png")

    # ---- headline numbers ----------------------------------------------------
    print()
    print("=== headline resolutions (sigma_reco / E_true) ===")
    et = reco["Analog"][0]
    for energy in (10, 100, 300):
        idx = np.argmin(np.abs(et - energy))
        a = reco["Analog"][2][idx]
        m = reco["MIP"][2][idx]
        h = reco["Hits"][2][idx]
        c = reco["Cluster"][2][idx]
        print(f"  E={energy:>3d} GeV:  analog={a:.4f}  MIP={m:.4f}  hits={h:.4f}  cluster={c:.4f}")


if __name__ == "__main__":
    main()
