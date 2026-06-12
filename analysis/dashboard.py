"""Plotting + reconstruction utilities for the DECAL 3-panel physics dashboard.

Given trained `QuantileNet` ensembles, this module:
  1. evaluates the ensemble on a smooth energy grid to get the median surrogate
     and quantile bands (`get_ensemble_metrics`),
  2. builds SciPy interpolators on a fine grid for fast Brent-method inversion
     (`get_interpolators`),
  3. runs Neyman-construction reconstruction over a test energy set
     (`reco_metrics_over_grid`),
  4. produces the 3-panel dashboard: reconstructed linearity, reconstructed
     resolution vs E, stochastic resolution vs 1/sqrt(E) (`plot_dashboard`).

Used by analysis/verify_ensembles.py and the 03_ml_training_and_eval notebook.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# ---- ensemble evaluation -----------------------------------------------------

def get_ensemble_metrics(ensemble, x_max: float, y_frac_max: float,
                         e_grid: np.ndarray, device: torch.device,
                         is_linear_analog: bool = False):
    """Evaluate the ensemble on a smooth E grid; return (median, resolution).

    `is_linear_analog=True` uses a fixed slope (estimated between 10 and 50 GeV)
    instead of the local gradient — appropriate for the True Analog readout
    where the response should be perfectly linear at low E.
    """
    x_tensor = torch.tensor(e_grid / x_max, dtype=torch.float32,
                            device=device).unsqueeze(1)
    preds = []
    for m in ensemble:
        m.eval()
        with torch.no_grad():
            preds.append(m(x_tensor).cpu().numpy())
    avg = np.mean(preds, axis=0)
    preds_abs = avg * y_frac_max * e_grid[:, None]
    q_low, q_med, q_high = preds_abs[:, 0], preds_abs[:, 1], preds_abs[:, 2]
    sigma_y = (q_high - q_low) / 2.0

    if is_linear_analog:
        idx10 = np.argmin(np.abs(e_grid - 10.0))
        idx50 = np.argmin(np.abs(e_grid - 50.0))
        slope = np.full_like(e_grid, (q_med[idx50] - q_med[idx10]) / 40.0)
    else:
        slope = np.maximum(np.gradient(q_med, e_grid), 1e-6)

    resolution = sigma_y / slope / e_grid
    return q_med, resolution


# ---- Neyman-inversion reconstruction -----------------------------------------

def get_interpolators(ensemble, x_max: float, y_frac_max: float,
                      device: torch.device,
                      e_grid: np.ndarray = None):
    """Build (f_low, f_med, f_high) SciPy interpolators of the ensemble-averaged
    quantile curves vs true energy. Used as input to Brent inversion below.
    """
    if e_grid is None:
        e_grid = np.linspace(1, 500, 1000)
    x_tensor = torch.tensor(e_grid / x_max, dtype=torch.float32,
                            device=device).unsqueeze(1)
    preds = []
    for m in ensemble:
        m.eval()
        with torch.no_grad():
            preds.append(m(x_tensor).cpu().numpy())
    avg = np.mean(preds, axis=0)
    preds_abs = avg * y_frac_max * e_grid[:, None]
    f_low = interp1d(e_grid, preds_abs[:, 0], kind="linear", fill_value="extrapolate")
    f_med = interp1d(e_grid, preds_abs[:, 1], kind="linear", fill_value="extrapolate")
    f_high = interp1d(e_grid, preds_abs[:, 2], kind="linear", fill_value="extrapolate")
    return f_low, f_med, f_high


def invert_brent(y_obs: float, f_curve, lo: float = 5.0, hi: float = 450.0) -> float:
    """Find E such that f_curve(E) == y_obs, with graceful saturation handling.

    Brent's method requires the objective to change sign between lo and hi.
    When the surrogate saturates at high E, that may not be the case — we
    fall back to clipping to lo or hi as appropriate.
    """
    def objective(e):
        return float(f_curve(e)) - y_obs
    try:
        return brentq(objective, lo, hi)
    except ValueError:
        return hi if objective(hi) < 0 else lo


def reco_metrics_over_grid(f_low, f_med, f_high,
                           e_test: np.ndarray = None):
    """Run Neyman reconstruction over a grid of true energies.

    Returns (e_test, response_ratio, resolution). Response is E_reco/E_true
    (should be ~1 by construction). Resolution uses the Neyman crossover:
    upper E bound from inverting the *lower* signal quantile, and vice versa.
    """
    if e_test is None:
        e_test = np.linspace(10, 400, 100)
    response, resolution = [], []
    for e_true in e_test:
        y_obs = f_med(e_true)
        e_reco = invert_brent(y_obs, f_med)
        e_reco_hi = invert_brent(y_obs, f_low)   # Neyman crossover
        e_reco_lo = invert_brent(y_obs, f_high)  # Neyman crossover
        response.append(e_reco / e_true)
        resolution.append((e_reco_hi - e_reco_lo) / (2.0 * e_true))
    return e_test, np.array(response), np.array(resolution)


# ---- the 3-panel dashboard plot ----------------------------------------------

READOUT_COLORS = {
    "Analog":  "royalblue",
    "MIP":     "forestgreen",
    "Hits":    "crimson",
    "Cluster": "darkorchid",
    "Cluster (baseline)": "darkorchid",
    "Cluster (improved)": "teal",
}

READOUT_LABELS = {
    "Analog":  "True Analog",
    "MIP":     "MIP Proxy",
    "Hits":    "Raw Hits",
    "Cluster": "Naive 2D Clustering",
    "Cluster (baseline)": "Cluster (baseline)",
    "Cluster (improved)": "Cluster (improved)",
}


def plot_dashboard(reco_results, out_path_prefix=None, show=False):
    """Produce the 3-panel reconstruction dashboard.

    `reco_results` is a dict keyed by readout name (one of "Analog", "MIP",
    "Hits", "Cluster") mapping to (e_test, response, resolution) tuples.

    If `out_path_prefix` is given, saves to {prefix}_linearity.png and
    {prefix}_resolution.png. If `show` is True, also calls plt.show().
    """
    import matplotlib
    if out_path_prefix is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    e_test = next(iter(reco_results.values()))[0]

    # Panel 1: Reconstructed Linearity
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, (et, resp, _) in reco_results.items():
        ax.plot(et, resp, color=READOUT_COLORS.get(key, "gray"), lw=2, label=READOUT_LABELS.get(key, key))
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylim(0.8, 1.2)
    ax.set_title("Reconstructed Linearity ($E_{reco}/E_{true}$)", fontsize=13)
    ax.set_xlabel("True Beam Energy ($E_{true}$) [GeV]")
    ax.set_ylabel("Response Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path_prefix:
        plt.savefig(f"{out_path_prefix}_linearity.png", dpi=110)
    if show:
        plt.show()
    else:
        plt.close()

    # Panels 2 + 3: Resolution + Stochastic
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for key, (et, _, res) in reco_results.items():
        axes[0].plot(et, res, color=READOUT_COLORS.get(key, "gray"), lw=2, label=READOUT_LABELS.get(key, key))
    axes[0].set_title("Reconstructed Resolution ($\\sigma_{reco}/E_{true}$)", fontsize=13)
    axes[0].set_xlabel("True Beam Energy ($E_{true}$) [GeV]")
    axes[0].set_ylabel("Energy Resolution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key, (et, _, res) in reco_results.items():
        axes[1].plot(1.0 / np.sqrt(et), res, color=READOUT_COLORS.get(key, "gray"), lw=2,
                     label=READOUT_LABELS.get(key, key))
    axes[1].set_title("Stochastic Resolution ($\\sigma/E$ vs $1/\\sqrt{E}$)",
                      fontsize=13, pad=40)
    axes[1].set_xlabel("$1/\\sqrt{E_{true}}$ [GeV$^{-1/2}$]")
    axes[1].set_ylabel("Energy Resolution")

    def _to_e(x):
        return 1.0 / np.maximum(x, 1e-10) ** 2
    def _from_e(e):
        return 1.0 / np.sqrt(np.maximum(e, 1e-10))

    sec = axes[1].secondary_xaxis("top", functions=(_to_e, _from_e))
    sec.set_xlabel("True Beam Energy [GeV]")
    sec.set_xticks([400, 100, 50, 25, 10])
    sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path_prefix:
        plt.savefig(f"{out_path_prefix}_resolution.png", dpi=110)
    if show:
        plt.show()
    else:
        plt.close()
