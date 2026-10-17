"""Drawn schematics for the notebooks of the reticle study (09 to 12): matplotlib only, no data files.

Each function draws on the axes `ax` it is given, or on a figure of its own when `ax` is None, and
returns the figure. A number in a drawing comes from the code's constants (deposit_jobs: WINDOW_X,
WINDOW_Y, GUARD, TILE_X, TILE_Y, EV_PER_PAIR, N_LAYERS; sensor_crossings, as deposit_jobs.sc:
N_FACES), or from the arguments for what the loaded files define (pixel pitches, thickness,
threshold). One style: thin black lines, a few muted colours, labels in plain words.

    chain(current)                                  the stages of the study; `current` ('09' to '12') is marked
    detector(pitch_v_um, pitch_u_um, thickness_um)  the barrel end-on, one module opened, one sensor plane
    slab_crossing(pitch_v_um, pitch_u_um, thickness_um, threshold_e=None)
                                                    one track through the silicon: pairs, drift, lit pixels
    tile_window(pitch_v_um=None, pitch_u_um=None)   a tile in its window; a group cut into two jobs
    group_roots()                                   nodes, parents, roots and groups in one slab
    reticle_edge_logic(nv, nu, pitch_v_um=None, pitch_u_um=None)
                                                    a plane cut into reticles; what the edge logic sees
    angle_geometry(pitch_v_um, pitch_u_um, thickness_um)
                                                    how many pixels a slanted track spans; an L and a line
    estimator_ladder()                              one hit map, four estimators, four counts
    five_blocks(n_showers=200, n_blocks=5)          fit on four blocks of showers, score on the fifth
    table_pull()                                    why a table scored per bin of the truth pulls toward the middle
    layer_weights(n_thin, thin_mm, n_thick, thick_mm)
                                                    the layers behind thin and thick plates and their weights
"""
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)                # deposit_jobs, imported inside the functions that need it

INK, GREY, LIGHT = 'k', '0.5', '0.9'
BLUE, LBLUE = '#4C72B0', '#CAD7EB'
ORANGE, LORANGE = '#DD8452', '#F5D9C4'
RED, LRED = '#C44E52', '#F0CFD0'
GREEN, PURPLE = '#55A868', '#8172B3'
LW, FS, SMALL = 0.9, 10, 8.5
WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
ORDINALS = {2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth',
            10: 'tenth'}
_erf = np.vectorize(math.erf)


def _axes(ax, figsize, equal=True):
    # the axes to draw on, and its figure; axis lines off
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.set_axis_off()
    if equal:
        ax.set_aspect('equal')
    return fig, ax


def _box(ax, x, y, w, h, text='', fc='white', ec=INK, lw=LW, fs=FS, weight='normal', ls='-', rs=None, color=INK):
    # a box with rounded corners and centred text
    rs = 0.12 * min(w, h) if rs is None else rs
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rs}', fc=fc, ec=ec, lw=lw,
                                ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight=weight, color=color)


def _arrow(ax, p0, p1, color=INK, lw=LW, style='-|>', ms=10, ls='-'):
    ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                                      mutation_scale=ms, linestyle=ls))


def _note(ax, text, xy, xytext, fs=SMALL, color=INK, ha='left', va='center'):
    # a label with a thin line to the point it names
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=color, ha=ha, va=va,
                arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6, shrinkA=2, shrinkB=1))


def _dim(ax, p0, p1, text, offset, fs=SMALL, ha='center', va='center'):
    # a dimension line with arrows at both ends and its text beside it
    _arrow(ax, p0, p1, style='<|-|>', ms=7, lw=0.7)
    ax.text((p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1], text, fontsize=fs, ha=ha, va=va)


def _grid(ax, x0, y0, nx, ny, w, h, color='0.8', lw=0.5):
    for i in range(nx + 1):
        ax.plot([x0 + i * w] * 2, [y0, y0 + ny * h], color=color, lw=lw)
    for j in range(ny + 1):
        ax.plot([x0, x0 + nx * w], [y0 + j * h] * 2, color=color, lw=lw)


def _cross_out(ax, x, y, w, h, text, fs=SMALL):
    # a box with its text, struck through in red: what is not seen
    _box(ax, x, y, w, h, text, fc='white', ec=GREY, fs=fs, color='0.35')
    ax.plot([x + 0.04 * w, x + 0.96 * w], [y + 0.15 * h, y + 0.85 * h], color=RED, lw=1.4)


def _wiggle(p0, p1, n_waves=4, amp=0.08, n=80):
    # a wavy line from p0 to p1 (a photon), amplitude as a share of the length
    t = np.linspace(0, 1, n)
    d = np.subtract(p1, p0)
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    off = amp * np.hypot(*d) * np.sin(2 * np.pi * n_waves * t)
    return p0[0] + d[0] * t + nrm[0] * off, p0[1] + d[1] * t + nrm[1] * off


def _hit_bits(nv, nu):
    # a fixed, made-up pattern of hit pixels in an nv x nu reticle: a few small clusters (fractions of the sides)
    cells = [(0.14, 0.30), (0.19, 0.30), (0.19, 0.38), (0.45, 0.70), (0.72, 0.18), (0.72, 0.26), (0.77, 0.26),
             (0.33, 0.85), (0.90, 0.62), (0.90, 0.70), (0.95, 0.70), (0.55, 0.40)]
    bits = np.zeros((nu, nv), bool)
    for fv, fu in cells:
        bits[min(int(fu * nu), nu - 1), min(int(fv * nv), nv - 1)] = True
    return bits


# ---------------------------------------------------------------------------------------------
def chain(current, ax=None):
    """The chain of the study as boxes: what each stage holds, the files under it, the program that
    makes it above each arrow, the notebooks over the boxes; the boxes of notebook `current` filled."""
    books = {'09': (0, 1), '10': (2, 2), '11': (3, 3), '12': (4, 4)}
    if current not in books:
        raise ValueError(f"current must be one of {sorted(books)}, not {current!r}")
    holds = ["Geant4 steps\nin the silicon", "the charge of\nevery crossing\non every pixel", "one plane's\nhit map",
             "the count of\neach reticle", "the layer profile\nand the shower total"]
    files = ["steps_b0_evt2.npz", "jobs_b0.npz\ncharges_b0.npz", "ret_K1.npz ... ret_K8.npz", "", ""]
    makers = ["Geant4,\nstep_subset.py", "deposit_jobs.py,\nPIXELAV,\ncharge_maps.py", "the threshold,\nreticles.py",
              "mip_counting.py", "the sum over\nthe reticles"]
    W, H, G = 2.3, 1.2, 1.75
    fig, ax = _axes(ax, (13, 2.4))
    lo, hi = books[current]
    for i in range(5):
        x = G + i * (W + G)
        mine = lo <= i <= hi
        _box(ax, x, 0, W, H, holds[i], fc=LBLUE if mine else 'white', lw=1.3 if mine else LW, fs=9.5)
        _arrow(ax, (x - G + 0.12, H / 2), (x - 0.06, H / 2))
        ax.text(x - G / 2 + 0.03, H / 2 + 0.1, makers[i], ha='center', va='bottom', fontsize=8, color='0.25')
        if files[i]:
            ax.text(x + W / 2, -0.12, files[i], ha='center', va='top', fontsize=8, family='monospace', color='0.3')
    for nb, (a, b) in books.items():
        x0, x1 = G + a * (W + G), G + b * (W + G) + W
        mine = nb == current
        ax.plot([x0 + 0.05, x0 + 0.05, x1 - 0.05, x1 - 0.05], [H + 0.18, H + 0.3, H + 0.3, H + 0.18],
                color=BLUE if mine else GREY, lw=1.3 if mine else 0.8)
        ax.text((x0 + x1) / 2, H + 0.38, f"notebook {nb}" + (": this one" if mine else ""), ha='center', va='bottom',
                fontsize=9.5, fontweight='bold' if mine else 'normal', color=INK if mine else '0.3')
    ax.set_xlim(0, 5 * (W + G) + 0.05)
    ax.set_ylim(-0.75, H + 0.75)
    return fig


# ---------------------------------------------------------------------------------------------
def detector(pitch_v_um, pitch_u_um, thickness_um, ax=None):
    """The barrel end-on (N_FACES flat modules around the beam line, the shower axis hitting the axis
    module at right angles), one module opened into its N_LAYERS layers (not to scale), and one sensor
    plane with its v and u axes and one pixel to scale."""
    import deposit_jobs as dj
    n_faces, n_layers = dj.sc.N_FACES, dj.N_LAYERS
    fig, ax = _axes(ax, (13, 4.9))

    # (a) the barrel end-on; +y up, the face whose outward normal points along +y is the axis module
    half, r_in, r_out = np.pi / n_faces, 1.0, 1.45
    phis = np.radians(dj.sc.FACE_PHI0_DEG + np.arange(n_faces) * 360.0 / n_faces)
    k_axis = int(np.argmin(np.abs(np.angle(np.exp(1j * (phis - np.pi / 2))))))
    c = 1 / np.cos(half)
    for k, phi in enumerate(phis):
        corners = [(r * c * np.cos(a), r * c * np.sin(a)) for r, a in
                   ((r_in, phi - half), (r_out, phi - half), (r_out, phi + half), (r_in, phi + half))]
        ax.add_patch(Polygon(corners, closed=True, fc=LBLUE if k == k_axis else LIGHT, ec=INK, lw=LW))
    ax.add_patch(Circle((0, 0), 0.07, fc='white', ec=INK, lw=LW))
    ax.plot(0, 0, '.', color=INK, ms=3)
    ax.text(0, -0.16, "beam line\n(along v, out of the page)", ha='center', va='top', fontsize=SMALL)
    _arrow(ax, (0, 0.1), (0, r_in), color=ORANGE, lw=1.8, ms=12)
    ax.text(0.07, 0.5, "shower\naxis", color=ORANGE, fontsize=SMALL, va='center')
    ax.text(0, r_out + 0.08, "the axis module", ha='center', va='bottom', fontsize=SMALL)
    ax.text(0, -r_out * c - 0.15, f"the barrel end-on:\n{n_faces} flat modules", ha='center', va='top', fontsize=FS)

    # (b) one module opened: its layers, inner side down (not to scale)
    xb, wb, yb, hb = 2.75, 1.25, -1.45, 2.9
    hl = hb / n_layers
    for i in range(n_layers):
        y = yb + i * hl
        ax.add_patch(Rectangle((xb, y), wb, 0.62 * hl, fc='0.62', ec='none'))           # tungsten
        ax.add_patch(Rectangle((xb, y + 0.62 * hl), wb, 0.38 * hl, fc=BLUE, ec='none'))  # silicon
    ax.add_patch(Rectangle((xb, yb), wb, hb, fill=False, ec=INK, lw=LW))
    tx = r_out * np.tan(half)
    for p0, p1 in (((tx * r_in / r_out, r_in), (xb, yb)), ((tx, r_out), (xb, yb + hb))):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREY, lw=0.6, ls='--')
    _arrow(ax, (xb + wb / 2, yb - 0.45), (xb + wb / 2, yb), color=ORANGE, lw=1.8, ms=12)
    ax.text(xb + wb / 2 + 0.07, yb - 0.32, "shower axis", color=ORANGE, fontsize=SMALL, va='center')
    _arrow(ax, (xb - 0.22, yb), (xb - 0.22, yb + 0.9), lw=0.8, ms=8)
    ax.text(xb - 0.3, yb + 0.45, "w, outward", rotation=90, ha='right', va='center', fontsize=SMALL)
    i_mid = n_layers // 2
    _note(ax, "tungsten plate", (xb + wb, yb + (i_mid + 0.3) * hl), (xb + wb + 0.25, yb + (i_mid - 2.5) * hl))
    _note(ax, "silicon sensor", (xb + wb, yb + (i_mid + 0.8) * hl), (xb + wb + 0.25, yb + (i_mid + 3.5) * hl))
    ax.text(xb + wb + 0.08, yb + 0.5 * hl, "layer 0", fontsize=SMALL, va='center')
    ax.text(xb + wb + 0.08, yb + (n_layers - 0.5) * hl, f"layer {n_layers - 1}", fontsize=SMALL, va='center')
    ax.text(xb + wb / 2, yb + hb + 0.12, f"one module opened:\n{n_layers} layers, tungsten and silicon\n"
            "(not to scale)", ha='center', va='bottom', fontsize=FS)

    # (c) one sensor plane seen from outside, v to the right and u up, and one pixel to scale
    xc, yc, wc, hc = 6.45, -0.55, 2.3, 1.45
    y_si = yb + (n_layers - 3 + 0.8) * hl
    for p1 in ((xc, yc), (xc, yc + hc)):
        ax.plot([xb + wb, p1[0]], [y_si, p1[1]], color=GREY, lw=0.6, ls='--')
    ax.add_patch(Rectangle((xc, yc), wc, hc, fc=LBLUE, ec=INK, lw=LW))
    _grid(ax, xc, yc, 16, 10, wc / 16, hc / 10, color='white', lw=0.6)
    _arrow(ax, (xc, yc - 0.18), (xc + 0.9, yc - 0.18), lw=1.0)
    ax.text(xc + 0.95, yc - 0.18, "v, along the barrel axis", fontsize=SMALL, va='center')
    _arrow(ax, (xc - 0.18, yc), (xc - 0.18, yc + 0.9), lw=1.0)
    ax.text(xc - 0.26, yc + 0.45, "u, across the module", rotation=90, ha='right', va='center', fontsize=SMALL)
    ax.text(xc + wc / 2, yc + hc + 0.12, "one sensor plane:\none layer of one module", ha='center', va='bottom',
            fontsize=FS)
    s = 1.3 / (4 * pitch_v_um)                       # drawing units per um for the pixels to scale
    xp, yp = xc + wc + 0.8, yc + 0.05
    nv, nu = 4, 5
    pw, ph = pitch_v_um * s, pitch_u_um * s
    zoom = Rectangle((xc + 0.62 * wc, yc + 0.5 * hc), 0.12, 0.09, fill=False, ec=INK, lw=0.8)
    ax.add_patch(zoom)
    for p1 in ((xp, yp), (xp, yp + nu * ph)):
        ax.plot([xc + 0.62 * wc + 0.12, p1[0]], [yc + 0.5 * hc + 0.045, p1[1]], color=GREY, lw=0.6, ls='--')
    _grid(ax, xp, yp, nv, nu, pw, ph, color=INK, lw=0.6)
    ax.add_patch(Rectangle((xp + pw, yp + 2 * ph), pw, ph, fc=BLUE, ec=INK, lw=0.8))
    _dim(ax, (xp + pw, yp - 0.12), (xp + 2 * pw, yp - 0.12), f"{pitch_v_um:g} um", (0, -0.13))
    _dim(ax, (xp + nv * pw + 0.12, yp + 2 * ph), (xp + nv * pw + 0.12, yp + 3 * ph), f"{pitch_u_um:g} um", (0.08, 0),
         ha='left')
    xs = xp + nv * pw + 0.95                           # the same pixel in section, to the same scale
    ax.add_patch(Rectangle((xs, yp), pw, thickness_um * s, fc=BLUE, ec=INK, lw=0.8))
    _dim(ax, (xs + pw + 0.1, yp), (xs + pw + 0.1, yp + thickness_um * s), f"{thickness_um:g} um\nthick", (0.08, 0),
         ha='left')
    ax.text(xs + pw / 2, yp - 0.12, "in section", ha='center', va='top', fontsize=SMALL)
    ax.text((xp + xs + pw) / 2, yp + max(nu * ph, thickness_um * s) + 0.12, "one pixel, to scale", ha='center',
            va='bottom', fontsize=FS)
    ax.set_xlim(-r_out * c - 0.1, xs + pw + 0.9)
    ax.set_ylim(-r_out * c - 0.75, r_out + 0.75)
    return fig


# ---------------------------------------------------------------------------------------------
def slab_crossing(pitch_v_um, pitch_u_um, thickness_um, threshold_e=None, ax=None):
    """One charged track through the silicon, in section along v: the electron-hole pairs along it and
    along one delta ray, the electrons drifting to the pixel side (the outer face, z = thickness),
    widening on the way, and the pixels whose charge passes the threshold. The charges drawn are
    made up; only the pitch, the thickness and the threshold are the sample's."""
    import deposit_jobs as dj
    p, t = float(pitch_v_um), float(thickness_um)
    n_pix = 7
    fig, ax = _axes(ax, (12, 5.0))
    ax.add_patch(Rectangle((0, 0), n_pix * p, t, fc='#EEF2F8', ec=INK, lw=LW))
    for i in range(1, n_pix):
        ax.plot([i * p] * 2, [0, t], color='0.75', lw=0.5, ls=':')
    # the track, from the inner face (z = 0) to the outer face, and one delta ray
    v_in, v_out = 2.3 * p, 4.0 * p
    track = lambda f: (v_in + f * (v_out - v_in), f * t)
    _arrow(ax, (v_in - 0.3 * (v_out - v_in), -0.3 * t), (v_in - 0.08 * (v_out - v_in), -0.08 * t), lw=1.4, ms=11)
    ax.plot([v_in - 0.1 * (v_out - v_in), v_out], [-0.1 * t, t], color=INK, lw=1.4)
    d0 = np.array(track(0.3))
    f = np.linspace(0, 1, 30)
    dv = d0[0] + 1.5 * p * f + 0.12 * p * np.sin(3 * np.pi * f)          # a curling delta ray, made up
    dz = d0[1] - 0.12 * t * np.sin(np.pi * f) + 0.18 * t * f
    ax.plot(dv, dz, color=PURPLE, lw=1.2)
    # electron-hole pairs, the drift of the electrons up to the pixels, and its widening
    u_dir = np.array([v_out - v_in, t]) / np.hypot(v_out - v_in, t)
    nrm = np.array([-u_dir[1], u_dir[0]])
    pairs = [np.array(track(g)) for g in np.linspace(0.07, 0.93, 9)] + [np.array((dv[i], dz[i])) for i in (10, 18, 27)]
    width = lambda z: 0.42 * p * np.sqrt(max(t - z, 0) / t)              # diffusion: grows with the drift length
    for k, q in enumerate(pairs):
        if k in (1, 4, 7, 11):
            w = width(q[1])
            ax.add_patch(Polygon([(q[0], q[1]), (q[0] - w, t), (q[0] + w, t)], closed=True, fc=LBLUE, ec='none',
                                 alpha=0.8, zorder=1))
            _arrow(ax, (q[0], q[1] + 0.03 * t), (q[0], t - 0.01 * t), color=BLUE, lw=0.7, ms=6)
        ax.plot(*(q + 0.035 * t * nrm), 'o', ms=3.6, color=BLUE, zorder=3)
        ax.plot(*(q - 0.035 * t * nrm), 'o', ms=3.6, mfc='white', mec=RED, mew=0.9, zorder=3)
    # the charge each pixel collects (made up from the drawing) and the threshold
    samples = [np.array(track(g)) for g in np.linspace(0, 1, 200)] + [np.array((a, b)) for a, b in zip(dv, dz)] * 2
    edges = np.arange(n_pix + 1) * p
    q_pix = np.zeros(n_pix)
    for v0, z0 in samples:
        sig = max(0.5 * width(z0), 1e-3 * p)
        cdf = 0.5 * (1 + _erf((edges - v0) / (sig * np.sqrt(2))))
        q_pix += np.diff(cdf)
    q_pix /= q_pix.max()
    cut = 0.22
    base, hbar = 1.2 * t, 0.55 * t
    for i in range(n_pix):
        lit = q_pix[i] >= cut
        ax.add_patch(Rectangle((i * p + 0.07 * p, t), 0.86 * p, 0.09 * t, fc=INK if lit else 'white', ec=INK, lw=0.7))
        ax.add_patch(Rectangle((i * p + 0.25 * p, base), 0.5 * p, hbar * q_pix[i], fc=BLUE if lit else LIGHT,
                               ec=INK if lit else GREY, lw=0.6))
    ax.plot([0, n_pix * p], [base + cut * hbar] * 2, color=RED, lw=0.9, ls='--')
    thr = "threshold" if threshold_e is None else f"threshold, {threshold_e:,.0f} e"
    ax.text(n_pix * p + 0.08 * p, base + cut * hbar, thr, color=RED, fontsize=SMALL, va='center')
    ax.text(-0.08 * p, base + 0.5 * hbar, "charge collected\nby each pixel", ha='right', va='center', fontsize=SMALL)
    ax.text(-0.08 * p, t + 0.045 * t, "pixels: the dark ones\npass the threshold", ha='right', va='center',
            fontsize=SMALL)
    # labels
    ax.text(n_pix * p + 0.08 * p, t, "outer face: the pixel side", fontsize=SMALL, va='center')
    ax.text(n_pix * p + 0.08 * p, 0, "inner face", fontsize=SMALL, va='center')
    _dim(ax, (-0.12 * p, 0.06 * t), (-0.12 * p, 0.94 * t), "", (0, 0))
    ax.text(-0.22 * p, 0.5 * t, f"{t:g} um of\nsilicon", fontsize=SMALL, ha='right', va='center')
    _dim(ax, (6 * p, -0.1 * t), (7 * p, -0.1 * t), f"{p:g} um", (0, -0.06 * t), ha='center', va='top')
    ax.text(v_in - 0.32 * (v_out - v_in), -0.33 * t, "a charged particle", ha='right', va='top', fontsize=SMALL)
    _note(ax, "a delta ray", (dv[-1], dz[-1]), (n_pix * p + 0.25 * p, 0.25 * t), ha='left')
    _note(ax, "electrons drift to the pixels,\nwidening on the way (diffusion)",
          (pairs[7][0] + 0.3 * p, 0.86 * t), (n_pix * p + 0.25 * p, 0.7 * t), ha='left')
    ax.plot([], [], 'o', ms=4, color=BLUE, label='electron')
    ax.plot([], [], 'o', ms=4, mfc='white', mec=RED, label='hole')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.28), fontsize=SMALL, frameon=False, handletextpad=0.2,
              title=f"electron-hole pairs along the track:\non average one per {dj.EV_PER_PAIR:g} eV deposited",
              title_fontsize=SMALL, alignment='left')
    ax.text(n_pix * p / 2, -0.62 * t, f"in section along v; the pixels are {p:g} um along v and {pitch_u_um:g} um "
            "along u; the charges are drawn, not computed", ha='center', va='top', fontsize=SMALL, color='0.3')
    ax.set_xlim(-1.9 * p, n_pix * p + 2.9 * p)
    ax.set_ylim(-0.8 * t, base + hbar + 0.1 * t)
    return fig


# ---------------------------------------------------------------------------------------------
def tile_window(pitch_v_um=None, pitch_u_um=None, ax=None):
    """Left: the TILE_X x TILE_Y tile inside its WINDOW_X x WINDOW_Y window, a guard band of GUARD pixels
    on every side. Right: a group whose track crosses a tile boundary, cut into two jobs, each with its
    own window; the two windows overlap. Pixels to their true shape when the pitches are given."""
    import deposit_jobs as dj
    WX, WY, G, TX, TY = dj.WINDOW_X, dj.WINDOW_Y, dj.GUARD, dj.TILE_X, dj.TILE_Y
    pv = 1.0 if pitch_v_um is None else float(pitch_v_um)
    pu = 1.0 if pitch_u_um is None else float(pitch_u_um)
    fig, ax = _axes(ax, (13, 3.9))
    # (a) one window, centred in height on the two windows of (b)
    ya = 0.5 * TY * pu
    ax.add_patch(Rectangle((0, ya), WX * pv, WY * pu, fc=LORANGE, ec='none'))
    ax.add_patch(Rectangle((G * pv, ya + G * pu), TX * pv, TY * pu, fc=LBLUE, ec='none'))
    _grid(ax, 0, ya, WX, WY, pv, pu, color='white', lw=0.7)
    ax.add_patch(Rectangle((G * pv, ya + G * pu), TX * pv, TY * pu, fill=False, ec=BLUE, lw=1.3))
    ax.add_patch(Rectangle((0, ya), WX * pv, WY * pu, fill=False, ec=INK, lw=1.0, ls='--'))
    ax.text(WX * pv / 2, ya + WY * pu / 2, f"tile\n{TX} x {TY} pixels", ha='center', va='center', fontsize=FS)
    _note(ax, f"guard band:\n{G} pixels on\nevery side", (WX * pv - 0.5 * G * pv, ya + 0.3 * WY * pu),
          (WX * pv + 1.2 * pv, ya + 0.05 * WY * pu))
    ax.text(WX * pv / 2, ya + WY * pu + 0.6 * pu, f"a window: {WX} x {WY} pixels,\nthe tile and its guard band",
            ha='center', va='bottom', fontsize=FS)
    ax.text(WX * pv / 2, ya - 0.6 * pu, "PIXELAV computes the charge of one job on its window", ha='center', va='top',
            fontsize=SMALL)
    # (b) two tiles, one above the other along u, and a group crossing from one into the other
    x0 = (WX + 9) * pv
    for k, (col, lcol) in enumerate(((BLUE, LBLUE), (RED, LRED))):
        ax.add_patch(Rectangle((x0 + G * pv, (G + k * TY) * pu), TX * pv, TY * pu, fc=lcol, ec='none', alpha=0.8))
        ax.add_patch(Rectangle((x0, k * TY * pu), WX * pv, WY * pu, fill=False, ec=col, lw=1.1, ls='--'))
    _grid(ax, x0 + G * pv, G * pu, TX, 2 * TY, pv, pu, color='white', lw=0.5)
    ax.add_patch(Rectangle((x0, TY * pu), WX * pv, (WY - TY) * pu, fill=False, ec=GREY, lw=0, hatch='////'))
    ax.plot([x0 + G * pv, x0 + (G + TX) * pv], [(G + TY) * pu] * 2, color=INK, lw=1.4)
    a = np.array((x0 + (G + 0.3 * TX) * pv, (G + 0.35 * TY) * pu))
    b = np.array((x0 + (G + 0.65 * TX) * pv, (G + 1.6 * TY) * pu))
    f_cut = ((G + TY) * pu - a[1]) / (b[1] - a[1])
    m = a + f_cut * (b - a)
    ax.plot(*zip(a, m), color=BLUE, lw=3.2, solid_capstyle='butt')
    ax.plot(*zip(m, b), color=RED, lw=3.2, solid_capstyle='butt')
    ax.text(x0 + WX * pv + 0.5 * pv, (G + 0.35 * TY) * pu, "job 1: the pieces\nin the lower tile", color=BLUE,
            fontsize=SMALL, va='center')
    ax.text(x0 + WX * pv + 0.5 * pv, (G + 1.6 * TY) * pu, "job 2: the pieces\nin the upper tile", color=RED,
            fontsize=SMALL, va='center')
    _note(ax, "tile boundary", (x0 + (G + 0.9 * TX) * pv, (G + TY) * pu), (x0 + WX * pv + 0.5 * pv, (G + TY) * pu))
    _note(ax, f"the two windows overlap\nhere, {2 * G} pixels wide", (x0 + 0.5 * G * pv, (G + TY) * pu),
          (x0 - 0.5 * pv, (G + TY) * pu), ha='right')
    ax.text(x0 - 0.5 * pv, 0.5 * WY * pu, "window\nof job 1", color=BLUE, fontsize=SMALL, ha='right', va='center')
    ax.text(x0 - 0.5 * pv, (TY + 0.5 * WY) * pu, "window\nof job 2", color=RED, fontsize=SMALL, ha='right',
            va='center')
    ax.text(x0 + WX * pv / 2, (TY + WY) * pu + 0.6 * pu, "one group crossing a tile boundary:\ntwo jobs, two windows",
            ha='center', va='bottom', fontsize=FS)
    ax.set_xlim(-0.5 * pv, x0 + (WX + 8.5) * pv)
    ax.set_ylim(-2.2 * pu, (TY + WY) * pu + 4.2 * pu)
    return fig


# ---------------------------------------------------------------------------------------------
def group_roots(ax=None):
    """One slab of silicon in section. Left: a charged track and the two delta rays it knocks out,
    boxed as one group with its root. Right: a photon converting inside the slab into an electron and
    a positron, two roots and two groups, because a photon is not charged."""
    fig, ax = _axes(ax, (12, 4.0))
    X, T = 11.0, 1.0
    ax.add_patch(Rectangle((0, 0), X, T, fc='#EEF2F8', ec=INK, lw=LW))
    ax.text(X + 0.1, T / 2, "one slab of\nsilicon: the\nsensor plane", fontsize=SMALL, va='center')
    # a charged track with two delta rays
    a, b = np.array((1.6, -0.45)), np.array((3.0, T + 0.45))
    ax.plot(*zip(a, b), color=INK, lw=1.6)
    _arrow(ax, tuple(a + 0.9 * (b - a)), tuple(b), lw=1.6, ms=10)
    for f0, dvx, sgn in ((0.42, 1.0, 1), (0.62, -0.9, -1)):
        s0 = a + f0 * (b - a)
        g = np.linspace(0, 1, 30)
        ax.plot(s0[0] + dvx * g, s0[1] + sgn * 0.08 * np.sin(np.pi * g) + 0.22 * g, color=PURPLE, lw=1.2)
    _box(ax, 0.75, 0.04, 3.15, T - 0.08, fc='none', ec=BLUE, lw=1.2, ls='--', rs=0.08)
    ax.text(2.3, T + 0.55, "one group: the track and the delta rays\nit knocked out in this slab", ha='center',
            va='bottom', fontsize=FS, color=BLUE)
    _note(ax, "root: the particle at\nthe top of the group", tuple(a + 0.2 * (b - a)), (0.0, -0.75), ha='left',
          va='top')
    s1 = a + 0.42 * (b - a)
    _note(ax, "node: one particle\nin one plane", (s1[0] + 0.95, s1[1] + 0.22), (3.2, -0.75), ha='left', va='top')
    _note(ax, "the delta ray's\nparent: the track", (s1[0] - 0.02, s1[1]), (-0.15, 0.62), ha='right')
    # a photon converting inside the slab
    cpt = np.array((7.6, 0.4))
    wx, wy = _wiggle((6.6, -0.6), tuple(cpt), n_waves=5, amp=0.035, n=300)
    ax.plot(wx, wy, color=GREY, lw=1.1)
    for (dx, col) in ((-0.95, BLUE), (1.05, RED)):
        e = cpt + np.array((dx, T + 0.25 - cpt[1]))
        ax.plot(*zip(cpt, e), color=col, lw=1.6)
        _arrow(ax, tuple(cpt + 0.85 * (e - cpt)), tuple(e), color=col, lw=1.6, ms=10)
        lo_x = min(cpt[0], cpt[0] + dx * (T - cpt[1]) / (T + 0.25 - cpt[1])) - 0.12
        _box(ax, lo_x, cpt[1] - 0.12, abs(dx) * 0.6 + 0.24, T - cpt[1] + 0.08, fc='none', ec=col, lw=1.1, ls='--',
             rs=0.05)
    ax.plot(*cpt, 'o', ms=4, color=INK)
    ax.text(5.9, -0.68, "a photon: no charge", fontsize=SMALL, ha='right', va='center', color='0.3')
    ax.text(7.6, T + 0.55, "a photon converting in the slab:\ntwo roots, two groups", ha='center', va='bottom',
            fontsize=FS)
    _note(ax, "each root is made inside the slab\n(created_in_slab); they join nothing,\nbecause their parent "
          "is not charged", tuple(cpt + np.array((0.05, -0.03))), (8.7, -0.75), ha='left', va='top')
    ax.set_xlim(-1.9, X + 1.4)
    ax.set_ylim(-1.55, T + 1.3)
    return fig


# ---------------------------------------------------------------------------------------------
def reticle_edge_logic(nv, nu, pitch_v_um=None, pitch_u_um=None, ax=None):
    """A sensor plane cut into reticles; one reticle of nv x nu pixels as bits (hit or not) going into
    the edge logic, and one number out. Struck through: what the edge logic does not see. The hit
    pattern is made up. Pixels to their true shape when the pitches are given."""
    fig, ax = _axes(ax, (13, 3.6))
    asp = 1.0 if pitch_v_um is None or pitch_u_um is None else float(pitch_u_um) / float(pitch_v_um)
    # the plane, cut into reticles
    rw = 0.95
    rh = rw * nu / nv * asp
    n_c, n_r = 3, max(3, int(round(1.9 / rh)))
    for i in range(n_c):
        for j in range(n_r):
            mine = (i, j) == (1, n_r // 2)
            ax.add_patch(Rectangle((i * rw, j * rh), rw, rh, fc=LBLUE if mine else '#F3F3F3', ec=INK,
                                   lw=1.2 if mine else 0.6))
    ax.text(n_c * rw / 2, n_r * rh + 0.12, "a sensor plane,\ncut into reticles", ha='center', va='bottom', fontsize=FS)
    # one reticle as bits
    x0 = n_c * rw + 0.9
    cw = 4.2 / nv
    ch = cw * asp
    y0 = n_r * rh / 2 - nu * ch / 2
    bits = _hit_bits(nv, nu)
    for j in range(nu):
        for i in range(nv):
            ax.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw, ch, fc=INK if bits[j, i] else 'white', ec='0.75',
                                   lw=0.4))
    ax.add_patch(Rectangle((x0, y0), nv * cw, nu * ch, fill=False, ec=BLUE, lw=1.2))
    for yy in (0, 1):
        ax.plot([2 * rw, x0], [(n_r // 2 + yy) * rh, y0 + yy * nu * ch], color=GREY, lw=0.6, ls='--')
    ax.text(x0 + nv * cw / 2, y0 + nu * ch + 0.12, f"one reticle: {nv} x {nu} pixels,\none bit each: hit or not",
            ha='center', va='bottom', fontsize=FS)
    # the edge logic and its one number
    xl = x0 + nv * cw + 0.75
    ym = y0 + nu * ch / 2
    _arrow(ax, (x0 + nv * cw + 0.08, ym), (xl - 0.04, ym), lw=1.2)
    _box(ax, xl, ym - 0.45, 1.75, 0.9, "the edge logic", fc=LBLUE, fs=FS)
    _arrow(ax, (xl + 1.79, ym), (xl + 2.35, ym), lw=1.2)
    _box(ax, xl + 2.4, ym - 0.45, 1.75, 0.9, "one number:\nthe count of\ncrossings", fs=9)
    # what it does not see
    yx = ym - 1.55
    ax.text(xl + 2.1, yx + 0.62, "it does not see:", ha='center', va='bottom', fontsize=SMALL)
    for k, txt in enumerate(("the charge", "the neighbouring\nreticles", "the shower axis")):
        _cross_out(ax, xl - 0.45 + k * 1.55, yx - 0.05, 1.4, 0.6, txt)
    ax.set_xlim(-0.1, xl + 4.3)
    ax.set_ylim(min(-0.1, yx - 0.2), max(n_r * rh, y0 + nu * ch) + 0.7)
    return fig


# ---------------------------------------------------------------------------------------------
def angle_geometry(pitch_v_um, pitch_u_um, thickness_um, ax=None):
    """Left: a track that crosses the slab at an angle theta to the normal moves thickness x
    tan(theta) sideways; it spans at least 2 pixels wherever it enters from arctan(p / t) on and at
    least 3 from arctan(2 p / t) on (drawn entering a pixel at its centre), along v and along u. Right: one 3-pixel L
    lit by two different tracks has one long axis, set by the pixel grid; a 3-pixel line has the
    long axis of its track."""
    t, pv, pu = float(thickness_um), float(pitch_v_um), float(pitch_u_um)
    fig, ax = _axes(ax, (13, 4.1))
    n, gap, lab = 4, 0.45 * t, 1.9 * t                   # pixels per section, gap between the rows, label room
    col_x = (0.0, n * pv + lab)
    for name, p, y0 in (("v", pv, t + gap), ("u", pu, 0.0)):
        for c, k in enumerate((1, 2)):                   # k pitches sideways: k + 1 pixels
            x0 = col_x[c]
            ax.add_patch(Rectangle((x0, y0), n * p, t, fc='#EEF2F8', ec=INK, lw=LW))
            for i in range(1, n):
                ax.plot([x0 + i * p] * 2, [y0, y0 + t], color='0.75', lw=0.5, ls=':')
            v_a, v_b = x0 + 0.5 * p, x0 + (0.5 + k) * p
            for i in range(n):
                hit = v_a <= x0 + (i + 1) * p and v_b >= x0 + i * p
                ax.add_patch(Rectangle((x0 + i * p + 0.06 * p, y0 + t), 0.88 * p, 0.09 * t,
                                       fc=INK if hit else 'white', ec=INK, lw=0.6))
            ax.plot([v_a, v_b], [y0, y0 + t], color=ORANGE, lw=1.6)
            ax.plot([v_a, v_a], [y0, y0 + 0.75 * t], color=GREY, lw=0.6, ls='--')
            th = np.degrees(np.arctan(k * p / t))
            ax.text(x0 + n * p + 0.08 * t, y0 + 0.5 * t, f"spans at least {k + 1} pixels\nwherever it enters "
                    f"from\narctan({k} x {p:g} / {t:g})\n= {th:.0f} degrees on", fontsize=SMALL, va='center')
        ax.text(-0.08 * t, y0 + 0.5 * t, f"along {name}:\n{p:g} um\npixels", ha='right', va='center', fontsize=SMALL)
    top = 2 * t + gap + 0.09 * t
    ax.text(0, top + 0.1 * t, "drawn: a track entering a pixel at its centre, at an angle to the normal (dashed);\n"
            "dark: the pixels it crosses. The angles hold wherever the track enters", ha='left', va='bottom',
            fontsize=SMALL, color='0.3')

    def long_axis(cells):
        cv = np.array([(i + 0.5) * pv for i, j in cells])
        cu = np.array([(j + 0.5) * pu for i, j in cells])
        vxx, vyy, vxy = np.var(cv), np.var(cu), np.mean((cv - cv.mean()) * (cu - cu.mean()))
        return cv.mean(), cu.mean(), 0.5 * np.arctan2(2 * vxy, vxx - vyy)

    def footprint(x0, y0, cells, tracks, label):
        _grid(ax, x0 - pv, y0 - pu, 5, 4, pv, pu, color='0.85', lw=0.5)
        for i, j in cells:
            ax.add_patch(Rectangle((x0 + i * pv, y0 + j * pu), pv, pu, fc='0.3', ec=INK, lw=0.6))
        mv, mu, ang = long_axis(cells)
        L = 1.6 * pv
        ax.plot([x0 + mv - L * np.cos(ang), x0 + mv + L * np.cos(ang)],
                [y0 + mu - L * np.sin(ang), y0 + mu + L * np.sin(ang)], color=BLUE, lw=1.3, ls='--')
        for (a, b), col in tracks:
            _arrow(ax, (x0 + a[0], y0 + a[1]), (x0 + b[0], y0 + b[1]), color=col, lw=1.3, ms=9)
        ax.text(x0 + 4.2 * pv, y0 + 1.0 * pu, label, ha='left', va='center', fontsize=SMALL)

    xL = col_x[1] + n * pv + lab + 1.4 * pv
    footprint(xL, t + gap + pu, [(0, 0), (1, 0), (1, 1)],
              [(((0.2 * pv, 0.4 * pu), (1.8 * pv, 1.2 * pu)), ORANGE), (((0.88 * pv, 0.12 * pu), (1.2 * pv, 1.88 * pu)),
                                                                       GREEN)],
              "a 3-pixel L, lit by\ntwo different tracks:\none long axis, set by\nthe pixel grid")
    footprint(xL, pu, [(0, 0), (1, 0), (2, 0)], [(((0.1 * pv, 0.45 * pu), (2.9 * pv, 0.6 * pu)), ORANGE)],
              "a 3-pixel line:\nits long axis is\nthe track's")
    ax.plot([], [], color=BLUE, ls='--', lw=1.3, label='long axis of the lit pixels')
    ax.plot([], [], color=ORANGE, lw=1.3, label='a track, seen from outside')
    ax.legend(loc='upper left', bbox_to_anchor=((xL - pv - (-1.0 * t)) / (xL + 7.6 * pv + 1.0 * t), 1.0),
              fontsize=SMALL, frameon=False)
    ax.set_xlim(-1.0 * t, xL + 7.6 * pv)
    ax.set_ylim(-0.15 * t, top + 0.5 * t)
    return fig


# ---------------------------------------------------------------------------------------------
def estimator_ladder(ax=None):
    """One hit map going into four estimators (plain name, code name in parentheses), four counts out,
    and the truth beside them. The hit map is made up."""
    import reticles                                     # the reticle is the PIXELAV window
    fig, ax = _axes(ax, (12, 4.2))
    nv, nu = reticles.SIZE_X, reticles.SIZE_Y
    cw = 3.0 / nv
    bits = _hit_bits(nv, nu)
    y0 = 1.3
    for j in range(nu):
        for i in range(nv):
            ax.add_patch(Rectangle((i * cw, y0 + j * cw), cw, cw, fc=INK if bits[j, i] else 'white', ec='0.8', lw=0.4))
    ax.add_patch(Rectangle((0, y0), nv * cw, nu * cw, fill=False, ec=BLUE, lw=1.2))
    ax.text(nv * cw / 2, y0 + nu * cw + 0.12, "the hit map of one reticle", ha='center', va='bottom', fontsize=FS)
    names = (("hit count", "hits"), ("cluster count", "clusters"), ("shape dictionary", "cluster_shape"),
             ("trees", "gbt"))
    xb, wb, hb = 4.4, 2.6, 0.72
    ym = y0 + nu * cw / 2
    for k, (plain, code) in enumerate(names):
        yb = 3.3 - k * 1.0
        _arrow(ax, (nv * cw + 0.1, ym), (xb - 0.05, yb + hb / 2), lw=0.8, ms=8)
        _box(ax, xb, yb, wb, hb, fc='white')
        ax.text(xb + wb / 2, yb + 0.47, plain, ha='center', va='center', fontsize=FS)
        ax.text(xb + wb / 2, yb + 0.2, f"({code})", ha='center', va='center', fontsize=SMALL, family='monospace')
        _arrow(ax, (xb + wb + 0.05, yb + hb / 2), (xb + wb + 0.65, yb + hb / 2), lw=0.8, ms=8)
        _box(ax, xb + wb + 0.7, yb + 0.06, 1.3, hb - 0.12, "a count", fc=LBLUE, fs=9)
    xt = xb + wb + 2.6
    _box(ax, xt, 1.25, 2.1, 1.3, "the truth:\nthe crossings in\nthe reticle", fc=LORANGE, fs=9)
    ax.text(xt + 1.05, 1.15, "known in the\nsimulation only", ha='center', va='top', fontsize=SMALL, color='0.3')
    ax.plot([xb + wb + 2.1, xb + wb + 2.3, xb + wb + 2.3, xb + wb + 2.1], [3.3 + hb - 0.06, 3.3 + hb - 0.06,
            0.3 + 0.06, 0.3 + 0.06], color=GREY, lw=0.8)
    ax.text(xb + wb + 2.38, 2.9, "each\ncompared\nwith it", fontsize=SMALL, va='center', color='0.3')
    ax.set_xlim(-0.1, xt + 2.2)
    ax.set_ylim(0.05, max(y0 + nu * cw + 0.6, 3.3 + hb + 0.15))
    return fig


# ---------------------------------------------------------------------------------------------
def five_blocks(n_showers=200, n_blocks=5, ax=None):
    """The showers as a strip cut into n_blocks blocks of consecutive showers; in each of n_blocks
    turns the estimators are fitted on all blocks but one and scored on that one, so every shower
    is scored once, by estimators that never saw it."""
    edges = np.round(np.linspace(0, n_showers, n_blocks + 1)).astype(int)
    fig, ax = _axes(ax, (11, 0.75 + 0.48 * (n_blocks + 1.5)), equal=False)
    W = 1.6
    for b in range(n_blocks):
        ax.text(b * W + W / 2, n_blocks + 0.45, f"showers {edges[b]} to {edges[b + 1] - 1}", ha='center', va='bottom',
                fontsize=SMALL)
    for turn in range(n_blocks):
        y = n_blocks - 1 - turn
        for b in range(n_blocks):
            score = b == turn
            ax.add_patch(Rectangle((b * W + 0.03, y + 0.1), W - 0.06, 0.72, fc=ORANGE if score else LBLUE,
                                   ec=INK, lw=0.6))
            ax.text(b * W + W / 2, y + 0.46, "score" if score else "fit", ha='center', va='center', fontsize=SMALL,
                    color='white' if score else INK)
        ax.text(-0.15, y + 0.46, f"turn {turn + 1}", ha='right', va='center', fontsize=SMALL)
    k = n_blocks - 1
    head = (f"fit on {WORDS.get(k, k)}, score on the {ORDINALS.get(n_blocks, f'{n_blocks}th')}: "
            f"{WORDS.get(n_blocks, n_blocks)} turns, every shower scored once")
    ax.text(n_blocks * W / 2, n_blocks + 1.1, head, ha='center', va='bottom', fontsize=FS)
    ax.set_xlim(-1.3, n_blocks * W + 0.1)
    ax.set_ylim(-0.05, n_blocks + 1.75)
    return fig


# ---------------------------------------------------------------------------------------------
def table_pull(ax=None):
    """Why a table scored per bin of the truth pulls toward the typical reticle, drawn from made-up
    reticles (a sketch, no data). Left: a tilted cloud of (hit count, true count) and the table, the
    mean truth at each hit count. Right: the mean of the table's estimates in each bin of the truth,
    bending toward the middle against the line estimate = truth."""
    fig, ax = _axes(ax, (12, 4.6), equal=False)
    rng = np.random.default_rng(3)                     # made-up reticles: most hold few crossings
    t = 0.3 + rng.gamma(1.6, 1.5, 3000)
    h = 3.0 * t * np.exp(0.4 * rng.standard_normal(t.size))
    keep = (t < 11.5) & (h < 45)
    t, h = t[keep], h[keep]
    W, H, X2 = 4.0, 3.0, 5.6                           # the size of a panel, the left edge of the right one
    sx, st = W / 45.0, H / 11.5                        # drawing units per hit pixel and per crossing
    for x0, xl, yl in ((0, "hit count", "true count"), (X2, "true count", "estimate")):
        _arrow(ax, (x0, 0), (x0 + W + 0.15, 0), lw=0.8, ms=8)
        _arrow(ax, (x0, 0), (x0, H + 0.15), lw=0.8, ms=8)
        ax.text(x0 + W / 2, -0.12, xl, ha='center', va='top', fontsize=FS)
        ax.text(x0 - 0.12, H / 2, yl, ha='right', va='center', rotation=90, fontsize=FS)
    # left: the cloud and the table, the mean truth in each slice of the hit count
    ax.plot(h * sx, t * st, '.', ms=2.0, color=GREY, alpha=0.5)
    e_h = np.quantile(h, np.linspace(0, 1, 21))       # slices of equal population
    k_h = np.clip(np.digitize(h, e_h) - 1, 0, len(e_h) - 2)
    full = list(range(len(e_h) - 2))                   # the last, widest slice left out
    hc = np.array([h[k_h == k].mean() for k in full])
    tm = np.array([t[k_h == k].mean() for k in full])
    ax.plot(hc * sx, tm * st, '-', color=BLUE, lw=2.2)
    ax.text(hc[-1] * sx + 0.1, tm[-1] * st, "the table:\nthe mean truth\nat each hit count", color=BLUE,
            fontsize=SMALL, va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax.text(W / 2, H + 0.25, "made-up reticles and the table fitted to them", ha='center', va='bottom', fontsize=FS)
    # right: each made-up reticle gets the table's value at its hit count; the mean per bin of the truth
    est = np.interp(h, hc, tm)
    e_t = np.linspace(0, 11.5, 13)
    k_t = np.digitize(t, e_t) - 1
    filled = [k for k in range(len(e_t) - 1) if np.sum(k_t == k) >= 40]
    tc = np.array([t[k_t == k].mean() for k in filled])
    em = np.array([est[k_t == k].mean() for k in filled])
    ax.plot([X2, X2 + W], [0, H], '--', color=GREY, lw=1.0)
    ax.text(X2 + W + 0.05, H, "estimate = truth", color='0.35', fontsize=SMALL, va='center')
    xs = X2 + tc * W / 11.5
    ax.plot(xs, em * st, '-o', ms=3.5, color=ORANGE, lw=2.0)
    ax.text(X2 + W / 2, H + 0.25, "the mean estimate in each bin of the truth", ha='center', va='bottom',
            fontsize=FS)
    i_lo, i_hi = 0, len(tc) - 1
    _note(ax, "at low truth the table\nover-estimates", (xs[i_lo], em[i_lo] * st),
          (X2 + 0.35, H - 0.35), ha='left', color=ORANGE)
    _note(ax, "at high truth it\nunder-estimates", (xs[i_hi], em[i_hi] * st), (X2 + W - 0.2, 0.5), ha='right',
          color=ORANGE)
    ax.text((X2 + W) / 2, -0.62, "a sketch: made-up reticles, no data", ha='center', va='top', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-0.6, X2 + W + 1.1)
    ax.set_ylim(-0.9, H + 0.75)
    return fig


# ---------------------------------------------------------------------------------------------
def layer_weights(n_thin, thin_mm, n_thick, thick_mm, ax=None):
    """The sensor layers of a module in depth: n_thin silicon layers each behind a tungsten plate of
    thin_mm, then n_thick behind plates of thick_mm (the plates to scale, the silicon not), and the
    weight of each in the sum over the shower, the tungsten it stands for in units of the thin plate."""
    fig, ax = _axes(ax, (13, 2.9), equal=False)
    gap, H = 0.35 * thin_mm, 1.0                     # space for the silicon after each plate (not to scale)
    x = 0.0
    starts = []
    for k in range(n_thin + n_thick):
        w_mm = thin_mm if k < n_thin else thick_mm
        ax.add_patch(Rectangle((x, 0), w_mm, H, fc='0.62', ec='none'))                     # tungsten
        ax.add_patch(Rectangle((x + w_mm + 0.3 * gap, 0), 0.4 * gap, H, fc=BLUE, ec='none'))  # silicon
        starts.append(x)
        x += w_mm + gap
    x_end, x_mid = x, starts[n_thin] - 0.5 * gap
    for (a, b, n, w_mm) in ((0.0, x_mid, n_thin, thin_mm), (x_mid + 0.5 * gap, x_end - gap, n_thick, thick_mm)):
        ax.plot([a, a, b, b], [H + 0.12, H + 0.22, H + 0.22, H + 0.12], color=INK, lw=0.9)
        ax.text((a + b) / 2, H + 0.3, f"weight {w_mm / thin_mm:g}", ha='center', va='bottom', fontsize=FS + 1,
                fontweight='bold')
        ax.text((a + b) / 2, -0.12, f"{n} layers, each behind {w_mm:g} mm of tungsten", ha='center', va='top',
                fontsize=FS)
    _note(ax, "tungsten plate", (starts[1] + 0.5 * thin_mm, 0.5 * H), (starts[1] - 0.3, -0.62), ha='left')
    _note(ax, "silicon layer", (starts[3] + thin_mm + 0.5 * gap, 0.5 * H), (starts[6], -0.62), ha='left')
    _arrow(ax, (-3.3 * thin_mm, 0.5 * H), (-0.15 * thin_mm, 0.5 * H), color=ORANGE, lw=1.8, ms=12)
    ax.text(-1.7 * thin_mm, 0.5 * H + 0.1, "the shower", color=ORANGE, fontsize=SMALL, ha='center', va='bottom')
    ax.text(x_end / 2, H + 0.85, f"a layer behind {thick_mm:g} mm stands for {thick_mm / thin_mm:g} times the "
            f"tungsten of one behind {thin_mm:g} mm", ha='center', va='bottom', fontsize=FS)
    ax.text(x_end, -0.62, "the plates to scale, the silicon not", ha='right', va='center', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-3.5 * thin_mm, x_end + 0.6 * thin_mm)
    ax.set_ylim(-0.9, H + 1.35)
    return fig
