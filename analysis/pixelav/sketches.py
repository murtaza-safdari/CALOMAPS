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
    edge_sharing(pitch_um, sigma_um)                a charge cloud straddling a pixel edge; the share beyond it
    reticle_edge_logic(nv, nu, pitch_v_um=None, pitch_u_um=None)
                                                    a plane cut into reticles; what the edge logic sees
    alone_together(nv, nu)                          two crossings in one reticle, counted alone and together
    turn_back()                                     one particle crossing the same slab twice
    angle_geometry(pitch_v_um, pitch_u_um, thickness_um)
                                                    how many pixels a slanted track spans; an L and a line
    estimator_ladder(with_layer=False)              one hit map, four estimators (five with the layer), the counts
    five_blocks(n_showers=200, n_blocks=5, highlight=None)
                                                    fit on four blocks of showers, score on the fifth
    table_pull()                                    why a table scored per bin of the truth pulls toward the middle
    layer_weights(n_thin, thin_mm, n_thick, thick_mm, show_weights=True, profiles=False, x0_per_layer=None)
                                                    the layers behind thin and thick plates and their weights
    fit_grid(cells, ks=(1, 2, 4, 8))                which densities each fit is trained on and applied to
    rounding_steps(steps, share)                    a number line in the rounding steps; what rounds up to 1

For the notebooks 00 to 08 (code facts come in as arguments; nothing below imports the analysis code):

    series_chain(current)                           the notebooks 00 to 08 and what they pass on
    barrel_face(rmin_mm, rmax_mm, n_faces, seg_half_deg, depth_window=None)
                                                    the barrel end-on, the +y face, the wedge, the depths kept
    four_readouts(mip_energy_kev, threshold_kev)    one made-up patch of pixels read four ways
    pion_regimes(n_layers)                          a pion punching through, and one interacting
    inversion()                                     a readout band turned into an energy interval
    density_net(layer_sizes, activation, alpha_clip, n_clip)
                                                    E in, four numbers out, a Crystal Ball curve, the score
    two_widths()                                    one Crystal Ball read two ways: sigma, and P16 to P84
    local_frame()                                   u, v, w of a sensor plane; alpha and beta of a track
    steps_to_crossings(layer_centers_mm, thickness_mm)
                                                    steps grouped into runs; the entry by time, not depth
    drop_reasons(thickness_um, max_cot, path_tolerance)
                                                    the five reasons a crossing stays out of a deck
    deck_window(nx, ny)                             PIXELAV's window on the plane, its x along v and y along u
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
    for f0, dvx, sgn, rise in ((0.42, 1.0, 1, 0.22), (0.62, -0.6, -1, 0.06)):   # the second ends inside the box
        s0 = a + f0 * (b - a)
        g = np.linspace(0, 1, 30)
        ax.plot(s0[0] + dvx * g, s0[1] + sgn * 0.08 * np.sin(np.pi * g) + rise * g, color=PURPLE, lw=1.2)
    _box(ax, 0.75, 0.04, 3.15, T - 0.08, fc='none', ec=BLUE, lw=1.2, ls='--', rs=0.08)
    ax.text(2.3, T + 0.55, "one group: the track and the delta rays\nit knocked out in this slab", ha='center',
            va='bottom', fontsize=FS, color=BLUE)
    _note(ax, "root: the particle at\nthe top of the group", tuple(a + 0.2 * (b - a)), (0.0, -0.75), ha='left',
          va='top')
    s1 = a + 0.42 * (b - a)                          # where the first delta ray leaves the track
    ax.plot(*s1, 'o', ms=3.5, color=INK, zorder=3)
    _note(ax, "node: one particle\nin one plane", tuple(s1), (3.2, -0.75), ha='left', va='top')
    _note(ax, "the delta ray's\nparent: the track", tuple(a + 0.53 * (b - a)), (-0.15, 0.62), ha='right')
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
def edge_sharing(pitch_um, sigma_um, *, ax=None):
    """A sketch of how a pixel edge shares the charge (sigma made up, labelled so). Left: a straight
    track crossing the slab at x, near the edge x0 between two pixels of pitch_um, and the charge
    cloud at the pixel side, a Gaussian of width sigma_um; the share beyond the edge shaded. Right:
    that share against x, from the centre of the pixel to the centre of its neighbour: the S-curve
    Phi((x - x0) / sigma)."""
    p, s = float(pitch_um), float(sigma_um)
    phi = lambda z: 0.5 * (1 + _erf(np.asarray(z, float) / np.sqrt(2)))
    fig, ax = _axes(ax, (12, 3.9))
    # left, in um: the slab (not to scale in depth), two pixels, the edge x0 = p between them
    x0, t, xt = p, 0.8 * p, p - 0.5 * s
    lo, hi = -0.45 * p, 2.45 * p
    ax.add_patch(Rectangle((lo, 0), hi - lo, t, fc='#EEF2F8', ec=INK, lw=LW))
    for a, b, fc, name in ((lo, x0 - p, 'white', ""), (x0 - p, x0, BLUE, "this pixel"),
                           (x0, x0 + p, ORANGE, "its neighbour"), (x0 + p, hi, 'white', "")):
        ax.add_patch(Rectangle((a + 0.04 * p, t), b - a - 0.08 * p, 0.07 * p, fc=fc, ec=INK, lw=0.6))
        if name:
            ax.text((a + b) / 2, -0.08 * p, name, ha='center', va='top', fontsize=SMALL)
    ax.plot([x0, x0], [0, t + 1.05 * p], color=INK, lw=0.9, ls='--')
    ax.text(x0, t + 1.1 * p, "the pixel edge, x0", ha='center', va='bottom', fontsize=SMALL)
    _arrow(ax, (xt, -0.35 * p), (xt, t), lw=1.4, ms=10)
    ax.text(xt - 0.04 * p, -0.38 * p, "a straight track at x", ha='right', va='top', fontsize=SMALL)
    # the charge cloud at the pixel side, split at the edge
    base, hg = t + 0.1 * p, 0.85 * p
    u = np.linspace(xt - 3.5 * s, xt + 3.5 * s, 200)
    g = base + hg * np.exp(-0.5 * ((u - xt) / s) ** 2)
    ax.fill_between(u[u <= x0], base, g[u <= x0], color=LBLUE, lw=0)
    ax.fill_between(u[u >= x0], base, g[u >= x0], color=LORANGE, lw=0)
    ax.plot(u, g, color=INK, lw=0.9)
    yh = base + hg * np.exp(-0.5)
    _dim(ax, (xt - s, yh), (xt, yh), "sigma", (-0.5 * s - 0.05 * p, 0), ha='right')
    _note(ax, "the share beyond the edge:\nit goes to the neighbour", (x0 + 0.4 * s, base + 0.12 * hg),
          (x0 + 0.55 * p, base + 0.55 * hg))
    ax.text(lo + 0.03 * p, base + 0.55 * hg, "the charge\nreaching the\npixel side", ha='left', va='center',
            fontsize=SMALL)
    # right: the share beyond the edge against x
    X2, W, H = hi + 1.45 * p, 1.7 * p, 1.4 * p
    xa, xb = x0 - 0.5 * p, x0 + 0.5 * p
    to_x = lambda x: X2 + (x - xa) / (xb - xa) * W
    _arrow(ax, (X2, 0), (X2 + W + 0.1 * p, 0), lw=0.8, ms=8)
    _arrow(ax, (X2, 0), (X2, H + 0.12 * p), lw=0.8, ms=8)
    xx = np.linspace(xa, xb, 200)
    ax.plot(to_x(xx), H * phi((xx - x0) / s), color=ORANGE, lw=2.0)
    ax.plot([to_x(x0)] * 2, [0, H], color=INK, lw=0.7, ls='--')
    ax.plot([X2, X2 + W], [H, H], color=GREY, lw=0.6, ls=':')
    for v, lab in ((0, "0"), (H, "1")):
        ax.text(X2 - 0.04 * p, v, lab, ha='right', va='center', fontsize=SMALL)
    ax.text(X2 - 0.2 * p, H / 2, "share beyond\nthe edge", ha='right', va='center', fontsize=SMALL)
    for x, lab in ((xa, "centre of\nthis pixel"), (x0, "x0"), (xb, "centre of\nits neighbour")):
        ax.text(to_x(x), -0.07 * p, lab, ha='center', va='top', fontsize=SMALL)
    ax.text(X2 + W / 2, -0.5 * p, "x, where the track crosses", ha='center', va='top', fontsize=SMALL)
    yt = H * float(phi((xt - x0) / s))
    ax.plot([to_x(xt)] * 2, [0, yt], color=GREY, lw=0.6, ls=':')
    ax.plot(to_x(xt), yt, 'o', ms=4, color=INK)
    ax.text(to_x(x0) + 0.08 * p, 0.82 * H, "Phi((x - x0) / sigma)", color=ORANGE, fontsize=FS, ha='left',
            va='center')
    ax.text(X2 + W / 2, H + 0.2 * p, "the share against x: an S-curve", ha='center', va='bottom', fontsize=FS)
    ax.text((lo + X2 + W) / 2, -0.85 * p, f"a sketch: a made-up sigma of {s:g} um; the pixels are {p:g} um, "
            "the depth not to scale", ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(lo - 0.1 * p, X2 + W + 0.3 * p)
    ax.set_ylim(-1.05 * p, t + 1.35 * p)
    return fig


# ---------------------------------------------------------------------------------------------
def reticle_edge_logic(nv, nu, pitch_v_um=None, pitch_u_um=None, ax=None):
    """A sensor plane cut into reticles; one reticle of nv x nu pixels as bits (hit or not) going into
    the edge logic, and one number out. Struck through: what the edge logic does not see. The hit
    pattern is made up. Pixels to their true shape, and the reticle's size in mm, when the pitches
    are given."""
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
    size = f"{nv} x {nu} pixels"
    if pitch_v_um is not None and pitch_u_um is not None:
        size += f" = {nv * float(pitch_v_um) / 1000:g} x {nu * float(pitch_u_um) / 1000:g} mm"
    ax.text(x0 + nv * cw / 2, y0 + nu * ch + 0.12, f"one reticle: {size},\none bit each: hit or not",
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
def alone_together(nv, nu, *, ax=None):
    """One reticle of nv x nu pixels drawn twice, with the same two crossings. Left: each crossing
    alone, its own hit pixels and its own cluster; summed, two clusters, and the pixel both crossings
    hit is counted twice. Right: both together, that pixel counted once and the two touching patches
    one cluster. The hit patterns are made up."""
    cv, cu = nv // 2, nu // 2
    A = [(cv - 3, cu - 1), (cv - 2, cu - 1), (cv - 1, cu), (cv, cu)]       # pixels (along v, along u)
    B = [(cv, cu), (cv + 1, cu + 1), (cv + 1, cu + 2), (cv + 2, cu + 2)]
    shared = set(A) & set(B)
    union = set(A) | set(B)
    fig, ax = _axes(ax, (13, 4.0))
    cw = 4.6 / nv
    X2 = nv * cw + 1.6

    def reticle(x0):
        _grid(ax, x0, 0, nv, nu, cw, cw, color='0.8', lw=0.4)
        ax.add_patch(Rectangle((x0, 0), nv * cw, nu * cw, fill=False, ec=BLUE, lw=1.2))

    def outline(x0, cells, col):
        i0, i1 = min(i for i, j in cells), max(i for i, j in cells) + 1
        j0, j1 = min(j for i, j in cells), max(j for i, j in cells) + 1
        _box(ax, x0 + (i0 - 0.3) * cw, (j0 - 0.3) * cw, (i1 - i0 + 0.6) * cw, (j1 - j0 + 0.6) * cw, fc='none', ec=col,
             lw=1.1, ls='--', rs=0.3 * cw)

    # left: each crossing alone, in its own colour; the pixel both hit split between the two
    reticle(0)
    for cells, col in ((A, BLUE), (B, RED)):
        for i, j in cells:
            if (i, j) not in shared:
                ax.add_patch(Rectangle((i * cw, j * cw), cw, cw, fc=col, ec=INK, lw=0.5))
        outline(0, cells, col)
    for i, j in shared:
        ax.add_patch(Polygon([(i * cw, j * cw), ((i + 1) * cw, j * cw), (i * cw, (j + 1) * cw)], fc=BLUE, ec='none'))
        ax.add_patch(Polygon([((i + 1) * cw, j * cw), ((i + 1) * cw, (j + 1) * cw), (i * cw, (j + 1) * cw)], fc=RED,
                             ec='none'))
        ax.add_patch(Rectangle((i * cw, j * cw), cw, cw, fill=False, ec=INK, lw=0.5))
        _note(ax, "hit by both:\ncounted once for each", ((i + 0.5) * cw, (j + 0.2) * cw), (nv * cw + 0.15, 0.25 * nu * cw))
    ax.text(nv * cw / 2, nu * cw + 0.12, "each crossing alone", ha='center', va='bottom', fontsize=FS)
    ax.text(nv * cw / 2, -0.12, f"alone: 2 clusters, {len(A) + len(B)} hit pixels", ha='center', va='top',
            fontsize=FS, fontweight='bold')
    # right: both together, every hit pixel once, one cluster
    reticle(X2)
    for i, j in union:
        ax.add_patch(Rectangle((X2 + i * cw, j * cw), cw, cw, fc=INK, ec='0.75', lw=0.5))
    outline(X2, sorted(union), GREEN)
    ax.text(X2 + nv * cw / 2, nu * cw + 0.12, "both together", ha='center', va='bottom', fontsize=FS)
    ax.text(X2 + nv * cw / 2, -0.12, f"together: 1 cluster, {len(union)} hit pixels", ha='center', va='top',
            fontsize=FS, fontweight='bold')
    ax.plot([], [], 's', ms=7, color=BLUE, label='the hit pixels of one crossing')
    ax.plot([], [], 's', ms=7, color=RED, label='the hit pixels of the other')
    ax.plot([], [], color=GREY, ls='--', lw=1.1, label='a cluster: touching hit pixels')
    ax.legend(loc='upper center', bbox_to_anchor=((X2 + nv * cw) / 2, -0.45), bbox_transform=ax.transData, ncol=3,
              fontsize=SMALL, frameon=False)
    ax.text((X2 + nv * cw) / 2, -0.85, f"one reticle of {nv} x {nu} pixels; the hit patterns are made up",
            ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(-0.1, X2 + nv * cw + 0.1)
    ax.set_ylim(-1.15, nu * cw + 0.55)
    return fig


# ---------------------------------------------------------------------------------------------
def turn_back(*, ax=None):
    """A sketch, side view: one slab of silicon with the tungsten plate behind it, and one track that
    crosses the slab, turns round in the tungsten and comes back through the same slab. The truth
    counts it twice: one particle, two crossings. The track is made up."""
    fig, ax = _axes(ax, (10, 3.6))
    xs, ws, xw, ww, Y = 0.0, 0.18, 0.3, 1.6, 2.4         # slab, plate, height (drawing units)
    ax.add_patch(Rectangle((xs, 0), ws, Y, fc=BLUE, ec=INK, lw=LW))
    ax.add_patch(Rectangle((xw, 0), ww, Y, fc='0.62', ec=INK, lw=LW))
    ax.text(xs + ws / 2, Y + 0.08, "one slab\nof silicon", ha='center', va='bottom', fontsize=SMALL)
    ax.text(xw + ww / 2, Y - 0.1, "the tungsten\nplate behind it", ha='center', va='top', fontsize=SMALL,
            color='white')
    # the track: in from the left, through the slab, round in the tungsten, back through the slab
    a, b = np.array((-2.2, 1.95)), np.array((1.05, 1.6))
    r, c = 0.42, np.array((1.05, 1.18))
    ang = np.linspace(np.pi / 2, -np.pi / 2, 60)
    arc = np.c_[c[0] + 1.25 * r * np.cos(ang), c[1] + r * np.sin(ang)]
    e = np.array((-2.2, 0.35))
    path = np.vstack([np.linspace(a, b, 20), arc, np.linspace(arc[-1], e, 20)])
    ax.plot(path[:, 0], path[:, 1], color=INK, lw=1.6)
    _arrow(ax, tuple(path[-4]), tuple(e), lw=1.6, ms=11)
    _arrow(ax, tuple(path[5]), tuple(path[8]), lw=1.6, ms=11)
    # the two crossings of the slab
    for k, (p0, p1) in enumerate(((a, b), (arc[-1], e))):
        f = (xs + ws / 2 - p0[0]) / (p1[0] - p0[0])
        q = p0 + f * (p1 - p0)
        ax.add_patch(Circle(tuple(q), 0.16, fill=False, ec=ORANGE, lw=1.6, zorder=3))
        txt = "first crossing" if k == 0 else "second crossing,\non the way back"
        _note(ax, txt, (q[0] - 0.16, q[1]), (-1.2, q[1] + (0.45 if k == 0 else -0.45)), ha='right', color=ORANGE)
    _note(ax, "it turns round\nin the tungsten", (c[0] + 1.25 * r, c[1]), (xw + ww + 0.2, c[1]), ha='left')
    ax.text(-2.2, Y + 0.08, "the shower comes\nfrom this side", ha='left', va='bottom', fontsize=SMALL, color='0.3')
    ax.text((-2.2 + xw + ww) / 2, -0.15, "one particle, two crossings", ha='center', va='top', fontsize=FS + 1,
            fontweight='bold')
    ax.text((-2.2 + xw + ww) / 2, -0.55, "a sketch: the track is made up", ha='center', va='top', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-2.5, xw + ww + 1.6)
    ax.set_ylim(-0.85, Y + 0.6)
    return fig


# ---------------------------------------------------------------------------------------------
def angle_geometry(pitch_v_um, pitch_u_um, thickness_um, ax=None):
    """Left: a track that crosses the slab at an angle theta to the normal moves thickness x
    tan(theta) sideways; it spans at least 2 pixels wherever it enters from arctan(p / t) on and at
    least 3 from arctan(2 p / t) on (drawn entering a pixel at its centre), along v and along u. Right: one 3-pixel L
    lit by two different tracks has one long axis, set by the pixel grid; a 3-pixel line has the
    long axis of its track. The title states the point: a rule that reads direction from the long
    axis is misled by an L."""
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
    ax.plot([], [], color=ORANGE, lw=1.3, label='the track projected onto the plane')
    ax.plot([], [], color=GREEN, lw=1.3, label='a second track, lighting the other arm')
    ax.legend(loc='upper left', bbox_to_anchor=(xL - pv, top + 0.5 * t), bbox_transform=ax.transData,
              fontsize=SMALL, frameon=False)
    ax.text((-1.0 * t + xL + 7.6 * pv) / 2, top + 0.7 * t, "a rule that reads direction from the long axis is "
            "misled by an L", ha='center', va='bottom', fontsize=FS + 1)
    ax.set_xlim(-1.0 * t, xL + 7.6 * pv)
    ax.set_ylim(-0.15 * t, top + 1.05 * t)
    return fig


# ---------------------------------------------------------------------------------------------
def estimator_ladder(*, with_layer=False, ax=None):
    """One hit map going into the estimators (plain name, code name in parentheses), under each what
    it reads, a count out of each, and the truth beside them: four estimators, five with `with_layer`
    (the trees given the layer as well). The hit map is made up."""
    import mip_counting as mc
    import reticles                                     # the reticle has the window's size
    nv, nu = reticles.SIZE_X, reticles.SIZE_Y
    n_feat = len(mc.FEATURES) + nv + nu                 # the numbers the trees read: FEATURES, then per row and column
    names = [("hit count", "hits", "h, the hit pixels"), ("cluster count", "clusters", "c, the clusters"),
             ("shape dictionary", "cluster_shape", "the shapes of the clusters"),
             ("trees", "gbt", f"{n_feat} numbers")]
    if with_layer:
        names.append(("trees with the layer", "gbt_layer", f"the same {n_feat} plus the layer"))
    n = len(names)
    step, hb, y_lo = 1.1, 0.72, 0.45
    y_hi = y_lo + (n - 1) * step + hb
    fig, ax = _axes(ax, (12, y_hi + 0.35))
    cw = 3.0 / nv
    bits = _hit_bits(nv, nu)
    ym = (y_lo + y_hi) / 2
    y0 = ym - nu * cw / 2
    for j in range(nu):
        for i in range(nv):
            ax.add_patch(Rectangle((i * cw, y0 + j * cw), cw, cw, fc=INK if bits[j, i] else 'white', ec='0.8', lw=0.4))
    ax.add_patch(Rectangle((0, y0), nv * cw, nu * cw, fill=False, ec=BLUE, lw=1.2))
    ax.text(nv * cw / 2, y0 + nu * cw + 0.12, "the hit map of one reticle", ha='center', va='bottom', fontsize=FS)
    xb, wb = 4.4, 2.6
    for k, (plain, code, reads) in enumerate(names):
        yb = y_lo + (n - 1 - k) * step
        _arrow(ax, (nv * cw + 0.1, ym), (xb - 0.05, yb + hb / 2), lw=0.8, ms=8)
        _box(ax, xb, yb, wb, hb, fc='white')
        ax.text(xb + wb / 2, yb + 0.47, plain, ha='center', va='center', fontsize=FS)
        ax.text(xb + wb / 2, yb + 0.2, f"({code})", ha='center', va='center', fontsize=SMALL, family='monospace')
        ax.text(xb + wb / 2, yb - 0.05, f"reads: {reads}", ha='center', va='top', fontsize=SMALL, color='0.3')
        _arrow(ax, (xb + wb + 0.05, yb + hb / 2), (xb + wb + 0.65, yb + hb / 2), lw=0.8, ms=8)
        _box(ax, xb + wb + 0.7, yb + 0.06, 1.3, hb - 0.12, "a count", fc=LBLUE, fs=9)
    xt = xb + wb + 2.6
    _box(ax, xt, ym - 0.65, 2.1, 1.3, "the truth:\nthe crossings in\nthe reticle", fc=LORANGE, fs=9)
    ax.text(xt + 1.05, ym - 0.75, "known in the\nsimulation only", ha='center', va='top', fontsize=SMALL, color='0.3')
    ax.plot([xb + wb + 2.1, xb + wb + 2.3, xb + wb + 2.3, xb + wb + 2.1], [y_hi - 0.06, y_hi - 0.06,
            y_lo + 0.06, y_lo + 0.06], color=GREY, lw=0.8)
    ax.text(xb + wb + 2.38, ym + 1.0, "each\ncompared\nwith it", fontsize=SMALL, va='center', color='0.3')
    ax.set_xlim(-0.1, xt + 2.2)
    ax.set_ylim(0.05, max(y0 + nu * cw + 0.6, y_hi + 0.15))
    return fig


# ---------------------------------------------------------------------------------------------
def five_blocks(n_showers=200, n_blocks=5, *, highlight=None, ax=None):
    """The showers as a strip cut into n_blocks blocks of consecutive showers; in each of n_blocks
    turns the estimators are fitted on all blocks but one and scored on that one, so every shower
    is scored once, by estimators that never saw it. `highlight` is a turn number counted from 1
    (turn 1 is the top row); that turn is outlined and labelled "this notebook"."""
    if highlight is not None and not 1 <= highlight <= n_blocks:
        raise ValueError(f"highlight must be a turn from 1 to {n_blocks}, not {highlight!r}")
    edges = np.round(np.linspace(0, n_showers, n_blocks + 1)).astype(int)
    fig, ax = _axes(ax, (11 if highlight is None else 12.2, 0.75 + 0.48 * (n_blocks + 1.5)), equal=False)
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
        mine = highlight == turn + 1
        ax.text(-0.15, y + 0.46, f"turn {turn + 1}", ha='right', va='center', fontsize=SMALL,
                fontweight='bold' if mine else 'normal')
        if mine:
            ax.add_patch(Rectangle((-0.04, y + 0.03), n_blocks * W + 0.08, 0.86, fill=False, ec=BLUE, lw=1.8))
            ax.text(n_blocks * W + 0.2, y + 0.46, "this notebook", ha='left', va='center', fontsize=FS,
                    fontweight='bold', color=BLUE)
    k = n_blocks - 1
    head = (f"fit on {WORDS.get(k, k)}, score on the {ORDINALS.get(n_blocks, f'{n_blocks}th')}: "
            f"{WORDS.get(n_blocks, n_blocks)} turns, every shower scored once")
    ax.text(n_blocks * W / 2, n_blocks + 1.1, head, ha='center', va='bottom', fontsize=FS)
    ax.set_xlim(-1.3, n_blocks * W + (0.1 if highlight is None else 1.6))
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
def layer_weights(n_thin, thin_mm, n_thick, thick_mm, *, show_weights=True, profiles=False, x0_per_layer=None,
                  ax=None):
    """The sensor layers of a module in depth: n_thin silicon layers each behind a tungsten plate of
    thin_mm, then n_thick behind plates of thick_mm (the plates to scale, the silicon not), and the
    weight of each in the sum over the shower, the tungsten it stands for in units of the thin plate.
    With show_weights False the weights are left out and the title names the plates. With profiles
    True two made-up depth profiles (sketches, not data) are drawn over the layers: an early shower
    and a late one, which puts more of its crossings behind the thick plates. With x0_per_layer (the
    depth in radiation lengths X0 at each layer, cumulative, one value per layer) an upper axis
    shows the depth in X0."""
    n_all = n_thin + n_thick
    if x0_per_layer is not None and len(x0_per_layer) != n_all:
        raise ValueError(f"x0_per_layer needs one value per layer ({n_all}), not {len(x0_per_layer)}")
    gap, H = 0.35 * thin_mm, 1.0                     # space for the silicon after each plate (not to scale)
    y_w = H + 0.75 if show_weights else H + 0.2      # where the X0 axis, the profiles or the title start
    y_x0 = y_w + 0.12                                # the X0 axis, when there is one
    if x0_per_layer is not None:
        y_w += 0.75
    hp = 1.7 if profiles else 0.0                    # the height of the profiles
    y_title = y_w + (hp + 0.25 if profiles else (0.1 if show_weights or x0_per_layer is not None else 0.05))
    fig, ax = _axes(ax, (13, 2.9 + (y_title - H - 0.85) * 0.9), equal=False)
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
        if show_weights:
            ax.plot([a, a, b, b], [H + 0.12, H + 0.22, H + 0.22, H + 0.12], color=INK, lw=0.9)
            ax.text((a + b) / 2, H + 0.3, f"weight {w_mm / thin_mm:g}", ha='center', va='bottom', fontsize=FS + 1,
                    fontweight='bold')
        ax.text((a + b) / 2, -0.12, f"{n} layers, each behind {w_mm:g} mm of tungsten", ha='center', va='top',
                fontsize=FS)
    _note(ax, "tungsten plate", (starts[1] + 0.5 * thin_mm, 0.5 * H), (starts[1] - 0.3, -0.62), ha='left')
    _note(ax, "silicon layer", (starts[3] + thin_mm + 0.5 * gap, 0.5 * H), (starts[6], -0.62), ha='left')
    _arrow(ax, (-3.3 * thin_mm, 0.5 * H), (-0.15 * thin_mm, 0.5 * H), color=ORANGE, lw=1.8, ms=12)
    ax.text(-1.7 * thin_mm, 0.5 * H + 0.1, "the shower", color=ORANGE, fontsize=SMALL, ha='center', va='bottom')
    w_k = np.array([thin_mm] * n_thin + [thick_mm] * n_thick)
    x_si = np.array(starts) + w_k + 0.5 * gap         # the silicon of each layer
    if x0_per_layer is not None:
        # depth in X0 along the module: 0 at the front face, the given value at each layer's silicon
        x0s = np.r_[0.0, np.asarray(x0_per_layer, float)]
        xs_ = np.r_[0.0, x_si]
        ax.plot([0, x_si[-1]], [y_x0] * 2, color=INK, lw=0.8)
        top = float(x0s[-1])
        major = 5 if top > 12 else 1
        for v in np.arange(0, top + 1e-9, 1.0):
            big = abs(v / major - round(v / major)) < 1e-9
            xv = float(np.interp(v, x0s, xs_))
            ax.plot([xv] * 2, [y_x0, y_x0 + (0.1 if big else 0.05)], color=INK, lw=0.8 if big else 0.5)
            if big:
                ax.text(xv, y_x0 + 0.13, f"{v:g}", ha='center', va='bottom', fontsize=SMALL)
        ax.text(x_si[-1] + 0.3 * thin_mm, y_x0, f"depth in X0\n(radiation lengths);\n{top:g} X0 at the last layer",
                ha='left', va='center', fontsize=SMALL)
    if profiles:
        # made-up profiles: crossings per layer against the tungsten in front of it, scaled so that both
        # showers have the same weighted total; drawn at the silicon of each layer
        depth = np.cumsum(w_k)
        shape = lambda d, a, b: d ** a * np.exp(-d / b)
        curves = []
        for a, b in ((2.0, 0.1 * depth[-1]), (4.5, 0.12 * depth[-1])):
            c = shape(depth, a, b)
            curves.append(c / np.sum(c * w_k / thin_mm))
        top = max(c.max() for c in curves)
        early, late = (y_w + 0.9 * hp * c / top for c in curves)
        ax.plot([0, x_end - gap], [y_w] * 2, color=GREY, lw=0.6)
        ax.fill_between(x_si[n_thin:], y_w, late[n_thin:], color=LRED, lw=0, alpha=0.8)
        ax.plot(x_si, early, '-o', ms=3, color=BLUE, lw=1.4)
        ax.plot(x_si, late, '-o', ms=3, color=RED, lw=1.4)
        i_e = int(np.argmax(early))
        ax.text(x_si[i_e] - 0.5 * thin_mm, early[i_e] + 0.05, "an early shower", color=BLUE, fontsize=SMALL,
                ha='right', va='bottom')
        i_l = int(np.argmax(late))
        _note(ax, "a late shower, more of it behind the thick plates:\nweight 1 counts it short",
              (x_si[i_l + 2], 0.5 * (late[i_l + 2] + y_w)), (x_si[i_l + 4] + 1.0, y_w + 0.75 * hp), color=RED)
        ax.text(-0.3 * thin_mm, y_w + 0.45 * hp, "crossings\nper layer", ha='right', va='center', fontsize=SMALL)
        ax.text(x_end, y_w - 0.03, "the two profiles are sketches, not data", ha='right', va='top', fontsize=SMALL,
                color='0.35')
    if show_weights:
        title = (f"a layer behind {thick_mm:g} mm stands for {thick_mm / thin_mm:g} times the tungsten of one "
                 f"behind {thin_mm:g} mm")
    else:
        title = f"{n_thin} layers behind {thin_mm:g} mm of tungsten, then {n_thick} behind {thick_mm:g} mm"
    ax.text(x_end / 2, y_title, title, ha='center', va='bottom', fontsize=FS)
    ax.text(x_end, -0.62, "the plates to scale, the silicon not", ha='right', va='center', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-3.5 * thin_mm, x_end + (0.6 if x0_per_layer is None else 6.5) * thin_mm)
    ax.set_ylim(-0.9, y_title + 0.5)
    return fig


# ---------------------------------------------------------------------------------------------
def fit_grid(cells, *, ks=(1, 2, 4, 8), ax=None):
    """Which densities each fit is trained on and applied to: one row per entry of `cells` (its key is
    the row label, its value a dict from K to 'trained', 'applied' or 'both'), one column per K in ks
    (K showers stacked in each reticle). A K a row does not name is left empty."""
    kinds = {'both': "trained and\napplied", 'trained': "trained", 'applied': "applied"}
    for row, d in cells.items():
        bad = {k: v for k, v in d.items() if v not in kinds}
        if bad:
            raise ValueError(f"row {row!r}: each value must be one of {sorted(kinds)}, not {bad}")
    labels = list(cells)
    cw, ch = 1.45, 0.62
    room = 0.085 * max(len(s) for s in labels) + 0.2
    n_r, n_k = len(labels), len(ks)
    fig, ax = _axes(ax, (room + cw * n_k + 0.5, 0.75 * n_r + 1.1), equal=False)
    for c, k in enumerate(ks):
        ax.text((c + 0.5) * cw, n_r * ch + 0.1, f"K = {k}", ha='center', va='bottom', fontsize=FS)
    for r, row in enumerate(labels):
        y = (n_r - 1 - r) * ch
        ax.text(-0.15, y + ch / 2, row, ha='right', va='center', fontsize=FS)
        for c, k in enumerate(ks):
            x, kind = c * cw, cells[row].get(k)
            if kind == 'both':
                ax.add_patch(Polygon([(x, y), (x + cw, y), (x, y + ch)], fc=LBLUE, ec='none'))
                ax.add_patch(Polygon([(x + cw, y), (x + cw, y + ch), (x, y + ch)], fc=LORANGE, ec='none'))
            elif kind is not None:
                ax.add_patch(Rectangle((x, y), cw, ch, fc=LBLUE if kind == 'trained' else LORANGE, ec='none'))
            ax.add_patch(Rectangle((x, y), cw, ch, fill=False, ec=INK, lw=0.6))
            if kind is not None:
                ax.text(x + cw / 2, y + ch / 2, kinds[kind], ha='center', va='center', fontsize=SMALL)
    ax.text(n_k * cw / 2, -0.12, "K: the showers stacked in each reticle", ha='center', va='top', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-room, n_k * cw + 0.1)
    ax.set_ylim(-0.45, n_r * ch + 0.45)
    return fig


# ---------------------------------------------------------------------------------------------
def rounding_steps(steps, share, *, ax=None):
    """A number line from 0 to twice the largest step, with ticks at the multiples of each step in
    `steps`. The band from half the largest step up to it is shaded and labelled with `share` (the
    fraction, from 0 to 1, of the reticles with a hit pixel whose estimate lies there); arrows send
    it up to the largest step, where rounding in that step puts it."""
    steps = sorted(float(s) for s in steps)
    s_max = steps[-1]
    fig, ax = _axes(ax, (11, 2.8), equal=False)
    X = 2 * s_max
    sx = 8.0 / X                                       # drawing units per unit of the estimate
    ax.add_patch(Rectangle((0.5 * s_max * sx, -0.04), 0.5 * s_max * sx, 0.5, fc=LORANGE, ec='none'))
    _arrow(ax, (-0.1, 0), (X * sx + 0.3, 0), lw=1.0, ms=9)
    for n, s in enumerate(steps):
        h = 0.08 + 0.1 * n                             # the larger the step, the longer its ticks
        for v in np.arange(0, X + 1e-9, s):
            ax.plot([v * sx] * 2, [-h, h], color=INK if s == s_max else GREY, lw=1.2 if s == s_max else 0.7)
    for v in np.arange(0, X + 1e-9, 0.5 * s_max):
        ax.text(v * sx, -0.32, f"{v:g}", ha='center', va='top', fontsize=FS)
    ax.text(X * sx / 2, -0.62, "the estimate of one reticle", ha='center', va='top', fontsize=SMALL)
    keys = ", ".join(f"steps of {s:g}" for s in steps)
    ax.text(X * sx + 0.35, -0.02, f"ticks, short to long:\n{keys}", ha='left', va='center', fontsize=SMALL,
            color='0.3')
    ax.text(0.75 * s_max * sx, 0.56, f"{share:.1%} of the reticles\nwith a hit pixel", ha='center', va='bottom',
            fontsize=FS, color=ORANGE, fontweight='bold')
    for f in (0.56, 0.7, 0.84):
        ax.annotate('', xy=(s_max * sx, 0.3), xytext=(f * s_max * sx, 0.22),
                    arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.0, mutation_scale=9, shrinkA=0, shrinkB=1,
                                    connectionstyle='arc3,rad=-0.35'))
    ax.text(s_max * sx + 0.12, 0.38, f"in steps of {s_max:g}, all\nof them become {s_max:g}", ha='left', va='bottom',
            fontsize=SMALL, color=ORANGE)
    ax.set_xlim(-0.4, X * sx + 2.4)
    ax.set_ylim(-0.95, 1.05)
    return fig


# =============================================================================================
# notebooks 00 to 08
# =============================================================================================
def _cb_shape(x, mu, sigma, alpha, n):
    # the Crystal Ball with its power-law tail on the low side (scipy's convention), not normalised
    z = (np.asarray(x, float) - mu) / sigma
    a = abs(alpha)
    A = (n / a) ** n * np.exp(-0.5 * a * a)
    B = n / a - a
    out = np.exp(-0.5 * z * z)
    tail = z <= -a
    out[tail] = A * (B - z[tail]) ** (-n)
    return out


def _axes_arrows(ax, x0, y0, w, h, xlabel, ylabel, fs=FS):
    # two drawn axes with arrow heads and their labels
    _arrow(ax, (x0, y0), (x0 + w + 0.12, y0), lw=0.8, ms=8)
    _arrow(ax, (x0, y0), (x0, y0 + h + 0.12), lw=0.8, ms=8)
    ax.text(x0 + w / 2, y0 - 0.12, xlabel, ha='center', va='top', fontsize=fs)
    ax.text(x0 - 0.12, y0 + h / 2, ylabel, ha='right', va='center', rotation=90, fontsize=fs)


# ---------------------------------------------------------------------------------------------
def series_chain(current, *, ax=None):
    """The notebooks 00 to 08 and what they pass on, as boxes: the geometry and the steering file,
    ddsim, the ROOT file, then three branches (01 and 01b; 02 and its .npz, then 03 and 04; 05, then
    06 and 07, 08 and the reticle chain of 09 to 12). Under each box the file or notebook name; the
    box of notebook `current` ('00' to '08', or '01b') filled."""
    boxes = {   # key: (column, row, text in the box, name under it)
        'geo': (0, 1, "geometry XML +\nsteering file", "my_custom_ecal.xml\nsim/run_sim.py"),
        'sim': (1, 1, "ddsim", "notebook 00"),
        'root': (2, 1, "the ROOT file\n(EDM4hep)", "sim_*.root"),
        '01': (3, 2, "01 and 01b:\nthe detector\nand the data", "01_detector_and_data\n01b_..._pions"),
        '02': (3, 1, "02 and\nthe .npz", "02_data_extraction\ndecal_extracted_data_*.npz"),
        '03': (4, 1, "03 and 04:\nresolution", "03_resolution_conventional\n04_resolution_ml_crystalball"),
        '05': (3, 0, "05:\nthe cascade", "05_shower_4vectors"),
        '06': (4, 0, "06 and 07:\nsensor crossings", "06_sensor_crossings_tracker\n07_sensor_crossings_calo"),
        '08': (5, 0, "08:\nthe PIXELAV deck", "08_pixelav_deck_inspection"),
        '09': (6, 0, "09 to 12:\nreticle chain", "09_ ... 12_"),
    }
    which = {'00': 'sim', '01': '01', '01b': '01', '02': '02', '03': '03', '04': '03', '05': '05', '06': '06',
             '07': '06', '08': '08'}
    if current not in which:
        raise ValueError(f"current must be one of {sorted(which)}, not {current!r}")
    mine = which[current]
    W, H, GX, GY = 2.15, 0.95, 2.75, 2.1
    fig, ax = _axes(ax, (13, 5.0))
    pos = {k: (c * GX, r * GY) for k, (c, r, _, _) in boxes.items()}
    for k, (c, r, text, name) in boxes.items():
        x, y = pos[k]
        me = k == mine
        _box(ax, x, y, W, H, text, fc=LBLUE if me else 'white', lw=1.3 if me else LW, fs=8.5,
             weight='bold' if me else 'normal')
        ax.text(x + W / 2, y - 0.08, name, ha='center', va='top', fontsize=6.5, family='monospace', color='0.3')
        if me:
            ax.text(x + W / 2, y + H + 0.08, "this notebook", ha='center', va='bottom', fontsize=FS,
                    fontweight='bold', color=BLUE)
    links = [('geo', 'sim'), ('sim', 'root'), ('root', '01'), ('root', '02'), ('02', '03'), ('root', '05'),
             ('05', '06'), ('06', '08'), ('08', '09')]
    for a, b in links:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        _arrow(ax, (xa + W + 0.04, ya + H / 2), (xb - 0.05, yb + H / 2), lw=0.9, ms=9)
    ax.set_xlim(-0.15, 6 * GX + W + 0.15)
    ax.set_ylim(-0.75, 2 * GY + H + 0.45)
    return fig


# ---------------------------------------------------------------------------------------------
def barrel_face(rmin_mm, rmax_mm, n_faces, seg_half_deg, *, depth_window=None, ax=None):
    """Left: the barrel end-on, n_faces flat faces between the radii rmin_mm and rmax_mm (measured
    to the middle of a face), z out of the page, the beam along +y onto the centre of the +y face
    (shaded), and the wedge of the extraction: the two rays at plus and minus seg_half_deg from +y.
    Right: the +y face, enlarged, where depth = y. With depth_window = (ymin, ymax) the accepted
    region, the wedge between those two depths, is shaded."""
    r0, r1, h = float(rmin_mm), float(rmax_mm), np.radians(seg_half_deg)
    half = np.pi / n_faces
    fig, ax = _axes(ax, (13, 5.0))
    c = 1 / np.cos(half)

    def face(phi, ra, rb):
        return [(r * c * np.cos(a), r * c * np.sin(a)) for r, a in
                ((ra, phi - half), (rb, phi - half), (rb, phi + half), (ra, phi + half))]

    # (a) the barrel end-on, in mm
    for k in range(n_faces):
        phi = np.pi / 2 + 2 * k * half
        ax.add_patch(Polygon(face(phi, r0, r1), closed=True, fc=LBLUE if k == 0 else LIGHT, ec=INK, lw=LW))
    ax.add_patch(Circle((0, 0), 0.035 * r1, fc='white', ec=INK, lw=LW))
    ax.plot(0, 0, '.', color=INK, ms=3)
    ax.text(0.06 * r1, -0.05 * r1, "z, out of the page", ha='left', va='top', fontsize=SMALL)
    _arrow(ax, (0, 0.05 * r1), (0, r0), color=ORANGE, lw=1.8, ms=12)
    ax.text(-0.03 * r1, 0.5 * r0, "the beam,\nalong +y", color=ORANGE, fontsize=SMALL, ha='right', va='center')
    for s in (-1, 1):
        ax.plot([0, s * 1.12 * r1 * np.sin(h)], [0, 1.12 * r1 * np.cos(h)], color=RED, lw=0.9, ls='--')
    ax.text(0, 1.14 * r1, f"the wedge: within {seg_half_deg:g} degrees of +y", ha='center', va='bottom',
            fontsize=SMALL, color=RED)
    _arrow(ax, (-1.2 * r1, -1.2 * r1), (-0.95 * r1, -1.2 * r1), lw=0.9, ms=8)
    _arrow(ax, (-1.2 * r1, -1.2 * r1), (-1.2 * r1, -0.95 * r1), lw=0.9, ms=8)
    ax.text(-0.93 * r1, -1.2 * r1, "x", fontsize=SMALL, va='center')
    ax.text(-1.2 * r1, -0.93 * r1, "y", fontsize=SMALL, ha='center', va='bottom')
    ax.text(0, -r1 * c - 0.08 * r1, f"the barrel end-on: {n_faces} flat faces", ha='center', va='top', fontsize=FS)
    # (b) the +y face enlarged: depth = y
    X0, s = 1.75 * r1, 4.0                                # the left edge of (b), its magnification
    yb0 = r0 - 0.12 * (r1 - r0)                           # the bottom of the zoom, in mm

    def zp(x, y):                                         # mm of the face to drawing units of (b)
        return X0 + s * (np.asarray(x) + r1 * np.tan(half) * 1.2), -0.25 * r1 + s * (np.asarray(y) - yb0)

    corners = np.array(face(np.pi / 2, r0, r1))
    ax.add_patch(Polygon(np.c_[zp(corners[:, 0], corners[:, 1])], closed=True, fc=LBLUE, ec=INK, lw=LW))
    if depth_window is not None:
        ya, yb = (float(v) for v in depth_window)
        quad = [(-ya * np.tan(h), ya), (-yb * np.tan(h), yb), (yb * np.tan(h), yb), (ya * np.tan(h), ya)]
        q = np.array(quad)
        ax.add_patch(Polygon(np.c_[zp(q[:, 0], q[:, 1])], closed=True, fc=LORANGE, ec=ORANGE, lw=1.0, alpha=0.75))
        for yv, va in ((ya, 'top'), (yb, 'bottom')):
            px, py = zp(yv * np.tan(h), yv)
            ax.text(px + 0.03 * r1, py, f"y = {yv:g} mm", fontsize=SMALL, ha='left', va=va, color=ORANGE)
        px, py = zp(0, 0.5 * (ya + yb))
        ax.text(px, py, "accepted: in the wedge\nand between the two depths", ha='center', va='center',
                fontsize=SMALL, color='0.2')
    y_hi = r1 + 0.12 * (r1 - r0)
    for sg in (-1, 1):
        px, py = zp([sg * yb0 * np.tan(h), sg * y_hi * np.tan(h)], [yb0, y_hi])
        ax.plot(px, py, color=RED, lw=0.9, ls='--')
    px, py = zp(0, yb0)
    _arrow(ax, (px, py), tuple(zp(0, r0)), color=ORANGE, lw=1.8, ms=12)
    for yv, lab in ((r0, f"rmin = {r0:g} mm"), (r1, f"rmax = {r1:g} mm")):
        px, py = zp(-r1 * np.tan(half), yv)
        ax.text(px - 0.03 * r1, py, lab, ha='right', va='center', fontsize=SMALL)
    px, py = zp(r1 * np.tan(half) * 1.05, 0.5 * (r0 + r1))
    xd = px + (0.75 if depth_window is not None else 0.12) * r1
    _arrow(ax, (xd, py - 0.25 * r1), (xd, py + 0.25 * r1), lw=1.0, ms=9)
    ax.text(xd + 0.05 * r1, py, "depth = y\non this face", ha='left', va='center', fontsize=FS)
    px, py = zp(0, y_hi)
    ax.text(px, py + 0.05 * r1, "the +y face, enlarged", ha='center', va='bottom', fontsize=FS)
    for p0 in ((r1 * c * np.cos(np.pi / 2 - half), r1 * c * np.sin(np.pi / 2 - half)),):
        ax.plot([p0[0], zp(r1 * np.tan(half), r1)[0]], [p0[1], zp(r1 * np.tan(half), r1)[1]], color=GREY, lw=0.6,
                ls=':')
    ax.set_xlim(-1.3 * r1, X0 + s * 2.4 * r1 * np.tan(half) + 1.4 * r1)
    ax.set_ylim(-r1 * c - 0.3 * r1, 1.3 * r1)
    return fig


# ---------------------------------------------------------------------------------------------
def four_readouts(mip_energy_kev, threshold_kev, *, ax=None):
    """One made-up patch of pixel energies (a sketch) read four ways with the rules of the
    extraction: the sum of every pixel's energy (analog); max(1, round(E / MIP)) in each pixel above
    the threshold, summed (MIP counting); the pixels above the threshold, counted (hits); the
    8-connected groups of those pixels, counted (clusters). The counts and groups are computed here
    from the made-up energies."""
    mip, thr = float(mip_energy_kev), float(threshold_kev)
    f = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],     # energies in units of the MIP, made up
                  [0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 2.6, 0.8, 0.0, 0.0, 0.0, 1.3],
                  [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.9],
                  [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 3.4, 1.0, 0.0]])
    E = f * mip
    fired = E > thr
    mips = np.where(fired, np.maximum(1, np.round(E / mip)), 0).astype(int)
    lab = np.zeros(E.shape, int)                           # 8-connected groups of the fired pixels
    n_grp = 0
    for r0, c0 in zip(*np.nonzero(fired)):
        if lab[r0, c0]:
            continue
        n_grp += 1
        stack = [(r0, c0)]
        while stack:
            r, cc = stack.pop()
            if not (0 <= r < E.shape[0] and 0 <= cc < E.shape[1]) or not fired[r, cc] or lab[r, cc]:
                continue
            lab[r, cc] = n_grp
            stack += [(r + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    ny, nx = E.shape
    fig, ax = _axes(ax, (13, 3.9))
    cw, G = 0.5, 1.15
    x_of = lambda k: k * (nx * cw + G)
    heads = [("analog (all_visible)", f"every pixel's energy, summed:\n{E.sum():.1f} keV"),
             ("MIP counting (all_mip)", f"max(1, round(E / {mip:g} keV)) in each\npixel above threshold: "
                                        f"{mips.sum()} MIPs"),
             ("hits (all_hits)", f"the pixels above {thr:g} keV:\n{int(fired.sum())} hits"),
             ("clusters (all_cluster)", f"8-connected groups of those\npixels: {n_grp} clusters")]
    cols = [BLUE, ORANGE, GREEN, PURPLE, RED]
    for k, (head, sub) in enumerate(heads):
        x0 = x_of(k)
        for r in range(ny):
            for cc in range(nx):
                x, y = x0 + cc * cw, (ny - 1 - r) * cw
                e = E[r, cc]
                if k == 0:
                    fc = LBLUE if e > 0 else 'white'
                elif k == 1:
                    fc = LORANGE if fired[r, cc] else 'white'
                else:
                    fc = INK if fired[r, cc] else 'white'
                ax.add_patch(Rectangle((x, y), cw, cw, fc=fc, ec='0.75', lw=0.5))
                if k == 0 and e > 0:
                    ax.text(x + cw / 2, y + cw / 2, f"{e:.0f}", ha='center', va='center', fontsize=6.5,
                            color=INK if e > thr else '0.45')
                if k == 1 and fired[r, cc]:
                    ax.text(x + cw / 2, y + cw / 2, f"{mips[r, cc]}", ha='center', va='center', fontsize=SMALL)
        if k == 3:
            for g in range(1, n_grp + 1):
                rr, cc_ = np.nonzero(lab == g)
                _box(ax, x0 + (cc_.min() - 0.18) * cw, (ny - 1 - rr.max() - 0.18) * cw,
                     (cc_.max() - cc_.min() + 1.36) * cw, (rr.max() - rr.min() + 1.36) * cw, fc='none',
                     ec=cols[(g - 1) % len(cols)], lw=1.3, ls='--', rs=0.2 * cw)
        ax.add_patch(Rectangle((x0, 0), nx * cw, ny * cw, fill=False, ec=INK, lw=0.9))
        ax.text(x0 + nx * cw / 2, ny * cw + 0.62, head, ha='center', va='bottom', fontsize=FS, fontweight='bold')
        ax.text(x0 + nx * cw / 2, ny * cw + 0.08, sub, ha='center', va='bottom', fontsize=SMALL)
    ax.text(x_of(0) + nx * cw / 2, -0.1, "energies in keV; grey: below\nthe threshold, kept by analog only",
            ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.text((x_of(3) + nx * cw) / 2, -0.75, f"a sketch: the energies are made up; the MIP is {mip:g} keV and the "
            f"threshold {thr:g} keV, the rules are the extraction's", ha='center', va='top', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-0.1, x_of(3) + nx * cw + 0.1)
    ax.set_ylim(-1.05, ny * cw + 1.1)
    return fig


# ---------------------------------------------------------------------------------------------
def pion_regimes(n_layers, *, ax=None):
    """Two copies of the stack of n_layers layers, a pion coming in from the left. Left: a straight
    track through every layer, one MIP in each (punch-through). Right: a track that interacts with a
    nucleus at some depth and opens into a spray (nuclear interaction). The tracks are made up."""
    fig, ax = _axes(ax, (13, 3.9))
    L, Hh, G = 6.0, 2.0, 2.2                            # the length and height of a stack, the gap
    dx = L / n_layers
    for k in range(2):
        x0 = k * (L + G)
        for i in range(n_layers):
            ax.add_patch(Rectangle((x0 + i * dx, 0), 0.62 * dx, Hh, fc='0.62', ec='none'))
            ax.add_patch(Rectangle((x0 + (i + 0.72) * dx, 0), 0.2 * dx, Hh, fc=BLUE, ec='none'))
        _arrow(ax, (x0 - 1.0, 0.5 * Hh), (x0 - 0.05, 0.5 * Hh), color=ORANGE, lw=1.8, ms=12)
        ax.text(x0 - 0.55, 0.5 * Hh + 0.1, "a pion", color=ORANGE, fontsize=SMALL, ha='center', va='bottom')
    # left: straight through, one MIP per layer
    ax.plot([0, L + 0.6], [0.5 * Hh] * 2, color=INK, lw=1.5)
    _arrow(ax, (L + 0.3, 0.5 * Hh), (L + 0.7, 0.5 * Hh), lw=1.5, ms=10)
    for i in range(n_layers):
        ax.plot(((i + 0.82) * dx), 0.5 * Hh, 'o', ms=3, color=ORANGE, zorder=3)
    ax.text(L / 2, -0.15, "punch-through: one MIP per layer", ha='center', va='top', fontsize=FS, fontweight='bold')
    # right: a nuclear interaction at a depth, then a spray
    x0 = L + G
    xi = x0 + 0.33 * L
    ax.plot([x0, xi], [0.5 * Hh] * 2, color=INK, lw=1.5)
    ax.plot(xi, 0.5 * Hh, '*', ms=11, color=RED, zorder=4)
    rng = np.random.default_rng(5)
    for ang in np.radians([-38, -24, -12, -4, 6, 15, 27, 40]):
        r = rng.uniform(0.45, 1.0) * (x0 + L - xi + 0.3)
        e = (xi + r * np.cos(ang), 0.5 * Hh + r * np.sin(ang) * 0.9)
        e = (min(e[0], x0 + L + 0.5), float(np.clip(e[1], 0.05, Hh - 0.05)))
        ax.plot([xi, e[0]], [0.5 * Hh, e[1]], color=PURPLE, lw=1.0)
    ax.text(xi, 0.5 * Hh + 0.62, "it hits a nucleus", color=RED, fontsize=SMALL, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
    ax.text(x0 + L / 2, -0.15, "nuclear interaction: a spray", ha='center', va='top', fontsize=FS,
            fontweight='bold')
    ax.text((x0 + L) / 2, Hh + 0.35, "the stack is 1.00 lambda_I (notebook 01), so about e^-1 = 37% "
            "of pions cross it untouched", ha='center', va='bottom', fontsize=FS)
    ax.text((x0 + L) / 2, -0.7, f"{n_layers} layers of tungsten and silicon, not to scale; a sketch: "
            "the tracks are made up", ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(-1.2, x0 + L + 0.8)
    ax.set_ylim(-1.05, Hh + 0.8)
    return fig


# ---------------------------------------------------------------------------------------------
def inversion(*, ax=None):
    """A sketch of turning a readout band into an energy interval, on linear axes with made-up
    curves. Left: a straight calibration line. Right: one that bends over at high energy. The same
    readout band of width sigma on each, projected down to the energy axis: narrow under the line,
    wide under the bend, where the curve is flat."""
    fig, ax = _axes(ax, (12, 4.3), equal=False)
    W, H, X2 = 4.0, 3.0, 5.7
    E = np.linspace(0, 1, 300)
    curves = (("a straight calibration", lambda e: 0.95 * e),
              ("a calibration that bends over", lambda e: 0.95 * (1 - np.exp(-4.0 * e)) / (1 - np.exp(-4.0))))
    y0, sg = 0.88, 0.035                                  # the readout and its band, as shares of the height
    for k, (title, fn) in enumerate(curves):
        x0 = k * X2
        _axes_arrows(ax, x0, 0, W, H, "true energy", "readout")
        ax.plot(x0 + W * E, H * fn(E), color=BLUE, lw=2.0)
        ax.add_patch(Rectangle((x0, H * (y0 - sg)), W, H * 2 * sg, fc=LORANGE, ec='none', alpha=0.8, zorder=0))
        lo, hi = (float(np.interp(v, fn(E), E)) for v in (y0 - sg, y0 + sg))
        for e_, v in ((lo, y0 - sg), (hi, y0 + sg)):
            ax.plot([x0 + W * e_] * 2, [0, H * v], color=ORANGE, lw=0.9, ls='--')
        ax.add_patch(Rectangle((x0 + W * lo, -0.06), W * (hi - lo), 0.12, fc=ORANGE, ec='none'))
        ax.text(x0 + W * (lo + hi) / 2, 0.12, "narrow" if k == 0 else "wide", ha='center', va='bottom',
                fontsize=SMALL, color=ORANGE, fontweight='bold')
        ax.text(x0 + W / 2, H + 0.25, title, ha='center', va='bottom', fontsize=FS)
    ax.text(0.08, H * (y0 + sg) + 0.06, "the same readout band, width sigma", fontsize=SMALL, color='0.25',
            va='bottom')
    ax.text(X2 + W + 0.05, H * 0.55, "where the curve is\nflat, the same band\nspans more energy",
            fontsize=SMALL, va='center', ha='left')
    ax.text((X2 + W) / 2, -0.6, "a sketch: made-up curves; projected down, the band gives the interval in energy",
            ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(-0.55, X2 + W + 1.4)
    ax.set_ylim(-0.85, H + 0.65)
    return fig


# ---------------------------------------------------------------------------------------------
def density_net(layer_sizes, activation, alpha_clip, n_clip, *, ax=None):
    """The density network: the true energy E goes into a network (its layer sizes and activation
    listed), four numbers come out (mu, sigma, alpha, n; alpha and n clipped to the ranges given),
    and they draw a Crystal Ball curve of the readout at that E. One observed readout is marked on
    the curve: its probability under the curve is the score of the fit. The curve's values are made
    up."""
    fig, ax = _axes(ax, (13, 4.0), equal=False)
    ax.add_patch(Circle((0.45, 2.0), 0.38, fc=LORANGE, ec=INK, lw=LW))
    ax.text(0.45, 2.0, "E", ha='center', va='center', fontsize=FS + 2)
    ax.text(0.45, 1.5, "the true\nenergy\n(normalised)", ha='center', va='top', fontsize=SMALL)
    _arrow(ax, (0.85, 2.0), (1.45, 2.0), lw=1.0)
    sizes = ", ".join(str(s) for s in layer_sizes)
    _box(ax, 1.5, 1.2, 2.6, 1.6, f"the network\n\nlayer sizes\n{sizes}\nactivation: {activation}", fc=LBLUE, fs=9)
    outs = (("mu", "the centre", None), ("sigma", "the core width", None),
            ("alpha", "where the tail starts", alpha_clip), ("n", "how slowly it falls", n_clip))
    for k, (nm, what, clip) in enumerate(outs):
        y = 3.25 - k * 0.95
        _arrow(ax, (4.15, 2.0), (4.75, y), lw=0.8, ms=8)
        _box(ax, 4.8, y - 0.27, 1.2, 0.54, nm, fc='white', fs=FS)
        txt = what if clip is None else f"{what}\nclipped to [{clip[0]:g}, {clip[1]:g}]"
        ax.text(6.1, y, txt, ha='left', va='center', fontsize=SMALL)
        _arrow(ax, (7.75, y), (8.35, 2.0), lw=0.8, ms=8)
    # the Crystal Ball at that E, and one observed readout
    x0, W, H = 8.7, 3.6, 2.7
    _axes_arrows(ax, x0, 0.6, W, H, "readout", "probability")
    r = np.linspace(-6, 3, 400)
    pdf = _cb_shape(r, 0.0, 1.0, 1.2, 3.0)
    to_x = lambda v: x0 + (v + 6) / 9 * W
    ax.plot(to_x(r), 0.6 + H * pdf / pdf.max(), color=BLUE, lw=2.0)
    ro = -1.6
    po = float(_cb_shape(np.array([ro]), 0.0, 1.0, 1.2, 3.0)[0])
    ax.plot([to_x(ro)] * 2, [0.6, 0.6 + H * po], color=ORANGE, lw=1.0, ls='--')
    ax.plot(to_x(ro), 0.6 + H * po, 'o', ms=6, color=ORANGE, zorder=4)
    _note(ax, "one observed readout:\nits probability under\nthe curve is the score", (to_x(ro), 0.6 + H * po),
          (x0 + 0.1, 0.6 + 0.92 * H), ha='left', color=ORANGE)
    ax.text(x0 + W / 2, 0.6 + H + 0.25, "a Crystal Ball curve at this E", ha='center', va='bottom', fontsize=FS)
    ax.text(x0 + W / 2, 0.1, "the curve's values are made up", ha='center', va='top', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-0.1, x0 + W + 0.4)
    ax.set_ylim(-0.25, 3.9)
    return fig


# ---------------------------------------------------------------------------------------------
def two_widths(*, ax=None):
    """A sketch: one Crystal Ball curve with made-up parameters read two ways. Its core width sigma,
    marked on the Gaussian side; and P16 and P84, the readouts below which 16% and 84% of the whole
    curve lie, the tail included (shaded where it starts, below mu - alpha sigma)."""
    mu, sg, al, n = 0.0, 1.0, 1.0, 2.5
    fig, ax = _axes(ax, (11, 4.0), equal=False)
    r = np.linspace(-9, 3.5, 4000)
    pdf = _cb_shape(r, mu, sg, al, n)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    p16, p84 = (float(np.interp(q, cdf, r)) for q in (0.16, 0.84))
    W, H = 8.0, 2.8
    to_x = lambda v: (v - r[0]) / (r[-1] - r[0]) * W
    y = H * pdf / pdf.max()
    tail = r <= mu - al * sg
    ax.fill_between(to_x(r[tail]), 0, y[tail], color=LRED, lw=0)
    ax.plot(to_x(r), y, color=BLUE, lw=2.0)
    _arrow(ax, (0, 0), (W + 0.2, 0), lw=0.8, ms=8)
    ax.text(W / 2, -0.12, "readout", ha='center', va='top', fontsize=FS)
    yh = H * np.exp(-0.5)
    _dim(ax, (to_x(mu), yh), (to_x(mu + sg), yh), "", (0, 0))
    ax.text(to_x(mu + sg) + 0.1, yh, "sigma: the width\nof the core", ha='left', va='center', fontsize=SMALL,
            color=BLUE)
    ax.plot([to_x(mu)] * 2, [0, H], color=GREY, lw=0.6, ls=':')
    for v, nm in ((p16, "P16"), (p84, "P84")):
        ax.plot([to_x(v)] * 2, [0, 1.12 * H], color=INK, lw=1.0, ls='--')
        ax.text(to_x(v), 1.14 * H, nm, ha='center', va='bottom', fontsize=FS)
    _dim(ax, (to_x(p16), 1.05 * H), (to_x(p84), 1.05 * H), "", (0, 0))
    ax.text(to_x(p16) - 0.12, 0.92 * H, "P16 to P84: the middle 68%\nof the whole curve, tail included",
            ha='right', va='center', fontsize=SMALL)
    ax.text(to_x(-4.2), 0.25 * H, "the tail", ha='center', va='bottom', fontsize=SMALL, color=RED)
    ax.text(W / 2, -0.55, "a sketch: made-up parameters; one fit read two ways", ha='center', va='top',
            fontsize=SMALL, color='0.35')
    ax.set_xlim(-0.3, W + 1.6)
    ax.set_ylim(-0.85, 1.32 * H)
    return fig


# ---------------------------------------------------------------------------------------------
def local_frame(*, ax=None):
    """One sensor plane in its own frame, following the definition of `to_local` in
    analysis/sensor_crossings.py: u across the pitch (tangential), v along the barrel axis
    (cylinder z), w along the outward normal (radial, the depth); on the +y face u = -x, v = z,
    w = y. Left: a track through the plane. Right: its projections on the (u, w) and (v, w) planes,
    with alpha and beta measured from the plane, cot alpha = p_u / p_w and cot beta = p_v / p_w.
    The track is made up."""
    fig, ax = _axes(ax, (13, 4.4))
    ob = np.array((0.55, 0.38))                           # v drawn going into the page, up and to the right
    P = lambda u, v, w: np.array((u, w)) + v * ob

    # the plane, spanned by u and v, at w = 0
    quad = np.array([P(0, 0, 0), P(4, 0, 0), P(4, 3, 0), P(0, 3, 0)])
    ax.add_patch(Polygon(quad, closed=True, fc=LBLUE, ec=INK, lw=LW, alpha=0.9))
    o = P(0, 0, 0)
    for d, nm in (((1.4, 0, 0), "u: across the pitch"), ((0, 1.9, 0), "v: along the barrel axis"),
                  ((0, 0, 1.6), "w: the outward normal")):
        e = P(*d)
        _arrow(ax, tuple(o), tuple(e), lw=1.4, ms=11)
        off = {0: (0.05, -0.22), 1: (0.12, 0.0), 2: (-0.1, 0.1)}[[i for i in range(3) if d[i]][0]]
        ax.text(e[0] + off[0], e[1] + off[1], nm, fontsize=SMALL, ha='right' if d[2] else 'left', va='center')
    pu, pv, pw = 0.9, 1.1, 1.6                            # the track's direction, made up
    c = np.array((2.4, 1.4, 0.0))
    a, b = c - 0.9 * np.array((pu, pv, pw)), c + 0.9 * np.array((pu, pv, pw))
    ax.plot(*zip(P(*a), P(*c)), color=ORANGE, lw=1.6, ls=':')
    ax.plot(*zip(P(*c), P(*b)), color=ORANGE, lw=1.6)
    _arrow(ax, tuple(P(*(c + 0.7 * (b - c)))), tuple(P(*b)), color=ORANGE, lw=1.6, ms=11)
    ax.plot(*P(*c), 'o', ms=4, color=ORANGE)
    ax.text(*(P(*b) + np.array((-0.15, 0.05))), "a track,\nmomentum (p_u, p_v, p_w)", fontsize=SMALL,
            color=ORANGE, va='center', ha='right')
    ax.text(*(P(2, 0, 0) + np.array((0, -0.5))), "one sensor plane", ha='center', va='top', fontsize=FS)

    # the two projections
    def proj(x0, p_side, side, ang):
        _arrow(ax, (x0, 0), (x0 + 2.3, 0), lw=1.0, ms=9)
        _arrow(ax, (x0, 0), (x0, 2.3), lw=1.0, ms=9)
        ax.text(x0 + 2.35, 0, side, fontsize=FS, va='center')
        ax.text(x0, 2.35, "w", fontsize=FS, ha='center', va='bottom')
        L = 2.0 / np.hypot(p_side, pw)
        _arrow(ax, (x0, 0), (x0 + L * p_side, L * pw), color=ORANGE, lw=1.6, ms=11)
        th = np.degrees(np.arctan2(pw, p_side))
        t = np.radians(np.linspace(0, th, 40))
        ax.plot(x0 + 0.7 * np.cos(t), 0.7 * np.sin(t), color=INK, lw=0.8)
        ax.text(x0 + 0.85 * np.cos(np.radians(th / 2)), 0.85 * np.sin(np.radians(th / 2)), ang, fontsize=FS)
        ax.text(x0 + 1.15, -0.25, f"cot {ang} = p_{side} / p_w", ha='center', va='top', fontsize=FS)
        ax.text(x0 + 1.15, 2.7, f"seen on the ({side}, w) plane", ha='center', va='bottom', fontsize=SMALL)

    proj(6.6, pu, "u", "alpha")
    proj(10.4, pv, "v", "beta")
    ax.text(10.4, -0.85, "alpha and beta are measured from the plane; a track along w has both cot = 0",
            ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(-1.6, 13.4)
    ax.set_ylim(-1.2, 3.3)
    return fig


# ---------------------------------------------------------------------------------------------
def steps_to_crossings(layer_centers_mm, thickness_mm, *, ax=None):
    """One particle's Geant4 steps in a few silicon layers, time against depth. Each layer is a band
    of thickness_mm around its centre (the gaps between the bands not to scale; the depths written
    are the layers' own). The steps in one layer in a row of time form a run, one crossing; the
    earliest step of a run is its entry. The last run goes inward: its deepest point is where it
    came in, so time order, not depth order, fixes the direction. The steps are made up."""
    zc = [float(z) for z in layer_centers_mm]
    t = float(thickness_mm)
    nl = len(zc)
    BW, GAP = 1.0, 1.6                                    # a band's drawn width and the drawn gap
    xb = lambda k, f: k * (BW + GAP) + f * BW             # layer k, fraction f across its thickness
    fig, ax = _axes(ax, (12, 4.3), equal=False)
    TOP = 5.2
    for k in range(nl):
        ax.add_patch(Rectangle((xb(k, 0), 0), BW, TOP, fc='#EEF2F8', ec=BLUE, lw=0.9))
        ax.text(xb(k, 0.5), TOP + 0.08, f"layer {k}", ha='center', va='bottom', fontsize=SMALL)
        ax.text(xb(k, 0), -0.1, f"{zc[k] - t / 2:.2f}", ha='center', va='top', fontsize=7, rotation=90)
        ax.text(xb(k, 1), -0.1, f"{zc[k] + t / 2:.2f}", ha='center', va='top', fontsize=7, rotation=90)
        if k < nl - 1:
            ax.text(xb(k, 1) + GAP / 2, 0.15, "tungsten", ha='center', va='bottom', fontsize=7, color='0.45')
    # the runs: (layer, fractions across the layer in time order, the time of the first step)
    runs = [(0, (0.1, 0.45, 0.85), 0.4), (1, (0.05, 0.5, 0.95), 1.4), (2, (0.08, 0.4, 0.78), 2.4)]
    if nl >= 3:
        runs.append((2, (0.9, 0.55, 0.15), 3.9))            # it turned back behind layer 2: inward
    path = []
    for n, (k, fs, t0) in enumerate(runs):
        pts = [(xb(k, f), t0 + 0.22 * i) for i, f in enumerate(fs)]
        if fs[-1] < fs[0] and path:                       # it went on into the tungsten and turned there
            path.append((xb(k, 1) + 0.55 * GAP, 0.5 * (path[-1][1] + t0)))
        path += pts
        xs_, ys_ = zip(*pts)
        inward = fs[-1] < fs[0]
        col = RED if inward else INK
        ax.plot(xs_, ys_, 'o', ms=5, color=col, zorder=4)
        ax.add_patch(Circle(pts[0], 0.17, fill=False, ec=ORANGE, lw=1.6, zorder=5))
        _box(ax, xb(k, 0) - 0.12, t0 - 0.25, BW + 0.24, 0.22 * (len(fs) - 1) + 0.5, fc='none', ec=GREY, lw=0.8,
             ls='--', rs=0.1)
        side = xb(k, 1) + 0.18
        ax.text(side, t0 + 0.2, f"run {n + 1}" + (": inward" if inward else ""), fontsize=SMALL, va='center',
                color=col)
    px, py = zip(*path)
    ax.plot(px, py, color=GREY, lw=0.7, zorder=2)
    k, fs, t0 = runs[-1]
    if fs[-1] < fs[0]:
        _note(ax, "its entry: the earliest step,\nat the outer face", (xb(k, fs[0]) + 0.17, t0),
              (xb(k, 1) + 0.25, t0 - 1.0), color=ORANGE)
        _note(ax, "by depth this would be\nthe entry: wrong", (xb(k, fs[-1]), t0 + 0.44),
              (xb(k, 0) - 1.4, t0 + 0.95), ha='center', color=RED)
    ax.plot([], [], 'o', ms=5, color=INK, label='a step in the silicon')
    ax.plot([], [], 'o', ms=8, mfc='none', mec=ORANGE, mew=1.6, label='the entry of a run:\nits earliest step')
    ax.legend(loc='upper left', bbox_to_anchor=(xb(0, 1) + 0.05, TOP), bbox_transform=ax.transData, fontsize=SMALL,
              frameon=False, handletextpad=0.3)
    _arrow(ax, (-0.5, 0), (-0.5, TOP), lw=0.8, ms=8)
    ax.text(-0.6, TOP / 2, "time", rotation=90, ha='right', va='center', fontsize=FS)
    ax.text(xb(nl - 1, 1) / 2, -0.75, f"depth y [mm]: each band is one layer's {t * 1000:g} um of silicon, "
            "the gaps not to scale", ha='center', va='top', fontsize=SMALL)
    ax.text(xb(nl - 1, 1) / 2, -1.15, "a sketch: the steps are made up", ha='center', va='top', fontsize=SMALL,
            color='0.35')
    ax.set_xlim(-1.0, xb(nl - 1, 1) + 1.6)
    ax.set_ylim(-1.45, TOP + 0.45)
    return fig


# ---------------------------------------------------------------------------------------------
def drop_reasons(thickness_um, max_cot, path_tolerance, *, ax=None):
    """Five small sections of the slab, one per reason a crossing is kept out of a PIXELAV deck, in
    the order the converter tests them: a track lying in the plane (grazing); |cot alpha| or |cot
    beta| above max_cot, a track far from the normal (steep: the driver makes no cluster); a point outside the slab (outside_si); a
    path shorter than the straight chord by more than the tolerance, a track that starts or stops in
    the silicon or whose momentum points more steeply than its line (path_short); a path longer than
    the chord by more than the tolerance, a curling track (path_long). The tracks are made up; the
    slab is drawn thicker than it is."""
    fig, ax = _axes(ax, (13.5, 3.0))
    W, T, G = 2.5, 0.8, 0.45                              # a slab's drawn length and thickness, the gap
    tol = float(path_tolerance)
    titles = [("a track in the plane", "(grazing)"),
              (f"|cot alpha| or |cot beta| > {max_cot:g}", "(steep)"),
              ("a point outside the slab", "(outside_si)"),
              (f"path < (1 - {tol:g}) x chord", "(path_short)"),
              (f"path > (1 + {tol:g}) x chord", "(path_long)")]
    for k, (t1, t2) in enumerate(titles):
        x0 = k * (W + G)
        ax.add_patch(Rectangle((x0, 0), W, T, fc='#EEF2F8', ec=INK, lw=LW))
        ax.text(x0 + W / 2, T + 0.42, t1, ha='center', va='bottom', fontsize=SMALL)
        ax.text(x0 + W / 2, T + 0.12, t2, ha='center', va='bottom', fontsize=FS, family='monospace')
        if k == 0:                                        # along the plane: p_w = 0, cot is not finite
            ax.plot([x0 + 0.2, x0 + W - 0.2], [0.5 * T] * 2, color=ORANGE, lw=1.6)
            _arrow(ax, (x0 + W - 0.5, 0.5 * T), (x0 + W - 0.2, 0.5 * T), color=ORANGE, lw=1.6, ms=10)
            ax.text(x0 + W / 2, -0.12, "p_w = 0: no cot", ha='center', va='top', fontsize=SMALL)
        elif k == 1:                                      # |cot| beyond the limit: draws as nearly flat
            cot = 1.5 * max_cot
            run = T * cot                                 # sideways per slab thickness, in drawing units
            xa = x0 + 0.1
            ax.plot([xa, x0 + W - 0.1], [0, (W - 0.2) * T / run], color=ORANGE, lw=1.6)
            ax.text(x0 + W / 2, -0.12, f"|cot| = {cot:g}: {cot:g} slab thicknesses\nsideways per one crossed",
                    ha='center', va='top', fontsize=SMALL)
        elif k == 2:                                      # the recorded position is not in the silicon
            ax.plot([x0 + 0.9, x0 + 1.6], [-0.35, T + 0.25], color=ORANGE, lw=1.2, ls=':')
            ax.plot(x0 + 1.75, -0.22, 'X', ms=8, color=RED, zorder=4)
            ax.text(x0 + W / 2, -0.32, "the recorded point lies\noutside its slab", ha='center', va='top',
                    fontsize=SMALL)
        elif k == 3:                                      # it starts inside: shorter than the chord
            xs_, ys_ = x0 + 1.05, 0.45 * T
            ax.plot([x0 + 0.85, x0 + 1.45], [0, T], color=GREY, lw=1.0, ls='--')
            ax.plot([xs_, x0 + 1.45], [ys_, T], color=ORANGE, lw=1.6)
            ax.plot(xs_, ys_, 'o', ms=5, color=ORANGE)
            ax.text(x0 + W / 2, -0.12, "it starts (or stops) in the silicon;\ndashed: the chord", ha='center',
                    va='top', fontsize=SMALL)
        else:                                             # it curls: longer than the chord
            ax.plot([x0 + 0.8, x0 + 1.5], [0, T], color=GREY, lw=1.0, ls='--')
            g = np.linspace(0, 1, 200)
            cx = x0 + 0.8 + 0.7 * g + 0.28 * np.sin(4 * np.pi * g)
            cy = T * g + 0.12 * np.sin(4 * np.pi * g + 1.2) * np.sin(np.pi * g)
            ax.plot(cx, np.clip(cy, 0, T), color=ORANGE, lw=1.4)
            ax.text(x0 + W / 2, -0.12, "a curling track (scattering, a\nsoft electron); dashed: the chord",
                    ha='center', va='top', fontsize=SMALL)
    ax.text((5 * W + 4 * G) / 2, -0.9, f"each panel: the {thickness_um:g} um slab in section, drawn thicker "
            "than it is; chord = thickness x sqrt(1 + cot alpha^2 + cot beta^2); a sketch: the tracks are made up",
            ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(-0.1, 5 * W + 4 * G + 0.1)
    ax.set_ylim(-1.25, T + 0.85)
    return fig


# ---------------------------------------------------------------------------------------------
def deck_window(nx, ny, *, ax=None):
    """The sensor plane with v to the right and u up, and PIXELAV's window of nx x ny pixels over it,
    the window's x along v and its y along u, centred on the pixel the crossing goes through. The
    mid-plane point (modx along v, mody along u) is marked; from it cot beta steers the track along
    x (v) and cot alpha along y (u). Pixels drawn square; the point is made up."""
    fig, ax = _axes(ax, (12, 5.0))
    cw = 0.28
    mx, my = nx + 10, ny + 8                              # the plane drawn around the window, in pixels
    _grid(ax, 0, 0, mx, my, cw, cw, color='0.88', lw=0.5)
    ax.add_patch(Rectangle((0, 0), mx * cw, my * cw, fill=False, ec=GREY, lw=0.8))
    i0, j0 = 5, 4                                         # the window's first pixel on the plane
    ax.add_patch(Rectangle((i0 * cw, j0 * cw), nx * cw, ny * cw, fc=LBLUE, ec=BLUE, lw=1.4, alpha=0.8))
    _grid(ax, i0 * cw, j0 * cw, nx, ny, cw, cw, color='white', lw=0.5)
    ic, jc = i0 + nx // 2, j0 + ny // 2
    ax.add_patch(Rectangle((ic * cw, jc * cw), cw, cw, fc='white', ec=INK, lw=0.9))
    p = np.array(((ic + 0.62) * cw, (jc + 0.35) * cw))
    ax.plot(*p, 'o', ms=5, color=ORANGE, zorder=5)
    cb, ca = 2.6, 1.4                                      # a direction, made up, in pixels per thickness
    _arrow(ax, tuple(p), (p[0] + cb * cw * 2, p[1]), color=BLUE, lw=1.3, ms=10)
    _arrow(ax, tuple(p), (p[0], p[1] + ca * cw * 2), color=RED, lw=1.3, ms=10)
    _arrow(ax, tuple(p), (p[0] + cb * cw * 2, p[1] + ca * cw * 2), color=ORANGE, lw=1.6, ms=11)
    xr = (mx + 0.6) * cw
    _note(ax, "the mid-plane point (modx, mody):\nmodx along v, mody along u", tuple(p), (xr, (jc - 5.5) * cw))
    _note(ax, "cot beta steers the track\nalong the window's x (v)", (p[0] + cb * cw * 1.6, p[1]),
          (xr, (jc - 1) * cw), color=BLUE)
    _note(ax, "cot alpha steers it\nalong the window's y (u)", (p[0], p[1] + ca * cw * 1.0),
          (xr, (jc + 2) * cw), color=RED)
    _note(ax, "the track, projected", (p[0] + cb * cw * 1.7, p[1] + ca * cw * 1.7), (xr, (jc + 6) * cw),
          color=ORANGE)
    ax.text((i0 + nx / 2) * cw, (j0 + ny) * cw + 0.1, f"PIXELAV's window: {nx} x {ny} pixels,\nits x along v, "
            "its y along u", ha='center', va='bottom', fontsize=FS, color=BLUE,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
    _arrow(ax, (0, -0.3), (2.2, -0.3), lw=1.0, ms=9)
    ax.text(2.3, -0.3, "v (the window's x)", fontsize=SMALL, va='center')
    _arrow(ax, (-0.3, 0), (-0.3, 2.2), lw=1.0, ms=9)
    ax.text(-0.38, 1.1, "u (the window's y)", rotation=90, ha='right', va='center', fontsize=SMALL)
    ax.text(mx * cw / 2, -0.7, "the sensor plane; pixels drawn square; the point and the direction are made up",
            ha='center', va='top', fontsize=SMALL, color='0.35')
    ax.set_xlim(-0.9, xr + 3.4)
    ax.set_ylim(-1.05, my * cw + 0.2)
    return fig
