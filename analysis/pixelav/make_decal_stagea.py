#!/usr/bin/env python3
"""Generate a SIMPLE PIXELAV Stage-A sensor model for the CALOMAPS DECAL sensor.

DECAL sensor (from geometry/my_custom_ecal.xml): 320 um thick silicon, 100 um square pixel
pitch, 12-sided Si-W barrel. This writes the two Stage-A files PIXELAV reads:

  ppixel2.decal.init  -- header, params, and a 26x13x51 E-field grid (V/cm).
  wgt_pot.decal.init  -- the matching 3x3-per-node Ramo weighting potential grid.

SIMPLE-FIELD MODEL (authorized in lieu of a TCAD field map):
  * E-field: UNIFORM along -z (depth), E_z = -V_bias/thickness, E_x=E_y~0. An overdepleted
    planar sensor. PIXELAV forces the transverse field to ~0 at the x/y cell boundaries.
  * B = 0 (no Lorentz drift) -- the cleanest, best-defined choice; can be set to a solenoid
    value later.
  * Weighting potential: the exact Ramo weighting potential of one 100 um pixel electrode at the
    collection plane (z=thick), all other pixels + backplane grounded, solved analytically by 2-D
    FFT (each transverse Fourier mode decays as sinh(G z)/sinh(G thick); the DC mode -> z/thick).
    This is geometry-correct for our pitch/thickness (no rad damage, planar field).

Grid conventions (matched to fieldsav/wgtpotsav + pixinit in ppixelav2.c):
  node (ix,iy,iz) -> first-quadrant physical (xf,yf,z):
     xf = ix*(xsize/2)/(npixx-1),  yf = iy*(ysize/2)/(npixy-1),  z = iz*thick/(npixz-1)
  E-field file line:  ix iy iz Ex Ey Ez       (V/cm; pixinit multiplies by 100 -> V/m)
  weighting line:     ix iy iz W00..W22       (3x3; W[k][l] = weighting of pixel x-offset l-1,
                                               y-offset 1-k, at the first-quadrant point)
  write order is  for iz: for iy: for ix:  (matches pixinit's read loops)
"""
import numpy as np

# ---- DECAL sensor geometry + simple operating point -------------------------------------
THICK = 320.0          # um  (Si slice thickness, 0.032 cm)
XSIZE = 100.0          # um  (pixel pitch x = ECal_cell_size)
YSIZE = 100.0          # um  (pixel pitch y, square)
NPX, NPY, NPZ = 26, 13, 51      # E-field / weighting grid dims (<= NARRAYX/Y/Z = 26/13/51)
V_BIAS = 320.0         # V    -> uniform E_z = V_bias/thick = 1 V/um = 10000 V/cm
TEMP = 300.0           # K    (room-temperature test-beam operation)
BFIELD = (0.0, 0.0, 0.0)        # T   (no field: simplest well-defined model)
RHE, RHH = 1.0, 1.0    # Hall factors (irrelevant at B=0)
PEAKTIM, SAMPTIM = 20000.0, 4000.0   # ps  preamp CR-RC shaping (as bundled BPix)
EHOLE = 0              # collect electrons
NEW_DRDE = 1           # NIST ESTAR dE/dx
PIMOM = 1.0            # <1.1 -> PIXELAV default 45 GeV (per-track ppion overrides)
FILEBASE = 17000       # output file base index
HEADER = "DECAL_320um_100x100:simpleUniformE@-320V,B=0,300K,FFT-Ramo-wgt"

EZ_VCM = -V_BIAS / (THICK * 1e-4)    # thick um -> cm; E_z in V/cm  (= -10000)


def fft_weighting():
    """Ramo weighting potential w_central(x,y,z) of one pixel at z=thick (others+backplane=0),
    returned as a callable sampling function via per-z FFT planes + bilinear interpolation."""
    L = 16 * XSIZE                      # periodic box: 16 pitches (isolated-pixel approximation)
    Ng = 256                            # 6.25 um transverse resolution
    xs = (np.arange(Ng) - Ng // 2) * (L / Ng)     # centered grid coords (um)
    dx = L / Ng
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    f = ((np.abs(X) < XSIZE / 2) & (np.abs(Y) < YSIZE / 2)).astype(float)   # central electrode @ top
    F = np.fft.fft2(f)
    k = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    G = np.sqrt(KX**2 + KY**2)

    def plane(z):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            prop = np.sinh(G * z) / np.sinh(G * THICK)
        big = (G * THICK) > 30.0                 # stable large-G form: sinh ratio -> exp(G(z-thick))
        prop[big] = np.exp(G[big] * (z - THICK))
        prop[0, 0] = z / THICK                   # DC mode
        w = np.real(np.fft.ifft2(F * prop))
        return np.clip(w, 0.0, 1.0)

    def make_sampler(z):
        w = plane(z)
        def samp(xq, yq):                         # bilinear interp at arbitrary (xq,yq) arrays (um)
            fx = (np.asarray(xq) + L / 2) / dx
            fy = (np.asarray(yq) + L / 2) / dx
            i0 = np.clip(np.floor(fx).astype(int), 0, Ng - 2); j0 = np.clip(np.floor(fy).astype(int), 0, Ng - 2)
            tx = np.clip(fx - i0, 0, 1); ty = np.clip(fy - j0, 0, 1)
            return (w[i0, j0]*(1-tx)*(1-ty) + w[i0+1, j0]*tx*(1-ty)
                    + w[i0, j0+1]*(1-tx)*ty + w[i0+1, j0+1]*tx*ty)
        return samp
    return make_sampler


def write_efield(path):
    with open(path, "w") as fp:
        fp.write(HEADER + "\n")
        fp.write(f"{PIMOM:.2f} {FILEBASE}\n")
        fp.write(f"{BFIELD[0]:.3f} {BFIELD[1]:.3f} {BFIELD[2]:.3f}\n")
        fp.write(f"{THICK:.1f} {XSIZE:.1f} {YSIZE:.1f} {TEMP:.1f} 0.0 0.0 "
                 f"{RHE:.2f} {RHH:.2f} {PEAKTIM:.1f} {SAMPTIM:.1f} {EHOLE} {NEW_DRDE} {NPX} {NPY} {NPZ}\n")
        for iz in range(NPZ):
            for iy in range(NPY):
                for ix in range(NPX):
                    fp.write(f"{ix+1} {iy+1} {iz+1} {1e-6:.6e} {1e-6:.6e} {EZ_VCM:.6e}\n")


def write_weighting(path):
    make_sampler = fft_weighting()
    xf = np.arange(NPX) * (XSIZE / 2) / (NPX - 1)        # first-quadrant node coords (um)
    yf = np.arange(NPY) * (YSIZE / 2) / (NPY - 1)
    zf = np.arange(NPZ) * THICK / (NPZ - 1)
    central_mid = central_n = 0.0
    with open(path, "w") as fp:
        fp.write(f"{THICK:.1f} {XSIZE:.1f} {YSIZE:.1f} {NPX} {NPY} {NPZ}\n")
        for iz in range(NPZ):
            samp = make_sampler(zf[iz])
            for iy in range(NPY):
                for ix in range(NPX):
                    vals = []
                    for kk in range(3):                   # W[k][l]: x-offset l-1, y-offset 1-k
                        for ll in range(3):
                            xq = xf[ix] - (ll - 1) * XSIZE
                            yq = yf[iy] - (1 - kk) * YSIZE
                            vals.append(float(samp(xq, yq)))
                    if ix == 0 and iy == 0 and iz == NPZ // 2:
                        central_mid = vals[4]             # W[1][1]
                    if ix == 0 and iy == 0 and iz == NPZ - 2:
                        central_n = vals[4]
                    fp.write(f"{ix+1} {iy+1} {iz+1} " + " ".join(f"{v:.6e}" for v in vals) + "\n")
    return central_mid, central_n


if __name__ == "__main__":
    import sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    ef = f"{outdir}/ppixel2.decal.init"
    wf = f"{outdir}/wgt_pot.decal.init"
    write_efield(ef)
    cmid, cn = write_weighting(wf)
    print(f"wrote {ef}  (uniform E_z = {EZ_VCM:.1f} V/cm, thick={THICK}um, pitch={XSIZE}um, B={BFIELD})")
    print(f"wrote {wf}  (FFT Ramo weighting; central-pixel w: mid-depth={cmid:.4f}, near-collection={cn:.4f})")
    print("sanity: central-pixel weighting should rise 0 (backplane) -> ~1 (collection).")
