"""The most probable and the mean energy loss of a charged particle in a thin silicon layer.

    most_probable_loss_kev(beta_gamma, thickness_um)    keV, the peak of the loss distribution
    mean_loss_kev(beta_gamma, thickness_um)             keV, the textbook mean at minimum ionisation

Both are vectorised over beta_gamma (a number or an array; beta gamma = p / (m c)) and return a
float for a number, an array of the same shape for an array. Notebooks 09 and 10 use them as rough
guides for the charge of one crossing (divide by 3.68 eV per electron-hole pair).

THE MOST PROBABLE LOSS. Review of Particle Physics, "Passage of particles through matter",
eq. 34.12, for a layer of thickness x (cm) and density rho (g/cm3):

    Delta_p = xi * [ ln(2 m c^2 beta^2 gamma^2 / I) + ln(xi / I) + j - beta^2 - delta(beta gamma) ]
    xi      = (K / 2) * (Z / A) * rho * x / beta^2                                  (MeV)

with, for silicon,
    K    = 0.307075 MeV cm2/mol      K_RPP
    Z/A  = 14 / 28.0855              Z_A
    rho  = 2.33 g/cm3                RHO_SI
    I    = 173 eV                    I_SI (in MeV)
    m c^2 = 0.51099895 MeV           M_E (the electron mass)
    j    = 0.200                     J_LVB

and delta the density effect in Sternheimer's parametrisation for silicon, with
X = log10(beta gamma):

    delta = 2 ln(10) X - C + a (X1 - X)^k     for X0 <= X < X1
    delta = 2 ln(10) X - C                    for X >= X1
    delta = delta0 * 10^(2 (X - X0))          for X < X0

    C = 4.4355 (C_D), X0 = 0.2015 (X0_D), X1 = 2.8716 (X1_D), a = 0.14921 (A_D),
    k = 3.2546 (K_D), delta0 = 0.14 (D0_D).

Above X1 (beta gamma 744) the most probable loss no longer rises: the density effect cancels the
growth of the logarithms. For 100 um of silicon it is 24.1 keV at beta gamma 3.5 and 26.1 keV for
a fast particle.

THE MEAN LOSS, as notebooks 09 and 10 define it: the textbook minimum ionisation in silicon,
DEDX_MIN = 1.66 MeV cm2/g, times RHO_SI and the thickness: 38.7 keV in 100 um, the same for every
beta gamma (the argument only sets the shape of the result). It is a rough guide: in a thin
layer energetic delta rays carry part of the loss out of the layer, so the mean deposit is lower.
"""
import numpy as np

DEDX_MIN = 1.66      # MeV cm2/g, minimum ionisation in silicon (textbook value)
RHO_SI = 2.33        # g/cm3
# RPP eq. 34.12: MeV cm2/mol, -, MeV, MeV, -
K_RPP, Z_A, I_SI, M_E, J_LVB = 0.307075, 14 / 28.0855, 173e-6, 0.51099895, 0.200
# the density effect for silicon (Sternheimer): C, X0, X1, a, k, delta0
C_D, X0_D, X1_D, A_D, K_D, D0_D = 4.4355, 0.2015, 2.8716, 0.14921, 3.2546, 0.14


def _out(a):
    # a float for a number, the array otherwise
    return float(a) if np.ndim(a) == 0 else a


def density_effect(beta_gamma):
    """delta(beta gamma) for silicon, the three branches of the parametrisation above."""
    bg = np.asarray(beta_gamma, dtype=np.float64)
    x = np.log10(bg)
    above_x0 = 2 * np.log(10) * x - C_D + A_D * np.clip(X1_D - x, 0.0, None) ** K_D   # the last term is 0 from X1 on
    below_x0 = D0_D * 10 ** (2 * (x - X0_D))
    return _out(np.where(x >= X0_D, above_x0, below_x0))


def most_probable_loss_kev(beta_gamma, thickness_um):
    """The most probable energy loss [keV] in thickness_um of silicon, RPP eq. 34.12 (module docstring)."""
    bg = np.asarray(beta_gamma, dtype=np.float64)
    b2 = bg ** 2 / (1 + bg ** 2)
    dlt = np.asarray(density_effect(bg))
    xi = K_RPP / 2 * Z_A * RHO_SI * thickness_um * 1e-4 / b2                                  # MeV
    return _out(1e3 * xi * (np.log(2 * M_E * bg ** 2 / I_SI) + np.log(xi / I_SI) + J_LVB - b2 - dlt))


def mean_loss_kev(beta_gamma, thickness_um):
    """The mean energy loss [keV] in thickness_um of silicon as the notebooks take it: DEDX_MIN x RHO_SI x
    thickness, the same for every beta_gamma (module docstring)."""
    mean = DEDX_MIN * RHO_SI * (thickness_um * 1e-4) * 1e3
    return _out(np.full(np.shape(beta_gamma), mean))
