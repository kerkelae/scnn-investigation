from typing import Sequence

import healpy as hp
import numpy as np
import scipy.special
import torch


def spherical_harmonic(
    l: int, m: int, thetas: Sequence[float], phis: Sequence[float]
) -> torch.Tensor:
    """Evaluate a real and symmetric spherical harmonic basis function.

    Args:
        l: Degree of the spherical harmonic.
        m: Order of the spherical harmonic.
        thetas: Polar angles in radians.
        phis: Azimuthal angles in radians.

    Returns:
        The evaluated spherical harmonic.
    """
    if l % 2 == 1:
        return torch.zeros(len(thetas))
    elif m < 0:
        return torch.Tensor(
            np.sqrt(2) * scipy.special.sph_harm(-m, l, phis, thetas).imag
        )
    elif m == 0:
        return torch.Tensor(scipy.special.sph_harm(m, l, phis, thetas).real)
    else:
        return torch.Tensor(
            np.sqrt(2) * scipy.special.sph_harm(m, l, phis, thetas).real
        )


l_max = 16  # maximum SH degree
n_coeffs = int(0.5 * (l_max + 1) * (l_max + 2))  # maximum number of SH coefficients

ls = torch.zeros(n_coeffs, dtype=int)
l0s = torch.zeros(n_coeffs, dtype=int)
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        ls[int(0.5 * l * (l + 1) + m)] = l  # SH degree
        l0s[int(0.5 * l * (l + 1) + m)] = int(
            0.5 * l * (l + 1)
        )  # index of the SH for which l = l and m = 0

n_sides = 2**4  # corresponds to 3072 vertices
vertices = torch.vstack(
    [torch.tensor(i) for i in hp.pix2vec(n_sides, np.arange(12 * n_sides**2))]
).T
thetas = torch.arccos(vertices[:, 2])
phis = (torch.arctan2(vertices[:, 1], vertices[:, 0]) + 2 * np.pi) % (2 * np.pi)
isft = torch.zeros((len(vertices), n_coeffs))  # inverse spherical Fourier transform
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        isft[:, int(0.5 * l * (l + 1) + m)] = spherical_harmonic(l, m, thetas, phis)
sft = torch.linalg.pinv(isft.T @ isft) @ isft.T  # spherical Fourier transform
