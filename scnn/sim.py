import math

from dipy.core.gradients import gradient_table
import numpy as np
import torch

from .sh import l0s, ls, sft, vertices


rf_btens = torch.tensor(
    gradient_table(np.ones(len(vertices)), vertices, btens="LTE").btens
).float()
sim_sft = sft[0:45]
sim_l0s = l0s[0:45]
sim_ls = ls[0:45]


def compartment_model_simulation(
    bval: float,
    bvecs_isft: torch.Tensor,
    ads: torch.Tensor,
    rds: torch.Tensor,
    fs: torch.Tensor,
    odfs: torch.Tensor,
) -> torch.Tensor:
    """Simulate single-shell diffusion MR measurements.

    Args:
        bval: Simulated b-value.
        bvecs_isft: Inverse spherical Fourier transform matrix for the simulated
            gradient directions.
        ads: Axial diffusivities of the simulated compartments. The shape must
            be (number of simulations, number of compartments).
        rds: Radial diffusivities of the simulated compartments. The shape must
            be (number of simulations, number of compartments).
        fs: Signal fractions of the simulated compartments. The shape must be
            (number of simulations, number of compartments).
        odfs: Simulated orientation distribution functions. The shape must be
            (number of simulations, number of SH coefficients).

    Returns:
        Simulated signals.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_simulations = ads.shape[0]
    n_compartments = ads.shape[1]
    Ds = torch.zeros(n_simulations, n_compartments, 3, 3).to(device)
    Ds[:, :, 2, 2] = ads
    Ds[:, :, 1, 1] = rds
    Ds[:, :, 0, 0] = rds
    response = torch.sum(
        fs.to(device).unsqueeze(2)
        * torch.exp(
            -torch.sum(
                bval * rf_btens.unsqueeze(0).unsqueeze(1).to(device) * Ds.unsqueeze(2),
                dim=(3, 4),
            )
        ),
        dim=1,
    )
    response_sh = (sim_sft.to(device) @ response.unsqueeze(-1)).squeeze(-1)
    convolution_sh = (
        torch.sqrt(4 * math.pi / (2 * sim_ls.to(device) + 1))
        * odfs.to(device)
        * response_sh[:, sim_l0s]
    )
    simulated_signals = bvecs_isft.to(device) @ convolution_sh.unsqueeze(-1)
    return simulated_signals
