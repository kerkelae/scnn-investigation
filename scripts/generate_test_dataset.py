import numpy as np
import torch

from scnn.sh import spherical_harmonic
from scnn.sim import compartment_model_simulation


if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
else:
    raise Exception("CUDA not available")


N_TEST = int(1e6)
BATCH_SIZE = int(1e3)
SNR = 50
SEED = 123


# Define the simulated acquisition protocol

bvals = torch.round(
    torch.tensor(np.loadtxt("../data/mri/preprocessed/sub-01/dwi.bval") / 1e3),
    decimals=1,
).float()
bvecs = torch.tensor(np.loadtxt("../data/mri/preprocessed/sub-01/dwi.bvec").T).float()

idx = bvals > 0
bvals = bvals[idx]
bvecs = bvecs[idx]

bs = torch.unique(bvals)
shell_indices = [torch.where(bvals == b)[0] for b in bs]

bvecs_isft_per_shell = []
for idx in shell_indices:
    shell_bvecs = bvecs[idx]
    thetas = torch.arccos(shell_bvecs[:, 2])
    phis = (torch.arctan2(shell_bvecs[:, 1], shell_bvecs[:, 0]) + 2 * torch.pi) % (
        2 * torch.pi
    )
    bvecs_isft = torch.zeros(len(shell_bvecs), 45)
    for l in range(0, 8 + 1, 2):
        for m in range(-l, l + 1):
            bvecs_isft[:, int(0.5 * l * (l + 1) + m)] = spherical_harmonic(
                l, m, thetas, phis
            )
    bvecs_isft_per_shell.append(bvecs_isft)


# Generate targets by random sampling

torch.random.manual_seed(SEED)
ds = torch.rand(N_TEST) * 3
fs = torch.rand(N_TEST)

odfs_sh = torch.load("../data/odfs/test_odfs_sh.pt").float()
Rs = torch.tensor(
    np.concatenate(
        (
            np.eye(45)[np.newaxis],
            np.load("../Rs.npy").reshape(-1, 45, 45),
        ),
        axis=0,
    )
).float()
np.random.seed(SEED)
odfs_sh = (
    Rs[np.random.choice(len(Rs), N_TEST)]
    @ odfs_sh[np.random.choice(len(odfs_sh), N_TEST)].unsqueeze(-1)
).squeeze(-1)

targets = torch.hstack((odfs_sh, ds.unsqueeze(1), fs.unsqueeze(1)))


# Generate signals

signals = torch.zeros(N_TEST, 120)

for i in range(0, N_TEST, BATCH_SIZE):
    print(f"{int(i / N_TEST * 100)}%", end="\r")

    idx = torch.arange(i, i + BATCH_SIZE)

    batch_ads = torch.vstack((ds[idx], ds[idx])).T
    batch_rds = torch.vstack(
        (
            torch.zeros(len(idx)),
            (1 - fs[idx]) * ds[idx],
        )
    ).T
    batch_fs = torch.vstack((fs[idx], 1 - fs[idx])).T
    batch_odfs_sh = odfs_sh[idx]

    for j, b in enumerate(bs):
        shell_signals = compartment_model_simulation(
            b.item(),
            bvecs_isft_per_shell[j],
            batch_ads,
            batch_rds,
            batch_fs,
            batch_odfs_sh,
        ).cpu()
        shell_signals = torch.abs(
            shell_signals
            + torch.normal(
                mean=torch.zeros(shell_signals.size()),
                std=torch.ones(shell_signals.size()) / SNR,
            )
            + 1j
            * torch.normal(
                mean=torch.zeros(shell_signals.size()),
                std=torch.ones(shell_signals.size()) / SNR,
            )
        )
        signals[
            idx.unsqueeze(1),
            shell_indices[j],
        ] = shell_signals.squeeze(-1).float()
print("100%")

torch.save(signals, "../test_signals.pt")
torch.save(targets, "../test_targets.pt")
