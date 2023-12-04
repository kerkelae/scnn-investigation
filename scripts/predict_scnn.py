import numpy as np
import torch

from scnn.models import SCNNModel
from scnn.sh import n_coeffs, spherical_harmonic

if torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    raise Exception("CUDA not available")

BATCH_SIZE = int(5e2)


# Acquisition protocol

bvals = torch.round(
    torch.tensor(np.loadtxt("../data/mri/preprocessed/sub-01/dwi.bval") / 1e3),
    decimals=1,
).float()
bvecs = torch.tensor(np.loadtxt("../data/mri/preprocessed/sub-01/dwi.bvec").T).float()

idx = bvals > 0
bvals = bvals[idx]
bvecs = bvecs[idx]

bs = torch.unique(bvals)
n_shells = len(bs)
shell_indices = [torch.where(bvals == b)[0] for b in bs]

bvecs_isft_per_shell = []
bvecs_sft_per_shell = []
for idx in shell_indices:
    shell_bvecs = bvecs[idx]
    thetas = torch.arccos(shell_bvecs[:, 2])
    phis = (torch.arctan2(shell_bvecs[:, 1], shell_bvecs[:, 0]) + 2 * np.pi) % (
        2 * np.pi
    )
    bvecs_isft = torch.zeros(len(shell_bvecs), 45)
    for l in range(0, 8 + 1, 2):
        for m in range(-l, l + 1):
            bvecs_isft[:, int(0.5 * l * (l + 1) + m)] = spherical_harmonic(
                l, m, thetas, phis
            )
    bvecs_isft_per_shell.append(bvecs_isft)
    bvecs_sft = torch.zeros((45, len(shell_bvecs)), dtype=float)
    bvecs_sft[0:45] = (
        torch.linalg.pinv(bvecs_isft[:, 0:45].T @ bvecs_isft[:, 0:45])
        @ bvecs_isft[:, 0:45].T
    )
    bvecs_sft_per_shell.append(bvecs_sft.float())


# Load test data and targets

signals = torch.load("../test_signals.pt")
signals_sh = torch.zeros(len(signals), n_shells, 45)
for i, idx in enumerate(shell_indices):
    signals_sh[:, i] = (bvecs_sft_per_shell[i] @ signals[:, idx].unsqueeze(-1)).squeeze(
        -1
    )
targets = torch.load("../test_targets.pt")


# Load trained models and predict

scnn_model = SCNNModel(2, 2).to(device)
scnn_model.load_state_dict(torch.load("../scnn_weights.pt"))
scnn_model.eval()

scnn_model_rot = SCNNModel(2, 2).to(device)
scnn_model_rot.load_state_dict(torch.load("../scnn_weights_rot.pt"))
scnn_model_rot.eval()

scnn_preds = torch.zeros(targets.size())
scnn_preds_rot = torch.zeros(targets.size())
with torch.no_grad():
    for i in range(0, len(signals), BATCH_SIZE):
        print(f"{int(100 * i / len(signals))}%", end="\r")
        idx = torch.arange(i, i + BATCH_SIZE)
        scnn_preds[idx] = scnn_model(
            torch.nn.functional.pad(signals_sh[idx], (0, n_coeffs - 45)).to(device)
        ).cpu()
        scnn_preds_rot[idx] = scnn_model_rot(
            torch.nn.functional.pad(signals_sh[idx], (0, n_coeffs - 45)).to(device)
        ).cpu()
print("100%")

torch.save(scnn_preds, "../scnn_preds.pt")
torch.save(scnn_preds_rot, "../scnn_preds_rot.pt")


# Rotation dataset

signals = torch.load("../rotation_signals.pt").reshape(-1, 120)
signals_sh = torch.zeros(len(signals), n_shells, 45)
for i, idx in enumerate(shell_indices):
    signals_sh[:, i] = (bvecs_sft_per_shell[i] @ signals[:, idx].unsqueeze(-1)).squeeze(
        -1
    )

scnn_preds = torch.zeros(len(signals), 47)
scnn_preds_rot = torch.zeros(len(signals), 47)
with torch.no_grad():
    for i in range(0, len(signals), BATCH_SIZE):
        print(f"{int(100 * i / len(signals))}%", end="\r")
        idx = torch.arange(i, i + BATCH_SIZE)
        scnn_preds[idx] = scnn_model(
            torch.nn.functional.pad(signals_sh[idx], (0, n_coeffs - 45)).to(device)
        ).cpu()
        scnn_preds_rot[idx] = scnn_model_rot(
            torch.nn.functional.pad(signals_sh[idx], (0, n_coeffs - 45)).to(device)
        ).cpu()
print("100%")

torch.save(scnn_preds, "../rotation_scnn_preds.pt")
torch.save(scnn_preds_rot, "../rotation_scnn_preds_rot.pt")
