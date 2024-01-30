import healpy as hp
import nibabel as nib
import numpy as np

from scnn.sh import spherical_harmonic


BATCH_SIZE = int(1e3)

n_sides = 2**5  # use high sampling density
vertices = np.vstack(hp.pix2vec(n_sides, np.arange(12 * n_sides**2))).T
thetas = np.arccos(vertices[:, 2])
phis = (np.arctan2(vertices[:, 1], vertices[:, 0]) + 2 * np.pi) % (2 * np.pi)
isft = np.zeros((len(vertices), 45))
for l in range(0, 8 + 1, 2):
    for m in range(-l, l + 1):
        isft[:, int(0.5 * l * (l + 1) + m)] = spherical_harmonic(l, m, thetas, phis)

for i in range(1, 6):
    mask = (
        nib.load(f"../mri/preprocessed/sub-0{i}/brain_mask.nii.gz")
        .get_fdata()
        .astype(bool)
    )
    odfs_sh = nib.load(f"sub-0{i}/wmfod.nii.gz").get_fdata()[mask]
    odfs_sh = (odfs_sh / odfs_sh[:, 0:1]) / (
        4 * np.pi * spherical_harmonic(0, 0, [0], [0]).item()
    )  # normalize

    for j in range(0, len(odfs_sh), BATCH_SIZE):
        idx = np.arange(j, j + BATCH_SIZE)
        if idx.max() >= len(odfs_sh):
            idx = np.arange(j, len(odfs_sh))

        batch_odfs_sh = odfs_sh[idx]
        batch_odfs = (isft @ batch_odfs_sh[..., np.newaxis])[..., 0]
        idx = ~(np.sum(np.isnan(batch_odfs), axis=1).astype(bool)) & (
            np.min(batch_odfs, axis=1) > -1e-2
        )  # exclude ODFs with NaNs or significantly negative values
        batch_odfs_sh = batch_odfs_sh[idx]
        if i == 1 and j == 0:
            filtered_odfs_sh = batch_odfs_sh
        else:
            filtered_odfs_sh = np.vstack((filtered_odfs_sh, batch_odfs_sh))

print(len(filtered_odfs_sh))
np.savetxt("train_odfs_sh.txt", filtered_odfs_sh)

mask = (
    nib.load(f"../data/preprocessed/sub-06/brain_mask.nii.gz").get_fdata().astype(bool)
)
odfs_sh = nib.load(f"sub-06/wmfod.nii.gz").get_fdata()[mask]
odfs_sh = odfs_sh / odfs_sh[:, 0][:, np.newaxis] / np.sqrt(4 * np.pi)
for j in range(0, len(odfs_sh), BATCH_SIZE):
    idx = np.arange(j, j + BATCH_SIZE)
    if idx.max() >= len(odfs_sh):
        idx = np.arange(j, len(odfs_sh))
    batch_odfs_sh = odfs_sh[idx]
    batch_odfs = (isft @ batch_odfs_sh[..., np.newaxis])[..., 0]
    idx = ~(np.sum(np.isnan(batch_odfs), axis=1).astype(bool)) & (
        np.min(batch_odfs, axis=1) > -1e-2
    )
    batch_odfs_sh = batch_odfs_sh[idx]
    if j == 0:
        filtered_odfs_sh = batch_odfs_sh
    else:
        filtered_odfs_sh = np.vstack((filtered_odfs_sh, batch_odfs_sh))
print(len(filtered_odfs_sh))
np.savetxt("val_odfs_sh.txt", filtered_odfs_sh)

mask = (
    nib.load(f"../data/preprocessed/sub-07/brain_mask.nii.gz").get_fdata().astype(bool)
)
odfs_sh = nib.load(f"sub-07/wmfod.nii.gz").get_fdata()[mask]
odfs_sh = odfs_sh / odfs_sh[:, 0][:, np.newaxis] / np.sqrt(4 * np.pi)
for j in range(0, len(odfs_sh), BATCH_SIZE):
    idx = np.arange(j, j + BATCH_SIZE)
    if idx.max() >= len(odfs_sh):
        idx = np.arange(j, len(odfs_sh))
    batch_odfs_sh = odfs_sh[idx]
    batch_odfs = (isft @ batch_odfs_sh[..., np.newaxis])[..., 0]
    idx = ~(np.sum(np.isnan(batch_odfs), axis=1).astype(bool)) & (
        np.min(batch_odfs, axis=1) > -1e-2
    )
    batch_odfs_sh = batch_odfs_sh[idx]
    if j == 0:
        filtered_odfs_sh = batch_odfs_sh
    else:
        filtered_odfs_sh = np.vstack((filtered_odfs_sh, batch_odfs_sh))
print(len(filtered_odfs_sh))
np.savetxt("test_odfs_sh.txt", filtered_odfs_sh)
