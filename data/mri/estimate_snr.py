import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion


for i in range(7):
    mask = binary_erosion(
        nib.load(f"preprocessed/sub-0{i + 1}/brain_mask.nii.gz")
        .get_fdata()
        .astype(bool),
        iterations=3,
    )
    bvals = np.loadtxt(f"preprocessed/sub-0{i + 1}/dwi.bval")
    b0_data = nib.load(f"preprocessed/sub-0{i + 1}/dwi.nii.gz").get_fdata()[mask][
        :, np.where(bvals == 0)[0]
    ]
    normalized_b0_data = b0_data / np.nanmean(b0_data, axis=1)[:, np.newaxis]
    if i == 0:
        concatenated_b0_data = normalized_b0_data
    else:
        concatenated_b0_data = np.concatenate(
            (concatenated_b0_data, normalized_b0_data), axis=0
        )

SNRs = 1 / np.nanstd(concatenated_b0_data, axis=1)

print(
    f"SNR = {np.round(SNRs.mean(), 1).astype(int)} ± {np.round(SNRs.std(), 1).astype(int)}"
)
