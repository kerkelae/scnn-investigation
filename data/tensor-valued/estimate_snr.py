import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion


mask = binary_erosion(
    nib.load("brain_mask.nii.gz").get_fdata().astype(bool),
    iterations=3,
)
bvals = np.loadtxt("lte-pte.bval")
b0_data = nib.load("lte-pte.nii.gz").get_fdata()[mask][:, np.where(bvals == 0)[0]]
normalized_b0_data = b0_data / np.nanmean(b0_data, axis=1)[:, np.newaxis]
SNRs = 1 / np.nanstd(normalized_b0_data, axis=1)

print(
    f"SNR = {np.round(SNRs.mean(), 1).astype(int)} ± {np.round(SNRs.std(), 1).astype(int)}"
)
