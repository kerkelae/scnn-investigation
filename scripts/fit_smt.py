from dipy.core.gradients import gradient_table
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
from dmipy.distributions.distribute_models import BundleModel
from dmipy.signal_models import cylinder_models, gaussian_models
import numpy as np
import torch


# Acquisition protocol

bvals = (
    np.round(np.loadtxt("../data/mri/preprocessed/sub-01/dwi.bval") / 1e3, decimals=1)
    * 1e3
)
bvecs = np.loadtxt("../data/mri/preprocessed/sub-01/dwi.bvec").T

idx = bvals > 0
bvals = bvals[idx]
bvecs = bvecs[idx]

bvecs = np.concatenate((np.zeros((1, 3)), bvecs), axis=0)
bvals = np.concatenate((np.zeros(1), bvals))

gtab = gradient_table(bvals, bvecs, small_delta=60e-3, big_delta=3)
scheme = gtab_dipy2dmipy(gtab)


# Test dataset

signals = torch.load("../test_signals.pt")
targets = torch.load("../test_targets.pt")
data = signals.numpy()
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)


# Model definition

stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()
bundle = BundleModel([stick, zeppelin])
bundle.set_tortuous_parameter(
    "G2Zeppelin_1_lambda_perp", "C1Stick_1_lambda_par", "partial_volume_0"
)
bundle.set_equal_parameter("G2Zeppelin_1_lambda_par", "C1Stick_1_lambda_par")
mcdmi_mod = modeling_framework.MultiCompartmentSphericalMeanModel(models=[bundle])


# Model fit

_ = mcdmi_mod.fit(scheme, data[0])  # compilation
mcdmi_fit = mcdmi_mod.fit(scheme, data)
mcmdi_csd_mod = modeling_framework.MultiCompartmentSphericalHarmonicsModel(
    models=[bundle]
)
for name, value in mcdmi_fit.fitted_parameters.items():
    mcmdi_csd_mod.set_fixed_parameter(name, value)
mcmdi_csd_fit = mcmdi_csd_mod.fit(scheme, data)
preds = np.concatenate(
    (
        mcmdi_csd_fit.fod_sh(),
        (mcdmi_fit.fitted_parameters["BundleModel_1_G2Zeppelin_1_lambda_par"] * 1e9)[
            :, np.newaxis
        ],
        mcdmi_fit.fitted_parameters["BundleModel_1_partial_volume_0"][:, np.newaxis],
    ),
    axis=1,
)

torch.save(torch.Tensor(preds), "../smt_preds.pt")


# Rotation dataset

signals = torch.load("../rotation_signals.pt").reshape(-1, 120)
data = signals.numpy()
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

mcdmi_fit = mcdmi_mod.fit(scheme, data)
mcmdi_csd_fit = mcmdi_csd_mod.fit(scheme, data)
preds = np.concatenate(
    (
        mcmdi_csd_fit.fod_sh(),
        (mcdmi_fit.fitted_parameters["BundleModel_1_G2Zeppelin_1_lambda_par"] * 1e9)[
            :, np.newaxis
        ],
        mcdmi_fit.fitted_parameters["BundleModel_1_partial_volume_0"][:, np.newaxis],
    ),
    axis=1,
)

torch.save(torch.Tensor(preds), "../rotation_smt_preds.pt")
