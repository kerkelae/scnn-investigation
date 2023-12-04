for i in {1..7}
do
    mkdir -p intermediate/sub-0${i}
    mrconvert -fslgrad raw/sub-0${i}/dwi.bvec raw/sub-0${i}/dwi.bval raw/sub-0${i}/dwi.nii.gz intermediate/sub-0${i}/dwi.mif
    dwidenoise intermediate/sub-0${i}/dwi.mif intermediate/sub-0${i}/dwi_denoised.mif
    dwiextract -bzero intermediate/sub-0${i}/dwi.mif - | mrmath -axis 3 - mean intermediate/sub-0${i}/mean_b0.nii.gz
    mrcat -axis 3 intermediate/sub-0${i}/mean_b0.nii.gz raw/sub-0${i}/b0_negpe.nii.gz intermediate/sub-0${i}/b0s.nii.gz
    dwifslpreproc -pe_dir AP -rpe_pair -se_epi intermediate/sub-0${i}/b0s.nii.gz intermediate/sub-0${i}/dwi_denoised.mif intermediate/sub-0${i}/dwi_preprocessed.mif
    mkdir -p preprocessed/sub-0${i}
    mrconvert -export_grad_fsl preprocessed/sub-0${i}/dwi.bvec preprocessed/sub-0${i}/dwi.bval intermediate/sub-0${i}/dwi_preprocessed.mif preprocessed/sub-0${i}/dwi.nii.gz
    bet2 preprocessed/sub-0${i}/dwi.nii.gz preprocessed/sub-0${i}/brain -n -m -f 0.3
done

