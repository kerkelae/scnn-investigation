for i in {1..7}
do
    mkdir -p sub-0${i}
    dwi2response dhollander -fslgrad ../mri/preprocessed/sub-0${i}/dwi.bvec ../mri/preprocessed/sub-0${i}/dwi.bval ../mri/preprocessed/sub-0${i}/dwi.nii.gz sub-0${i}/wm_response.txt sub-0${i}/gm_response.txt sub-0${i}/csf_response.txt
    dwi2fod -mask ../mri/preprocessed/sub-0${i}/brain_mask.nii.gz msmt_csd -fslgrad ../mri/preprocessed/sub-0${i}/dwi.bvec ../mri/preprocessed/sub-0${i}/dwi.bval ../mri/preprocessed/sub-0${i}/dwi.nii.gz sub-0${i}/wm_response.txt sub-0${i}/wmfod.nii.gz sub-0${i}/gm_response.txt sub-0${i}/gm.nii.gz sub-0${i}/csf_response.txt sub-0${i}/csf.nii.gz
done

