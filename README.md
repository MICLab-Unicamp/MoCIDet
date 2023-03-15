# MoCIDet

This repository has the code to assess T1-weighted neuroimages quality. The assessment verifies the presence of motion-artifacts on these images. The result is a probability of artifact presence on T1w volume analyzed. 

## Requirements
Having a GPU is not necessary, but will speed up training time.
Prediction time is around 14 sec 

### Software Requirements
To run this code, you need to have the following libraries installed:

python3
tensorflow >= 2.0
matplotlib
nibabel
pydicom
numpy
scikit-image
h5py 

The complete software list is reported in requirements.txt

## Usage

**python3 MoCIDet.py -data_path path/to/folder/containing/volumes -data_type 'image_type' -save_file 'name_to_save_predictions'**   


    arguments: 

      data_path is the path to the directory where the T1w acquisitions are stored

      data_type is the acquisition type:

                  nifti

                  dicom - volumetric dicom file (one file)

                  multi-dicom - volumetric multiple dicom files (path to the directory root)

                  dicom-2D - multiple 2D dicom files (path to the directory root)

      save_file is the txt file to save the motion-presence probabilities


    To display the partial results of the predictions use -display

    To save a sample of a slice presenting the minimum and the maximum probability use -save_slice


## Examples:

[1] 

*python3.7 MoCIDet.py -data_path test_anon/multi-dicom/NORM210 -data_type 'multi-dicom' -save_file 'test_multi-dicom' -display -save_slice*

test_anon/multi-dicom/NORM210 0.24444444444444444

Time:  14.42564868927002

[['test_anon/multi-dicom/NORM210' '0.24444444444444444']]

[2]

*python3.7 MoCIDet.py -data_path test_anon/multi-dicom/ -data_type 'multi-dicom' -save_file 'test_multi-dicom2' -display -save_slice*

test_anon/multi-dicom/NORM210 0.24444444444444444

Time:  14.927184820175171

[['test_anon/multi-dicom/NORM210' '0.24444444444444444']]


[3]

*python3.7 MoCIDet.py -data_path test_anon/multi-dicom/NORM210 -data_type 'dicom-2D' -save_file 'test_dicom_2D' -display -save_slice*

test_anon/multi-dicom/NORM210 0.

Time:  6.983335971832275

[['test_anon/multi-dicom/NORM210' '0.16666666666666666']]

[4]

*python3.7 MoCIDet.py -data_path test_anon/multi-dicom/ -data_type 'dicom-2D' -save_file 'test_dicom_2D2' -display -save_slice*

test_anon/multi-dicom/NORM210 0.16666666666666666

Time:  68.88042497634888

[['test_anon/multi-dicom/NORM210' '0.16666666666666666']]


[5]

*python3.7 MoCIDet.py -data_path test_anon/dicom/ -data_type 'dicom' -save_file 'test_dicom' -display -save_slice*

test_anon/dicom/133 0.044444444444444446

Time:  11.524296522140503

test_anon/dicom/3300 0.9777777777777777

Time:  12.145656824111938

[['test_anon/dicom/133' '0.044444444444444446']

['test_anon/dicom/3300' '0.9777777777777777']]

[6]

The same dicom files above were converted to nifti.

*python3.7 MoCIDet.py -data_path test_anon/nifti_from_dicom/ -data_type 'nifti' -save_file 'test_nifti_from_dicom' -display -save_slice*

test_anon/nifti_from_dicom/133.nii.gz 0.05925925925925926

Time:  14.275437831878662

test_anon/nifti_from_dicom/3300.nii 0.9555555555555556

Time:  14.370188474655151

test_anon/nifti_from_dicom/3300.nii.gz 0.9555555555555556

Time:  15.57526183128357

[['test_anon/nifti_from_dicom/133.nii.gz' '0.05925925925925926']

['test_anon/nifti_from_dicom/3300.nii' '0.9555555555555556']

['test_anon/nifti_from_dicom/3300.nii.gz' '0.9555555555555556']]


[7]

*python3.7 MoCIDet.py -data_path test_anon/nifti/ -data_type 'nifti' -save_file 'test_nifti' -display -save_slice*

test_anon/nifti/ABIDE_50002_MRI_MP-RAGE_br_raw_20120830172854796_S164623_I328631.nii 1.0

Time:  14.558613538742065

test_anon/nifti/ABIDE_50003_MRI_MP-RAGE_br_raw_20120830155445855_S164416_I328410.nii 1.0

Time:  15.803523540496826

test_anon/nifti/sub-10159_T1w.nii.gz 1.0

Time:  14.653321027755737

test_anon/nifti/sub-10206_T1w.nii.gz 0.05185185185185185

Time:  14.778350114822388

[['test_anon/nifti/ABIDE_50002_MRI_MP-RAGE_br_raw_20120830172854796_S164623_I328631.nii' '1.0']

['test_anon/nifti/ABIDE_50003_MRI_MP-RAGE_br_raw_20120830155445855_S164416_I328410.nii' '1.0']

['test_anon/nifti/sub-10159_T1w.nii.gz' '1.0']

['test_anon/nifti/sub-10206_T1w.nii.gz' '0.05185185185185185']]
