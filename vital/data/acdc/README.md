# ACDC project

### Description

This folder contains the necessary file to use the ACDC dataset: [ACDC](www.creatis.insa-lyon
.fr/Challenge/acdc/) 2017 dataset (Automatic Cardiac Delineation Challenge).

# Scripts

## ACDC Dataset Generator

### Description

The script can be used to generate the semantic segmentation dataset into the hdf5 format, from the nifti MRIs and
groundtruths.

### How to use `dataset_generator.py`

The raw data must be downloaded and organized in the following format

```yaml
/:  # root

    - training :
        - patient001:
            - Info.cfg
            - patient002_frame{ED}.nii.gz
            - patient002_frame{ED}_gt.nii.gz
            - patient002_frame{ES}.nii.gz
            - patient002_frame{ES}_gt.nii.gz
        ...
        - patient100: ...

    - testing:  # patient data
        - patient101: ...
        ...
        - patient150: ...
```

```bash
python dataset_generator.py --path='path/of/the/raw_MRI_nifti_data' --name='name/of/the/output.hdf5'
```
### Parameters explanation

* `--path`, the parent directory of all the pathologies.
* `--name`, name of the generated hdf5 file.
* `--data_augmentation`, `-d`, add data augmentation to the dataset (rotation -60 to 60).
* `--registering`, `-r`, apply registering (registering and rotation). Only works when groundtruths are provided.


Once you've finished generating the dataset, it can be used through the
[ACDC `VisionDataset` interface](dataset.py) to train and evaluate models. The data inside in the HDF5 dataset is
structured according to the following format:
```yaml
/:  # root of the file
    - prior: Average shape over all training patients
    - train:
        - patient{XXXX}:
          - ED:
              - img # MRI image (N, 256, 256, 1)
              - gt # segmentation (N x 256 x 256 x 4)
          - ES:
              - img # MRI image
              - gt # segmentation
        ....
        - patient{YYYY}: ...
    - val:
        - patient{XXXX}:
        ....
        - patient{YYYY}: ...
    - test:
        - patient{XXXX}:
        ....
        - patient{YYYY}: ...

```
