## Anatomically-Constrained Data Augmentation

### Description
The script generates a massive dataset of artificially generated segmentations (using rejection sampling) that are
classified as anatomically correct or incorrect. These numerous artificially generated and anatomically correct
segmentations can then be used in post-processing to replace predicted segmentations that are anatomically incorrect,
therefore guaranteeing the anatomical validity of the final results.

### How to run
If you've generated either the ACDC or CAMUS dataset and have trained an autoencoder system on it, you can perform
anatomically-constrained data augmentation by sampling the latent space learned by the autoencoder.

A few examples of basic commands to explore the CLI and run anatomically-constrained data augmentation are given below:
```bash
# list generic data augmentation options and
# datasets on which you can perform data augmentation
python anatomically_constrained_da.py -h

# list data augmentation options available on a specific dataset (e.g. CAMUS)
python anatomically_constrained_da.py \
  <PRETRAINED_AE> camus -h

# perform data augmentation on a specific dataset (e.g CAMUS)
python anatomically_constrained_da.py \
  <PRETRAINED_AE> camus --dataset_path <DATASET_PATH> --batch_size <BATCH_SIZE>
```
