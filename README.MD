# vit_ae_plus_plus
Codebase for the paper ViT-AE++: Improving Vision Transformer Autoencoder for Self-supervised Medical Image Representations

### Setup

Run `brats_split.py` and `egd_split.py` files in the `bootstrap` folder to create dataset splits.

Note: The `base_dir` path needs to be updated in both files. It should point to the local storage structure.

Similarly, the `BASE_PATH` in the `brats_dataset/brats.py` and `egd_dataset/egd.py` needs to be updated accordingly.

### Configurations
All the files use argparse and support command line arguments. However, configurations in the `config.ini` file can override any configuration given from the command line.
Some configurations in this file are essential such as the ones in the `K_FOLD` section. These determine the number of epochs, output_folder, logging directory, etc.


### Training the model

Download `ckp-399.pth` pretrained model weights from [here](https://filesender.switch.ch/filesender2/download.php?token=db2c8c56-11f3-4620-9767-a18f2b416c3b&files_ids=391858) and save it in the `model` folder. The model weights are necessary for using the perceptual loss.

Please use `k_fold_cross_valid_combined_brats.py` and `k_fold_cross_valid_combined_egd.py` to train models and extract ssl features for each split. 

There are also scripts to train a 3D resnet or train the model in two stages (first autoencoder and then contrastive training). However, these are experimental codes and not suitable for model training.

### Extracting features

The trained ViT model can be used as a feature extractor with an appropriate model weight and dataset split. The relevant code is present in `post_training_utils/extract_ssl_features.py` file. Please make changes in the `EXTRACT_SSL` section in the config.ini file. Also, please be mindful of the training data split
that you have chosen since the two datasets are not symmetric. Please refer to the dataset files for reference. 

### Evaluating features

The extracted SSL features are evaluated using classical models such as random forest, SVM, and logistic regression. The relevant script is present in `feature_evaluation_script/evaluation_k_fold_[Dataset].py` where 
`Dataset` can be egd or brats. Please feel free to play around and train different models for the extracted features. 
