### Setup

Run `brats_split.py` and `egd_split.py` files in the `boostrap` folder to create dataset splits.

Note: In both the files, the `base_dir` path needs to be updated. It should point to the local storage structure.

Similarly, the `BASE_PATH` in the `brats_dataset/brats.py` and `egd_dataset/egd.py` needs to be updated accordingly.

### Configurations
All the files use argparse and support command line arguments. However, configurations in the `config.ini` file can override any configuration given from the command line.
Some configurations in this file are important such as the ones in the `SETUP` section. These determine the number of epochs, output_folder, logging directory etc.


### Training the model

Download `ckp-399.pth` pretrained model weights from [here](https://filesender.switch.ch/filesender2/?s=download&token=7e4ac448-8627-434f-a3d6-9b77b4d286df) and save it in the `model` folder. The model weights are necessary for using the perceptual loss.

Please use `k_fold_cross_valid_combined_brats.py` and `k_fold_cross_valid_combined_egd.py` to train models and extract ssl features for each of the splits. 

There are also scripts to train a 3D resnet, or train the model in two stages (first autoencoder and then contrastive training). However, these are experimental code and not suitable for model training.

### Extracting features

With an appropriate model weight and dataset split, the trained ViT model can be used as a feature extractor. The relevant code is present in `post_training_utils/extract_ssl_features.py` file. Please make changes in the `EXTRACT_SSL` section in the config.ini file. Also, please be mindful of the training data split
that you have chosen since the two datasets are not symmetric. Please refer to the dataset files for reference. 