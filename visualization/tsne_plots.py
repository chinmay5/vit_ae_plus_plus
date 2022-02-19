import os
import pickle

import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from ablation.train_3d_resnet import get_all_feat_und_labels
from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from fine_tune.contrastive_training import get_args_parser
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed
from read_configs import bootstrap
import torchio as tio

base_dir = '/mnt/cat/chinmay/brats_processed'
radiomics_path = os.path.join(base_dir, 'data', 'radiomics_features', 'features_flair.npy')
labels_path = os.path.join(base_dir, 'label_all.npy')

split_index_path = os.path.join(PROJECT_ROOT_DIR, 'brats', 'k_fold', 'indices_file')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_perceptual_model(model_perc, args, device):
    args.finetune = os.path.join(PROJECT_ROOT_DIR, args.common_path, args.checkpoint_perc)
    checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    state_dict = model_perc.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model_perc, checkpoint_model)

    # load pre-trained model
    msg = model_perc.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model_perc.to(device)
    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}


def extract_contrast_feat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args_parser()
    args = args.parse_args()
    args = bootstrap(args=args, key='TSNE')
    transforms = [
        tio.RandomAffine(scales=(1.15, 1.2), degrees=15),
        # tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(0.3))
    ]
    transforms = tio.Compose(transforms)

    dataset_whole = get_dataset(dataset_name=args.dataset, mode='whole', args=args, transforms=transforms,
                                use_z_score=args.use_z_score)
    features, labels = get_all_feat_und_labels(dataset_whole)
    dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=transforms,
                               use_z_score=args.use_z_score)

    train_ids = pickle.load(open(os.path.join(split_index_path, f"train_{0}"), 'rb'))
    test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{0}"), 'rb'))

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_whole,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        sampler=train_subsampler,
    )

    # Small change needed to load the contrastive model
    args.nb_classes = -1
    model_contrast = get_models(model_name='contrastive', args=args)
    # Let us now load the model weights for the two different models
    print("Loading contrastive model")
    load_contrast_model(args, model_contrast, device=device)
    model_contrast.eval()
    # Boiler-plate for storing the features
    outPRED_CON = torch.FloatTensor().to(device)
    outPRED_AUG_CON = torch.FloatTensor().to(device)
    # Same for perceptual
    outPRED_PER = torch.FloatTensor().to(device)
    outPRED_AUG_PER = torch.FloatTensor().to(device)
    for data_iter_step, (augmented, original, labels) in enumerate(data_loader_test):
        augmented = augmented.to(device, non_blocking=True)
        original = original.to(device, non_blocking=True)
        exclude = True
        for idx in range(labels.size(0)):
            label = labels[idx]
            if label == 1:
                exclude = False
            else:
                continue
            original_s = original[idx:]
            augmented_s = augmented[idx:]
            with torch.no_grad():
                orig_feat_contrast = model_contrast.forward_features(original_s)
                aug_feat_contrast = model_contrast.forward_features(augmented_s)
            # Also for the perceptual features
        # Now augment all the features
        if exclude:
            continue
        outPRED_CON = torch.cat((outPRED_CON, orig_feat_contrast), 0)
        outPRED_AUG_CON = torch.cat((outPRED_AUG_CON, aug_feat_contrast), 0)

    # Let us now save all the features that we have generated for the test set
    feature_dir = os.path.join(PROJECT_ROOT_DIR, 'visualization')
    np.save(os.path.join(feature_dir, 'cont_orig.npy'), outPRED_CON.cpu().numpy())
    np.save(os.path.join(feature_dir, 'cont_aug.npy'), outPRED_AUG_CON.cpu().numpy())
    # Now we go for the perceptual ones
    print("Starting the perceptual one")
    del model_contrast
    torch.cuda.empty_cache()
    args.nb_classes = 2
    model_perc = get_models(model_name='vit', args=args)
    print("Loading perceptual model")
    load_perceptual_model(model_perc=model_perc, args=args, device=device)
    model_perc.eval()
    for data_iter_step, (augmented, original, labels) in enumerate(data_loader_test):
        augmented = augmented.to(device, non_blocking=True)
        original = original.to(device, non_blocking=True)
        exclude = True
        for idx in range(labels.size(0)):
            label = labels[idx]
            if label == 1:
                exclude = False
            else:
                continue
            original_s = original[idx:]
            augmented_s = augmented[idx:]
            with torch.no_grad():
                # Also for the perceptual features
                orig_feat_perc = model_perc.forward_features(original_s)
                aug_feat_perc = model_perc.forward_features(augmented_s)
        # Now augment all the features
        if exclude:
            continue
        outPRED_PER = torch.cat((outPRED_PER, orig_feat_perc), 0)
        outPRED_AUG_PER = torch.cat((outPRED_AUG_PER, aug_feat_perc), 0)
    np.save(os.path.join(feature_dir, 'perc_orig.npy'), outPRED_PER.cpu().numpy())
    np.save(os.path.join(feature_dir, 'perc_aug.npy'), outPRED_AUG_PER.cpu().numpy())



def load_contrast_model(args, model_contrast, device):
    model_path = os.path.join(PROJECT_ROOT_DIR, args.common_path, args.checkpoint_contr)
    print(model_path)
    assert os.path.exists(model_path), "Please ensure a trained model alredy exists"
    checkpoint = torch.load(model_path, map_location='cpu')
    model_contrast.load_state_dict(checkpoint['model'])
    model_contrast.to(device=device)


def create_tsne_plot(plot_name):
    # labels = enc.transform(labels.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # , learning_rate='auto',
    feature_dir = os.path.join(PROJECT_ROOT_DIR, 'visualization')
    perc_aug_feat = np.load(os.path.join(feature_dir, 'perc_aug.npy'))
    perc_orig_feat = np.load(os.path.join(feature_dir, 'perc_orig.npy'))
    feat = np.concatenate((perc_orig_feat, perc_aug_feat))
    # X_embedded = TSNE(n_components=2, init='random', verbose=True).fit_transform(perc_orig_feat)
    X_embedded = TSNE(n_components=2, init='random', verbose=True, perplexity=200, n_iter=5000).fit_transform(feat)
    plt.figure(figsize=(6, 5))
    plt.figtext = plot_name
    colors = 'r'
    X_embedded_orig = X_embedded[0: X_embedded.shape[0]//2]
    X_embedded_aug = X_embedded[X_embedded.shape[0] // 2: ]
    for i in range(X_embedded_orig.shape[0]):
        plt.scatter(X_embedded_orig[i, 0], X_embedded_orig[i, 1], c=colors)
    # plt.legend()
    # plt.show()
    # Again
    # X_embedded = TSNE(n_components=2, init='random', verbose=True).fit_transform(perc_aug_feat)
    # plt.figure(figsize=(6, 5))
    # plt.figtext = plot_name
    colors = 'g'
    for i in range(X_embedded_aug.shape[0]):
        plt.scatter(X_embedded_aug[i, 0], X_embedded_aug[i, 1], c=colors)
    plt.legend()
    plt.show()

#     Now the contrastive part
    feature_dir = os.path.join(PROJECT_ROOT_DIR, 'visualization')
    contr_aug_feat = np.load(os.path.join(feature_dir, 'cont_aug.npy'))
    contr_orig_feat = np.load(os.path.join(feature_dir, 'cont_orig.npy'))
    feat = np.concatenate((contr_orig_feat, contr_aug_feat))
    X_embedded = TSNE(n_components=2, init='random', verbose=True, n_iter=5000, perplexity=200).fit_transform(feat)
    plt.figure(figsize=(6, 5))
    plt.figtext = plot_name
    X_embedded_orig = X_embedded[0: X_embedded.shape[0] // 2]
    X_embedded_aug = X_embedded[X_embedded.shape[0] // 2:]
    colors = 'r'
    for i in range(X_embedded_orig.shape[0]):
        plt.scatter(X_embedded_orig[i, 0], X_embedded_orig[i, 1], c=colors)
    # plt.legend()
    # plt.show()
    # Again
    # X_embedded = TSNE(n_components=2, init='random', verbose=True).fit_transform(contr_aug_feat)
    # plt.figure(figsize=(6, 5))
    # plt.figtext = plot_name
    colors = 'g'
    for i in range(X_embedded_aug.shape[0]):
        plt.scatter(X_embedded_aug[i, 0], X_embedded_aug[i, 1], c=colors)
    plt.legend()
    plt.show()


def get_per_feat(radiomics):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'all_comps')
    train_X_vit = np.load(os.path.join(ssl_feature_dir, 'train_ssl_features.npy'))
    train_X_combined_radiomics = np.concatenate((radiomics, train_X_vit), axis=-1)
    return train_X_combined_radiomics


def get_per_contrast_feat(radiomics):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'all_comps_contrast')
    train_X_vit = np.load(os.path.join(ssl_feature_dir, 'train_contrast_ssl_features.npy'))
    train_X_combined_radiomics = np.concatenate((radiomics, train_X_vit), axis=-1)
    return train_X_combined_radiomics


def plot_figs():
    radiomics, labels = get_radiomics_feat()
    create_tsne_plot(features=radiomics, labels=labels, plot_name='radiomics')
    # Now we go for perceptual and radiomics combined
    perc_features = get_per_feat(radiomics)
    create_tsne_plot(features=perc_features, labels=labels, plot_name='perc+radiomics')
    # Now we go for perceptual, contrastive and radiomics combined
    perc_contrast_features = get_per_contrast_feat(radiomics)
    create_tsne_plot(features=perc_contrast_features, labels=labels, plot_name='perc+contrast+radiomics')


def get_radiomics_feat():
    split_index_path = os.path.join(PROJECT_ROOT_DIR, 'brats', 'k_fold', 'indices_file')
    idx = 1
    test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{idx}"), 'rb'))




    radiomics = np.asarray(np.load(radiomics_path))
    labels = np.load(labels_path)
    # train_idx = np.load(os.path.join(base_dir, 'data', 'train_indices.npy'))
    train_idx = np.load(os.path.join(base_dir, 'data', 'train_indices.npy'))
    train_X_rad_, train_y_ = radiomics[train_idx], labels[train_idx]
    return train_X_rad_, train_y_

if __name__ == '__main__':
    # extract_contrast_feat()
    create_tsne_plot(plot_name='test')