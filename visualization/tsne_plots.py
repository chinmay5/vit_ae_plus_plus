import os
import pickle

import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from fine_tune.contrastive_training import get_args_parser
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed
from read_configs import bootstrap

base_dir = '/mnt/cat/chinmay/brats_processed'
radiomics_path = os.path.join(base_dir, 'data', 'radiomics_features', 'features_flair.npy')
labels_path = os.path.join(base_dir, 'label_all.npy')


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
    dataset_train = get_dataset(dataset_name=args.dataset, mode='train', args=args, transforms=None,
                                use_z_score=args.use_z_score)
    dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=None,
                               use_z_score=args.use_z_score)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model_perc = get_models(model_name='vit', args=args)
    # Small change needed to load the contrastive model
    args.nb_classes = -1
    model_contrast = get_models(model_name='contrastive', args=args)
    # Let us now load the model weights for the two different models
    print("Loading contrastive model")
    load_contrast_model(args, model_contrast)
    print("Loading perceptual model")
    load_perceptual_model(model_perc=model_perc, args=args, device=device)

    model_contrast.eval()
    model_perc.eval()
    # Boiler-plate for storing the features
    outPRED_CON = torch.FloatTensor().to(device)
    outPRED_AUG_CON = torch.FloatTensor().to(device)
    # Same for perceptual
    outPRED_PER = torch.FloatTensor().to(device)
    outPRED_AUG_PER = torch.FloatTensor().to(device)
    for data_iter_step, (augmented, original, _) in enumerate(data_loader_test):

        augmented = augmented.to(device, non_blocking=True)
        original = original.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            orig_feat_contrast = model_contrast.forward_features(original)
            aug_feat_contrast = model_contrast.forward_features(augmented)
            # Also for the perceptual features
            orig_feat_perc = model_perc.forward_features(original)
            aug_feat_perc = model_perc.forward_features(augmented)
        # Now augment all the features
        outPRED_CON = torch.cat((outPRED_CON, orig_feat_contrast), 0)
        outPRED_AUG_CON = torch.cat((outPRED_AUG_CON, aug_feat_contrast), 0)
        outPRED_PER = torch.cat((outPRED_PER, orig_feat_perc), 0)
        outPRED_AUG_PER = torch.cat((outPRED_AUG_PER, aug_feat_perc), 0)

    # Let us now save all the features that we have generated for the test set
    feature_dir = os.path.join(PROJECT_ROOT_DIR, 'visualization')
    np.save(os.path.join(feature_dir, 'cont_orig.npy'), outPRED_CON.cpu().numpy())
    np.save(os.path.join(feature_dir, 'cont_aug.npy'), outPRED_AUG_CON.cpu().numpy())
    np.save(os.path.join(feature_dir, 'perc_orig.npy'), outPRED_PER.cpu().numpy())
    np.save(os.path.join(feature_dir, 'perc_aug.npy'), outPRED_AUG_PER.cpu().numpy())



def load_contrast_model(args, model_contrast):
    model_path = os.path.join(PROJECT_ROOT_DIR, args.common_path, args.checkpoint_contr)
    assert os.path.exists(model_path), "Please ensure a trained model alredy exists"
    checkpoint = torch.load(model_path, map_location='cpu')
    model_contrast.load_state_dict(checkpoint['model'])


def create_tsne_plot(features, labels, plot_name):
    # labels = enc.transform(labels.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # , learning_rate='auto',
    X_embedded = TSNE(n_components=2, init='random', verbose=True).fit_transform(features)
    target_ids = range(2)  # Since binary classification
    target_names = ['low', 'high']
    plt.figure(figsize=(6, 5))
    plt.figtext = plot_name
    colors = 'r', 'g'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], c=c, label=label)
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
    extract_contrast_feat()