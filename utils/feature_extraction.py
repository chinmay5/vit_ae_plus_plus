import os

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def generate_features(data_loader, model, device, ssl_feature_dir, feature_file_name='features.npy', label_file_name='gt_labels.npy', log_writer=None):
    """
    We save labels and predictions together. Saving labels just makes our life easier in terms of having pickles stored at the same location
    :param data_loader: train/test
    :param model: vit model
    :param device: cuda/cpu
    :param ssl_feature_dir: location for storing features
    :param feature_file_name: npy file for features default: features.npy
    :param label_file_name: npy file for labels default: gt_labels.npy
    :param log_writer: SummaryWriter default:None
    :return: None
    """
    # switch to evaluation mode
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    for batch in tqdm(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.forward_features(images)
        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target), 0)
    if feature_file_name is not None:
        print("Saving features!!!")
        np.save(os.path.join(ssl_feature_dir, feature_file_name), outPRED.cpu().numpy())
    if label_file_name is not None:
        print("Saving labels!!!")
        np.save(os.path.join(ssl_feature_dir, label_file_name), outGT.cpu().numpy())
    if log_writer is not None:
        metadata = [x.item() for x in outGT]
        log_writer.add_embedding(outPRED, metadata=metadata, tag='ssl_embedding')