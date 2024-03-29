### SOURCE: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py


import torch
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3, Inception_V3_Weights
from tqdm import tqdm

import numpy as np

def inception_score(imgs_loader, device, eps=1E-16):
    # load inception v3 model
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    model.eval()
    # enumerate splits of images/predictions
    scores = list()
    for batch in tqdm(imgs_loader):
        batch = batch.to(device)
        # predict p(y|x)
        p_yx = F.softmax(model(batch), dim=-1).data.cpu().numpy()
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)

    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std