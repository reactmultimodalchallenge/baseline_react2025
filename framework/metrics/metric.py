import torch
import numpy as np
from einops import repeat


def sample_truncnorm(shape, mean=0.0, std=0.5, low=-1.0, high=1.0):
    samples = torch.empty(shape, dtype=torch.float32)
    mask = torch.ones(shape, dtype=torch.bool)
    while mask.any():
        new = torch.randn(shape, dtype=torch.float32) * std + mean
        samples[mask] = new[mask]
        mask = (samples < low) | (samples > high)

    return samples


def baseline_random(target):
    B, L, D = target.shape
    assert D == 25

    bin_part = torch.randint(0, 2, (B, L, 15), dtype=torch.float32)
    # cont_part = torch.empty((B, L, 2)).uniform_(-1.0, 1.0)
    cont_part = sample_truncnorm((B, L, 2), mean=0.0, std=0.5, low=-1.0, high=1.0)
    cat_raw = torch.rand((B, L, 8))
    cat_part = cat_raw / cat_raw.sum(dim=-1, keepdim=True)

    return torch.cat([bin_part, cont_part, cat_part], dim=-1)


def baseline_mime(target):
    return repeat(target, 'l d -> b l d', b=10)


# def baseline_meanseq(target, pred=None):
#     # mean over time dim=1 → shape (B,1,25)
#     mean_seq = target.mean(dim=1, keepdim=True)
#     return mean_seq.repeat(1, target.size(1), 1)


# def baseline_meanfr(target, pred=None):
#     # mean over batch and time dims → shape (25,)
#     global_mean = target.mean(dim=(0,1), keepdim=True)  # → (1,1,25)
#     B, L, _ = target.shape
#     return global_mean.expand(B, L, target.size(2))


def s_mse(preds):
    # preds: (B, 10, 750, 25)
    dist = 0
    for b in range(preds.shape[0]):
        preds_item = preds[b]
        if preds_item.shape[0] == 1:
            return 0.0
        preds_item_ = preds_item.reshape(preds_item.shape[0], -1)
        dist_ = torch.pow(torch.cdist(preds_item_, preds_item_), 2)
        dist_ = torch.sum(dist_) / (preds_item.shape[0] * (preds_item.shape[0] - 1) * preds_item_.shape[1])
        dist += dist_
    return dist / preds.shape[0]


def FRVar(preds):
    if len(preds.shape) == 3:
        # preds: (10, 750, 25)
        var = torch.var(preds, dim=1)
        return torch.mean(var)
    elif len(preds.shape) == 4:
        # preds: (N, 10, 750, 25)
        var = torch.var(preds, dim=2)
        return torch.mean(var)


def FRDvs(preds):
    # preds: (N, 10, 750, 25)
    preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1)
    preds_ = preds_.transpose(0, 1)
    # preds_: (10, N, 750*25)
    dist = torch.pow(torch.cdist(preds_, preds_), 2)
    # dist: (10, N, N)
    dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1])
    return dist / preds_.shape[-1]


def compute_FRVar(pred):
    FRVar_list = []
    for k in range(pred.shape[0]):
        pred_item = pred[k]
        for i in range(0, pred_item.shape[0]):
            var = np.mean(np.var(pred_item[i].numpy().astype(np.float32), axis=0))
            FRVar_list.append(var)
    return np.mean(FRVar_list)
