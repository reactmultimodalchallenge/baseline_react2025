import torch
from einops import rearrange


def compute_FRDvs(preds):
    NotImplemented


# def compute_FRDvs(preds):
#     # preds: List: [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ]
#     group_scores = []
#     for pred_item in preds:
#         # num_preds, L, C = pred_item.shape
#         flat = rearrange(pred_item, 'n l c -> n (l c)')
#         dist = torch.pow(torch.cdist(flat, flat), 2)  # (num_preds, num_preds)
#     preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1)
#     preds_ = preds_.transpose(0, 1)
#     # preds_: (10, N, 750*...)
#     dist = torch.pow(torch.cdist(preds_, preds_), 2)
#     # dist: (10, N, N)
#     dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1])
#     return dist / preds_.shape[-1]


# def compute_FRDvs(preds):
#     # preds: (N, 10, 750, ...)
#     preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1)
#     preds_ = preds_.transpose(0, 1)
#     # preds_: (10, N, 750*...)
#     dist = torch.pow(torch.cdist(preds_, preds_), 2)
#     # dist: (10, N, N)
#     dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1])
#     return dist / preds_.shape[-1]