import torch


def compute_FRVar(preds):
    # preds: [Tensor(num_preds, l, 25), Tensor(num_preds, l, 25), ...]
    variance = []
    for pred in preds:
        variance.append(torch.var(pred, dim=1))

    variance = torch.stack(variance, dim=0)
    return torch.mean(variance)


# def compute_FRVar(preds):
#     if len(preds.shape) == 3:
#         # preds: (10, 750, ...)
#         var = torch.var(preds, dim=1)
#         return torch.mean(var)
#     elif len(preds.shape) == 4:
#         # preds: (N, 10, 750, ...)
#         var = torch.var(preds, dim=2)
#         return torch.mean(var)
