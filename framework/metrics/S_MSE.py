import torch


def compute_s_mse(preds):
    # preds: List: [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]

    dist = 0
    for pred_item in preds:
        assert pred_item.shape[0] > 1, "num_preds set to greater than 1"
        pred_item_ = pred_item.reshape(pred_item.shape[0], -1)  # (num_preds, d)
        dist_ = torch.pow(torch.cdist(pred_item_, pred_item_), 2)  # (num_preds, num_preds)
        dist_ = torch.sum(dist_) / (pred_item_.shape[0] * (pred_item_.shape[0] - 1) * pred_item_.shape[1])
        dist += dist_
    return dist / len(preds)

    # for b in range(preds.shape[0]):
    #     preds_item = preds[b]
    #     if preds_item.shape[0] == 1:
    #         return 0.0
    #     preds_item_ = preds_item.reshape(preds_item.shape[0], -1)
    #     dist_ = torch.pow(torch.cdist(preds_item_, preds_item_), 2)
    #     dist_ = torch.sum(dist_) / (preds_item.shape[0] * (preds_item.shape[0] - 1) * preds_item_.shape[1])
    #     dist += dist_
    # return dist / preds.shape[0]