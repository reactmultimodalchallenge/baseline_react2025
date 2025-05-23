from __future__ import print_function

import torch


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, k=1, reduction="min", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(axis=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, k=1, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds==k, features]
    # target has shape of [batch_size, num_preds==k, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


class MSELoss_AE:
    def __init__(self, w_mse=1, w_kld=1, w_coeff=1, **kwargs):
        self.w_mse = w_mse
        self.w_kld = w_kld
        self.w_coeff = w_coeff

    def __call__(self, prediction, target, coefficients_3dmm, target_coefficients, mu, logvar):
        batch_size = prediction.shape[0]

        prediction = prediction.reshape(prediction.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        coefficients_3dmm = coefficients_3dmm.reshape(coefficients_3dmm.shape[0], -1)
        target_coefficients = target_coefficients.reshape(target_coefficients.shape[0], -1)

        MSE = ((prediction - target) ** 2).mean()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        COEFF = ((coefficients_3dmm - target_coefficients) ** 2).mean()

        loss_r = self.w_mse * MSE + self.w_kld * KLD + self.w_coeff * COEFF
        return {"loss": loss_r, "mse": MSE, "kld": KLD, "coeff": COEFF}
