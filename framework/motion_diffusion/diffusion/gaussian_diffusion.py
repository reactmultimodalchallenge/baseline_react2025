"""
This code started out as a PyTorch port of Ho et al's diffusion modules:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.

Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""

import enum
import math
from cv2 import CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION
import numpy as np
import torch as th
import torch.nn as nn
from einops import rearrange
from framework.motion_diffusion.diffusion.utils.util import prob_mask_like

LOSSES_TYPES = ["mse", "mse_l1", ]
MSE, MSE_L1 = LOSSES_TYPES


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif 'sqrt' in schedule_name:
        sqrt_schedulers = {
            "10sqrt1e-4": lambda t: max(0, 1 - np.power(t + 0.0001, 1 / 10)),
            "5sqrt1e-4": lambda t: max(0, 1 - np.power(t + 0.0001, 1 / 5)),
            "3sqrt1e-4": lambda t: max(0, 1 - np.power(t + 0.0001, 1 / 3)),
            "sqrt1e-4": lambda t: max(0, 1 - np.sqrt(t + 0.0001)),
            "sqrt2e-2": lambda t: max(0, 1 - np.sqrt(t + 0.02)),
            "sqrt5e-2": lambda t: max(0, 1 - np.sqrt(t + 0.05)),
            "sqrt1e-1": lambda t: max(0, 1 - np.sqrt(t + 0.1)),
            "sqrt2e-2": lambda t: max(0, 1 - np.sqrt(t + 0.2)),
        }
        assert schedule_name in sqrt_schedulers.keys(), f"Unknown sqrt scheduler {schedule_name}"
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            sqrt_schedulers[schedule_name],
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(
            min(1 - alpha_bar(t2) / max(0.00001, alpha_bar(t1)), max_beta))  # the max is to prevent singularities
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


mean_type_dict = {
    "previous_x": ModelMeanType.PREVIOUS_X,
    "start_x": ModelMeanType.START_X,
    "epsilon": ModelMeanType.EPSILON
}


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


var_type_dict = {
    "learned": ModelVarType.LEARNED,
    "fixed_small": ModelVarType.FIXED_SMALL,
    "fixed_large": ModelVarType.FIXED_LARGE,
    "learned_range": ModelVarType.LEARNED_RANGE
}


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion modules.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            noise_schedule,
            steps,
            predict='start_x',
            var_type="fixed_large",
            losses="mse",  # this can be a string or an array of strings
            losses_multipliers=1.,  # this can be a single float or an array of floats
            rescale_timesteps=False,
            noise_std=1,
            **kwargs
    ):
        assert predict in mean_type_dict.keys(), f"predict='{predict}' not supported"
        self.model_mean_type = mean_type_dict[predict]
        assert var_type in var_type_dict.keys(), f"var_type='{var_type}' not supported"
        self.model_var_type = var_type_dict[var_type]

        # support for a linear combination (losses_multipliers)) of several loses
        if isinstance(losses, str):
            losses = [losses, ]  # retro-compatibility
        if isinstance(losses_multipliers, float):
            losses_multipliers = [losses_multipliers, ]
        assert len(losses) == len(losses_multipliers)
        self.losses = losses
        self.losses_multipliers = losses_multipliers

        self.rescale_timesteps = rescale_timesteps
        self.noise_std = noise_std

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(noise_schedule, steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)  #
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # posterior_variance = beta_t * (1 - alpha_(t-1)^bar) / (1 - alpha_t^bar)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0). 
        -> using the reparametrization trick of:
            sqrt(alfa) * x_0 + sqrt(1 - alfa) * eps

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start) * self.noise_std
        assert noise.shape == x_start.shape
        weighed_x_start = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        weighed_noise = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return (weighed_x_start + weighed_noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # for sampling (inference)
        model_output = model.forward_with_cond_scale(x, self._scale_timesteps(t), model_kwargs)
        # model_output = model(x, self._scale_timesteps(t), model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            raise NotImplementedError("The ModelVarType {} is not implemented yet.".format(self.model_var_type))
            # assert model_output.shape == (B, C * 2, *x.shape[2:])
            # model_output, model_var_values = th.split(model_output, C, dim=1)
            # if self.model_var_type == ModelVarType.LEARNED:
            #     model_log_variance = model_var_values
            #     model_variance = th.exp(model_log_variance)
            # else:
            #     min_log = _extract_into_tensor(
            #         self.posterior_log_variance_clipped, t, x.shape
            #     )
            #     max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            #     # The model_var_values is [-1, 1] for [min_var, max_var].
            #     frac = (model_var_values + 1) / 2
            #     model_log_variance = frac * max_log + (1 - frac) * min_log
            #     model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (  # default
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # self.model_mean_type == ModelMeanType.START_X
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        # default the model predict START_X.
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else: # model predict gaussian noise.
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x) * self.noise_std
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        noise_to_add = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        sample = out["mean"] + noise_to_add
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "noise_to_add": noise_to_add}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            max_step=-1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param max_step: last diffusion step that wants to perform. If -1, all steps are performed
        :return: a non-differentiable batch of samples.
        """
        final = None
        for i, sample in enumerate(self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        )):
            if max_step != -1 and i == max_step:
                break
            final = sample

        return final

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            pred=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # out["mean"].shape == out["variance"].shape == out["log_variance"].shape == out["pred_xstart"].shape == (4800, 128)
        # after broadcast, out["mean"].shape == out["variance"].shape
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it (do the reverse of first term in Equation 12 in DDIM)
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])  # following Equation 12 in DDIM.

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta  # hyperparameter
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # following Equation 12 in DDIM.
        noise = th.randn_like(x) * self.noise_std
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)  # x_pred * sqrt(alpha_(t-1))
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        # out["pred_xstart"] denotes 'predicted x_0' following Equation 12 in DDIM.
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
               - out["pred_xstart"]
               ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def denoise(self, model, x_start, t, model_kwargs=None, noise=None, all_kwargs=None):
        raise NotImplementedError

    def get_gt(
            self,
            model,
            obs,
            pred
    ):
        raise NotImplementedError


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class PriorLatentDiffusion(GaussianDiffusion):
    def __init__(self, cfg, train_timesteps, inference_timesteps):
        """
        :param cfg: diffusion_prior.scheduler
        # conf.diffusion_prior.scheduler,
        # conf.diffusion_prior.scheduler.num_train_timesteps,
        # conf.diffusion_prior.scheduler.num_inference_timesteps,
        """

        super().__init__(
            noise_schedule=cfg.get("noise_schedule", "cosine"),
            steps=train_timesteps,
            predict=cfg.get("predict", "start_x"),
            var_type=cfg.get("var_type", "fixed_large"),
            rescale_timesteps=cfg.get("rescale_timesteps", False),
            noise_std=cfg.get("noise_std", 1)
        )
        self.k = cfg.num_preds

        # for DDIM sampling
        if cfg.timestep_spacing == "linspace":
            self.indices = (
                np.linspace(0, train_timesteps - 1, inference_timesteps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif cfg.timestep_spacing == "leading":
            step_ratio = train_timesteps // inference_timesteps
            self.indices = (np.arange(0, inference_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        elif cfg.timestep_spacing == "trailing":
            step_ratio = train_timesteps // inference_timesteps
            self.indices = np.round(np.arange(train_timesteps, 0, -step_ratio)).astype(np.int64)
        else:
            self.indices = list(range(self.num_timesteps))[::-1] # [999, 998, ... 0]

    def ddim_sample(
            self,
            model,
            x,
            t,
            t_prev,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # out["mean"].shape == out["variance"].shape == out["log_variance"].shape == out["pred_xstart"].shape == (4800, 128)
        # after broadcast, out["mean"].shape == out["variance"].shape
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it (do the reverse of first term in Equation 12 in DDIM)
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])  # following Equation 12 in DDIM.

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, t_prev, x.shape)

        sigma = (
                eta  # hyperparameter
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # following Equation 12 in DDIM.
        noise = th.randn_like(x) * self.noise_std
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)  # x_pred * sqrt(alpha_(t-1))
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        # out["pred_xstart"] denotes 'predicted x_0' following Equation 12 in DDIM.
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_progressive(
            self,
            matcher,
            model,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            shape=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """

        # ---------
        # copy of model_kwargs
        model_kwargs = model_kwargs.copy()

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std

        # indices = list(range(self.num_timesteps))[::-1]
        indices = self.indices

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for idx, i in enumerate(indices):
            t = th.tensor([i] * shape[0], device=device)  # timestep inverse traversal
            t_prev = th.tensor([indices[idx+1]] * shape[0], device=device) if idx < (len(indices)-1) else \
                th.tensor([0] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    t_prev,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                out = {
                    "encoded_prediction": out["pred_xstart"],
                    "sample_enc": out["sample"],
                }

                yield out
                img = out["sample_enc"]

    def denoise(self, model, x_start, t, model_kwargs=None, noise=None, all_kwargs=None):
        t = t.repeat_interleave(self.k, dim=0)  # (2, 25) -> (2, ..., 2, 25, ..., 25)
        bs = t.shape[0]
        # t.shape: (batch_size * k, )

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start) * self.noise_std  # k different noises for each corresponding x_0

        x_t = self.q_sample(x_start, t, noise=noise)  # apply perturbations from '0' to 't' to original image (x_start)
        model_output = model.forward(x_t, self._scale_timesteps(t), model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]  # here the target is set to x_0 (x_start) rather than noise (eps)

        results = {
            "encoded_prediction": model_output,  # encoded
            "encoded_target": target,  # encoded
        }
        # undo the repeat_interleave for the k predictions
        results = {k: v.view(-1, self.k, *results[k].shape[1:]) for k, v in results.items()}

        return results


class DecoderLatentDiffusion(GaussianDiffusion):
    def __init__(self, cfg, train_timesteps, inference_timesteps):
        super().__init__(
            noise_schedule=cfg.get("noise_schedule", "cosine"),
            steps=train_timesteps,
            predict=cfg.get("predict", "start_x"),
            var_type=cfg.get("var_type", "fixed_large"),
            rescale_timesteps=cfg.get("rescale_timesteps", False),
            noise_std=cfg.get("noise_std", 1)
        )
        self.k = cfg.num_preds

        # for DDIM sampling
        if cfg.timestep_spacing == "linspace":
            self.indices = (
                np.linspace(0, train_timesteps - 1, inference_timesteps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif cfg.timestep_spacing == "leading":
            step_ratio = train_timesteps // inference_timesteps
            self.indices = (np.arange(0, inference_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        elif cfg.timestep_spacing == "trailing":
            step_ratio = train_timesteps // inference_timesteps
            self.indices = np.round(np.arange(train_timesteps, 0, -step_ratio)).astype(np.int64)
        else:
            self.indices = list(range(self.num_timesteps))[::-1] # [999, 998, ... 0]

    def ddim_sample(
            self,
            model,
            x,
            t,
            t_prev,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # out["mean"].shape == out["variance"].shape == out["log_variance"].shape == out["pred_xstart"].shape == (4800, 128)
        # after broadcast, out["mean"].shape == out["variance"].shape
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it (do the reverse of first term in Equation 12 in DDIM)
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])  # following Equation 12 in DDIM.

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, t_prev, x.shape)

        sigma = (
                eta  # hyperparameter
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # following Equation 12 in DDIM.
        noise = th.randn_like(x) * self.noise_std
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)  # x_pred * sqrt(alpha_(t-1))
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        # out["pred_xstart"] denotes 'predicted x_0' following Equation 12 in DDIM.
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_progressive(
            self,
            matcher,
            model,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            shape=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        self.shape = shape
        # copy of model_kwargs
        model_kwargs = model_kwargs.copy()

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(size=shape, device=device) * self.noise_std

        # indices = list(range(self.num_timesteps))[::-1]
        indices = self.indices

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for idx, i in enumerate(indices):
            t = th.tensor([i] * shape[0], device=device)  # timestep inverse traversal
            t_prev = th.tensor([indices[idx+1]] * shape[0], device=device) if idx < (len(indices)-1) else \
                th.tensor([0] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t, # t
                    t_prev, # t-1
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                out = {
                    "decoded_prediction": out["pred_xstart"],
                    "sample_enc": out["sample"],
                }

                yield out
                img = out["sample_enc"]

    def denoise(self, model, x_start, t, model_kwargs=None, noise=None, all_kwargs=None):
        # t = t.repeat_interleave(self.k, dim=0)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start) * self.noise_std  # k different noises for each corresponding x_0

        x_t = self.q_sample(x_start, t, noise=noise)  # apply perturbations from '0' to 't' to original image (x_start)
        model_output = model(x_t, self._scale_timesteps(t), model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]  # here the target is set to x_0 (x_start) or noise (epsilon)

        results = {
            "prediction_emotion": model_output,
            "target_emotion": target,
        }
        return results
