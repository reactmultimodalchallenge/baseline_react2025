"""
Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import os
from einops import rearrange
from omegaconf import DictConfig
from hydra.utils import instantiate
from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser, \
    lengths_to_mask
from framework.motion_diffusion.diffusion.gaussian_diffusion import DecoderLatentDiffusion
from framework.motion_diffusion.diffusion.resample import UniformSampler
from framework.motion_diffusion.diffusion.rnn import LatentEmbedder
from framework.utils.util import from_pretrained_checkpoint, save_checkpoint


class BaseLatentModel(nn.Module):
    def __init__(self, cfg, emb_preprocessing=False, freeze_encoder=True, **kwargs):
        super(BaseLatentModel, self).__init__()
        self.emb_preprocessing = emb_preprocessing
        self.freeze_encoder = freeze_encoder
        def_dtype = torch.get_default_dtype()

        self.audio_encoder = instantiate(cfg.audio_encoder)
        if cfg.latent_embedder is not None:
            self.latent_embedder = instantiate(cfg.latent_embedder)
            model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.latent_embedder.checkpoint_path)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.latent_embedder.load_state_dict(state_dict)
            print(f"Successfully loaded latent embedder from {model_path}")
        else:
            self.latent_embedder = LatentEmbedder()

        if self.freeze_encoder:  # freeze modules
            for para in self.latent_embedder.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype)
        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")

    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class DecoderLatentMatcher(BaseLatentModel):
    def __init__(self,
                 conf: DictConfig = None,
                 module_dict_cfg: DictConfig = None,
                 stage: str = 'fit',
                 task: str = 'online',
                 **kwargs):
        cfg = conf.args
        super(DecoderLatentMatcher, self).__init__(
            module_dict_cfg,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
            **kwargs,
        )

        self.stage = stage
        self.task = task
        self.token_len = cfg.token_len
        self.window_size = cfg.get("window_size", 30)
        self.s_ratio = cfg.get("s_ratio", 2)
        self.emotion_dim = cfg.get("nfeats", 25)
        self.encode_emotion = cfg.get("encode_emotion", False)
        self.encode_3dmm = cfg.get("encode_3dmm", False)

        self.init_params = {
            "task": task,
            "window_size": self.window_size,
            "encode_emotion": self.encode_emotion,
            "encode_3dmm": self.encode_3dmm,
            "ablation_skip_connection": cfg.get("ablation_skip_connection", True),
            "nfeats": cfg.get("nfeats", 25),
            "latent_dim": cfg.get("latent_dim", 512),
            "ff_size": cfg.get("ff_size", 1024),
            "num_layers": cfg.get("num_layers", 6),
            "num_heads": cfg.get("num_heads", 4),
            "dropout": cfg.get("dropout", 0.1),
            "normalize_before": cfg.get("normalize_before", False),
            "activation": cfg.get("activation", "gelu"),
            "flip_sin_to_cos": cfg.get("flip_sin_to_cos", True),
            "return_intermediate_dec": cfg.get("return_intermediate_dec", False),
            "position_embedding": cfg.get("position_embedding", "learned"),
            "arch": cfg.get("arch", "trans_enc"),
            "freq_shift": cfg.get("freq_shift", 0),
            "time_encoded_dim": cfg.get("time_encoded_dim", 64),
            "s_audio_dim": cfg.get("s_audio_dim", 768),
            "s_audio_scale": cfg.get("s_audio_scale", cfg.get("latent_dim", 512) ** -0.5),
            "s_emotion_dim": cfg.get("s_emotion_dim", 25),
            "s_3dmm_dim": cfg.get("s_3dmm_dim", 58),
            "concat": cfg.get("concat", "concat_first"),
            "condition_concat": cfg.get("condition_concat", "token_concat"),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "s_audio_enc_drop_prob": cfg.get("s_audio_enc_drop_prob", 0.2),
            "s_latent_embed_drop_prob": cfg.get("s_latent_embed_drop_prob", 0.2),
            "s_3dmm_enc_drop_prob": cfg.get("s_3dmm_enc_drop_prob", 0.2),
            "s_emotion_enc_drop_prob": cfg.get("s_emotion_enc_drop_prob", 1.0),
            "past_l_emotion_drop_prob": cfg.get("past_l_emotion_drop_prob", 1.0),
        }
        self.use_past_frames = cfg.get("use_past_frames", False)

        self.model = TransformerDenoiser(**self.init_params)

        self.decoder_diffusion = DecoderLatentDiffusion(
            conf.scheduler,
            conf.scheduler.num_train_timesteps,
            conf.scheduler.num_inference_timesteps,
        )
        self.schedule_sampler = UniformSampler(self.decoder_diffusion)
        self.num_preds = conf.scheduler.num_preds

    def _forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
    ):
        with torch.no_grad():
            s_audio_encodings = self.audio_encoder._encode(speaker_audio_input)
            s_audio_encodings = s_audio_encodings.repeat_interleave(self.num_preds, dim=0)

          # freeze latent RNN_VAE embedder to extract speaker latent embedding
            s_latent_embed = self.latent_embedder.encode(speaker_emotion_input).unsqueeze(1)
            s_latent_embed = s_latent_embed.repeat_interleave(self.num_preds, dim=0)
            # shape: (batch_size * num_preds, 1, ...)

            # s_3dmm_encodings = self.latent_3dmm_embedder.get_encodings(speaker_3dmm_input)
            s_3dmm_encodings = speaker_3dmm_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            s_emotion_encodings = speaker_emotion_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            past_listener_emotion = past_listener_emotion.repeat_interleave(
                self.num_preds, dim=0) if past_listener_emotion is not None else None
            # shape: (bs * num_preds, l_w, ...)

            motion_length = motion_length.repeat_interleave(
                self.num_preds, dim=0) if motion_length is not None else None

            model_kwargs = {
                "speaker_audio_encodings": s_audio_encodings,
                "speaker_latent_embed": s_latent_embed,
                "speaker_3dmm_encodings": s_3dmm_encodings,
                "speaker_emotion_encodings": s_emotion_encodings,
                "past_listener_emotion": past_listener_emotion,
                "motion_length": motion_length,
            }

        if self.stage == "test":
            bs, l, _ = s_audio_encodings.shape  # bz * num_preds
            with torch.no_grad():
                output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                    matcher=self,
                    model=self.model,
                    model_kwargs=model_kwargs,
                    shape=(bs, self.window_size if self.task == "online" else l, self.emotion_dim),
                )][-1]  # get last output

            output_listener_emotion = output["sample_enc"]  # (bz * num_preds, l_w, d=25)
            output_listener_emotion = rearrange(output_listener_emotion,
                                                "(b n) w d -> b n w d", n=self.num_preds)
            output_whole = {"prediction_emotion": output_listener_emotion}

        else:
            listener_emotion_input = listener_emotion_input.repeat_interleave(self.num_preds, dim=0)
            x_start_selected = listener_emotion_input  # (bs * num_preds, l_w, ...)

            t, _ = self.schedule_sampler.sample(x_start_selected.shape[0], x_start_selected.device)
            timesteps = t.long()

            output_whole = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                          model_kwargs=model_kwargs)
            if motion_length is not None:  # offline task zero masking
                device = x_start_selected.get_device()
                output_mask = lengths_to_mask(motion_length, device=device, max_len=x_start_selected.shape[1])
                # print(f'output_whole["prediction_emotion"] shape: {output_whole["prediction_emotion"].shape}')
                output_whole["prediction_emotion"] = (output_whole["prediction_emotion"]
                                                      * output_mask.float().unsqueeze(-1))

            output_whole = {k: v.view(-1, self.num_preds, *output_whole[k].shape[1:]) for k, v in output_whole.items()}
        return output_whole

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class LatentMatcher(nn.Module):
    def __init__(self,
                 task: str = "online",
                 stage: str = "fit",
                 device: str = None,
                 diffusion_decoder: DictConfig = None,
                 latent_embedder: DictConfig = None,
                 audio_encoder: DictConfig = None,
                 resumed_training: bool = False,
                 **kwargs):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task = task
        self.stage = stage
        self.kwargs = kwargs

        module_dict_cfg = DictConfig(
            {"latent_embedder": latent_embedder,
             "audio_encoder": audio_encoder,}
        )

        self.diffusion_decoder_cfg = diffusion_decoder
        self.diffusion_decoder = DecoderLatentMatcher(self.diffusion_decoder_cfg,
                                                      task=task,
                                                      stage=stage,
                                                      module_dict_cfg=module_dict_cfg,
                                                      **kwargs)
        load_ckpt = False
        want_last = False
        want_best = False

        if resumed_training:
            load_ckpt = True
            want_last = True
        if stage == "test":
            load_ckpt = True
            want_best = True

        if load_ckpt:
            ckpt_path = self.get_ckpt_path(
                self.diffusion_decoder.model,
                runid="resume_runid",
                epoch=None,
                best=want_best,
                last=want_last,
            )
            from_pretrained_checkpoint(str(ckpt_path), self.diffusion_decoder.model, device)

    def forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
    ):

        outputs = self.diffusion_decoder.forward(
            speaker_audio_input=speaker_audio_input,
            speaker_emotion_input=speaker_emotion_input,
            speaker_3dmm_input=speaker_3dmm_input,
            listener_emotion_input=listener_emotion_input,
            past_listener_emotion=past_listener_emotion,
            motion_length=motion_length,
        )
        # outputs['prediction_emotion']: (bz, num_preds, s_w, emotion_dim)
        return outputs

    def get_ckpt_path(self, model, runid="current_runid", epoch=None, best=False, last=False):
        ckpt_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = Path(self.kwargs.get(runid))
        ckpt_dir = str(ckpt_dir / run_id / model.get_model_name())
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = None
        if epoch is not None:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
        if best:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
        if last:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        assert ckpt_path is not None, "No checkpoint path is provided."
        return ckpt_path

    def save_ckpt(self, optimizer, epoch=None, best=False, last=False, best_loss=float("inf")):
        model = self.diffusion_decoder.model
        ckpt_path = self.get_ckpt_path(model, epoch=epoch, best=best, last=last)
        save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, best_loss=best_loss)