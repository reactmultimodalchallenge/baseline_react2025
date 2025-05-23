from typing import Dict, List
import math
import torch
import torch.nn as nn
from torch import Tensor
# from framework.motion_diffusion.diffusion.utils.util import tgt_biased_mask, memory_biased_mask
from framework.motion_diffusion.diffusion.operator.embeddings import (TimestepEmbedding, Timesteps)
from framework.motion_diffusion.diffusion.operator.position_encoding import build_position_encoding
from framework.motion_diffusion.diffusion.operator.cross_attention import (SkipTransformerEncoder,
                                                                           TransformerDecoder,
                                                                           TransformerDecoderLayer,
                                                                           TransformerEncoderLayer)


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def timestep_embedding(timesteps, dim, max_period=10000, dtype=torch.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].type(dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TransformerDenoiser(nn.Module):
    def __init__(self,
                 task: str = 'offline',
                 window_size: int = 30,
                 encode_emotion: bool = False,
                 encode_3dmm: bool = False,
                 ablation_skip_connection: bool = True,
                 nfeats: int = 25,
                 latent_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_dec",
                 freq_shift: int = 0,
                 time_encoded_dim: int = 64,
                 s_audio_dim: int = 78,
                 s_audio_scale: float = 1.0,
                 s_emotion_dim: int = 25,
                 s_embed_dim: int = 512,
                 s_3dmm_dim: int = 58,
                 concat: str = "concat_first",
                 condition_concat: str = "token_concat",
                 guidance_scale: float = 7.5,
                 s_audio_enc_drop_prob: float = 0.2,  # speaker_audio_encodings
                 s_latent_embed_drop_prob: float = 0.2,  # speaker_latent_embed
                 s_3dmm_enc_drop_prob: float = 0.2,  # speaker_3dmm_encodings
                 s_emotion_enc_drop_prob: float = 1.0,  # speaker_emotion_encodings
                 past_l_emotion_drop_prob: float = 1.0,  # past_listener_emotion
                 **kwargs) -> None:
        super().__init__()

        self.task = task
        self.window_size = window_size
        self.encode_emotion = encode_emotion
        self.encode_3dmm = encode_3dmm
        self.s_audio_scale = s_audio_scale
        self.latent_dim = latent_dim
        self.ablation_skip_connection = ablation_skip_connection
        self.arch = arch
        self.concat = concat
        self.condition_concat = condition_concat
        # for classifier-free guidance
        self.guidance_scale = guidance_scale
        self.s_audio_enc_drop_prob = s_audio_enc_drop_prob
        self.s_latent_embed_drop_prob = s_latent_embed_drop_prob
        self.s_3dmm_enc_drop_prob = s_3dmm_enc_drop_prob
        self.s_emotion_enc_drop_prob = s_emotion_enc_drop_prob
        self.past_l_emotion_drop_prob = past_l_emotion_drop_prob

        # project between emotion output feat and emotion latent embedding
        self.to_emotion_embed = nn.Linear(nfeats, self.latent_dim) if nfeats != self.latent_dim else nn.Identity()
        self.to_emotion_feat = nn.Linear(self.latent_dim, nfeats) if self.latent_dim != nfeats else nn.Identity()

        # project time to latent_dim
        self.time_proj = Timesteps(time_encoded_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_encoded_dim, self.latent_dim)

        self.speaker_latent_proj = nn.Sequential(nn.ReLU(), nn.Linear(s_embed_dim, self.latent_dim)) \
            if s_embed_dim != self.latent_dim else nn.Identity()

        self.speaker_audio_proj = nn.Linear(s_audio_dim, self.latent_dim) \
            if s_audio_dim != self.latent_dim else nn.Identity()

        if self.encode_3dmm:  # assume dimension of encoded 3dmm equals latent_dim
            self.speaker_3dmm_proj = nn.Identity()
        else:
            assert s_3dmm_dim != self.latent_dim, "wrong dimension of raw 3dmm features."
            self.speaker_3dmm_proj = nn.Linear(s_3dmm_dim, self.latent_dim)

        if self.encode_emotion:  # assume dimension of encoded emotion equals latent_dim
            self.emotion_proj = nn.Identity()
        else:
            assert s_emotion_dim != self.latent_dim, "wrong dimension of raw emotion features."
            self.emotion_proj = nn.Linear(s_emotion_dim, self.latent_dim)

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        # we concat conditions (including: speaker 3dmm, speaker audio, speaker emotion encodings) along last dimension.
        self.condition_proj = nn.Linear(self.latent_dim * 3, self.latent_dim) \
            if self.condition_concat == 'feat_concat' else nn.Identity()

        # define our transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        if self.arch == "trans_enc":  # Transformer Encoder
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)

        elif self.arch == "trans_dec":  # Transformer Decoder
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")

    def mask_cond(self, feature, mode='test', drop_prob=0.0):  # train or test
        bs, _, _ = feature.shape

        # classifier-free guidance
        if mode == 'test':  # inference
            uncond_feat, con_feat = feature.chunk(2)
            # con_feat = con_feat
            uncond_feat = torch.zeros_like(uncond_feat)
            feature = torch.cat((uncond_feat, con_feat), dim=0)

        else:  # train or val mode
            if drop_prob > 0.0:
                mask = torch.bernoulli(
                    torch.ones(bs, device=feature.device) *
                    drop_prob).view(
                    bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
                feature = feature * (1.0 - mask)

        return feature

    def get_model_kwargs(
            self,
            bs,
            mode,
            sample,
            model_kwargs,
    ):

        speaker_audio_encodings = model_kwargs.get('speaker_audio_encodings')
        if speaker_audio_encodings is None or self.s_audio_enc_drop_prob >= 1.0:
            speaker_audio_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_audio_encodings = self.speaker_audio_proj(speaker_audio_encodings)
            speaker_audio_encodings = self.mask_cond(speaker_audio_encodings, mode, self.s_audio_enc_drop_prob)
            speaker_audio_encodings = self.s_audio_scale * speaker_audio_encodings
        speaker_audio_encodings = speaker_audio_encodings.permute(1, 0, 2).contiguous()

        speaker_3dmm_encodings = model_kwargs.get("speaker_3dmm_encodings")
        if speaker_3dmm_encodings is None or self.s_3dmm_enc_drop_prob >= 1.0:
            speaker_3dmm_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_3dmm_encodings = self.speaker_3dmm_proj(speaker_3dmm_encodings)
            speaker_3dmm_encodings = self.mask_cond(speaker_3dmm_encodings, mode, self.s_3dmm_enc_drop_prob)
        speaker_3dmm_encodings = speaker_3dmm_encodings.permute(1, 0, 2).contiguous()

        speaker_emotion_encodings = model_kwargs.get("speaker_emotion_encodings")
        if speaker_emotion_encodings is None or self.s_emotion_enc_drop_prob >= 1.0:
            speaker_emotion_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_emotion_encodings = self.emotion_proj(speaker_emotion_encodings)
            speaker_emotion_encodings = self.mask_cond(speaker_emotion_encodings, mode, self.s_emotion_enc_drop_prob)
        speaker_emotion_encodings = speaker_emotion_encodings.permute(1, 0, 2).contiguous()

        speaker_latent_embed = model_kwargs.get('speaker_latent_embed')
        if speaker_latent_embed is None or self.s_latent_embed_drop_prob >= 1.0:
            speaker_latent_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_latent_embed = self.speaker_latent_proj(speaker_latent_embed)
            speaker_latent_embed = self.mask_cond(speaker_latent_embed, mode, self.s_latent_embed_drop_prob)
        speaker_latent_embed = speaker_latent_embed.permute(1, 0, 2).contiguous()

        past_listener_emotion = model_kwargs.get('past_listener_emotion')
        if past_listener_emotion is None or self.past_l_emotion_drop_prob >= 1.0:
            past_listener_emotion = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            past_listener_emotion = self.emotion_proj(past_listener_emotion)
            past_listener_emotion = self.mask_cond(past_listener_emotion, mode, self.past_l_emotion_drop_prob)
        past_listener_emotion = past_listener_emotion.permute(1, 0, 2).contiguous()

        return (speaker_audio_encodings,
                speaker_latent_embed,
                speaker_3dmm_encodings,
                speaker_emotion_encodings,
                past_listener_emotion)

    def _forward(
            self,
            sample,
            time_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_emotion,
            motion_length=None,  # Tensor: (bz, )
    ):

        # print("speaker_audio_encodings", speaker_audio_encodings.shape)
        # print("speaker_latent_embed", speaker_latent_embed.shape)
        # print("speaker_3dmm_encodings", speaker_3dmm_encodings.shape)
        # print("speaker_emotion_encodings", speaker_emotion_encodings.shape)
        # print("past_listener_emotion", past_listener_emotion.shape) \
        #     if past_listener_emotion is not None else print("past_listener_emotion None")

        # map to latent dim
        sample = self.to_emotion_embed(sample)
        device = time_embed.device

        emb_latent_list = [
            time_embed,
            speaker_audio_encodings,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            speaker_latent_embed,  # TODO motion_length applied rigorously
            past_listener_emotion,
        ]
        # [N', bs, d]

        emb_latent = torch.cat(emb_latent_list, dim=0)

        tgt_key_padding_mask = None
        memory_key_padding_mask = None

        if self.task == "offline":
            assert motion_length is not None, "motion_length should not be None when task is offline"
            mask = lengths_to_mask(motion_length,  # (bs,)
                                   device=device,
                                   max_len=speaker_audio_encodings.shape[0])
            # print(f"mask shape {mask.shape}")

            mask_u = torch.zeros(size=(time_embed.shape[1],
                                       time_embed.shape[0])).bool().to(device)
            # print(f"mask_u shape {mask_u.shape}")

            tgt_key_padding_mask = ~mask
            memory_key_padding_mask = torch.zeros(size=(time_embed.shape[1], 0),
                                                  dtype=torch.bool).to(device)
            # (bs, clip_len)
            for emb in emb_latent_list:
                if emb.shape[0] == mask_u.shape[-1]:
                    memory_key_padding_mask = torch.cat((memory_key_padding_mask, mask_u), dim=-1)
                elif emb.shape[0] == mask.shape[-1]:
                    memory_key_padding_mask = torch.cat((memory_key_padding_mask, ~mask), dim=-1)

        else:  # online task
            if motion_length is not None:  # during test stage
                mask_l = lengths_to_mask(motion_length,
                                         device=device,
                                         max_len=self.window_size)
                mask_s = torch.cat(
                    [torch.zeros_like(mask_l).bool().to(device),
                     ~mask_l], dim=-1)  # (bz, 2 * window_size)
                mask_u = torch.zeros(size=(time_embed.shape[1],
                                           time_embed.shape[0])).bool().to(device)

                tgt_key_padding_mask = ~mask_l
                memory_key_padding_mask = torch.zeros(size=(time_embed.shape[1], 0),
                                                      dtype=torch.bool).to(device)
                for emb in emb_latent_list:
                    if emb.shape[0] == mask_u.shape[-1]:
                        memory_key_padding_mask = torch.cat((memory_key_padding_mask, mask_u), dim=-1)
                    elif emb.shape[0] == mask_s.shape[-1]:
                        memory_key_padding_mask = torch.cat((memory_key_padding_mask, mask_s), dim=-1)
                    # else:
                    #     # memory_key_padding_mask = torch.cat((memory_key_padding_mask, mask_l), dim=-1)
                    #     raise ValueError(f"emb.shape[0] != mask_u.shape[-1] or emb.shape[0] != mask_s.shape[-1]")
                memory_key_padding_mask = None if memory_key_padding_mask.shape[-1] == 0 else memory_key_padding_mask

        if self.arch == "trans_dec":
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask).squeeze(0)
        else:
            raise NotImplementedError(f"{self.arch} is not supported.")

        sample = self.to_emotion_feat(sample)
        sample = sample.permute(1, 0, 2)

        return sample

    def forward_with_cond_scale(
            self,
            sample,  # noised x_t
            timesteps,
            model_kwargs,

    ):
        sample = torch.cat([sample] * 2, dim=0)
        bs, _, _ = sample.shape
        sample = sample.permute(1, 0, 2).contiguous()

        timesteps = torch.cat([timesteps] * 2, dim=0)
        # with embedding permutation: [batch_size, l, encoded_dim] => [l, batch_size, encoded_dim]
        time_emb = self.time_proj(timesteps)  # time_embedding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_embed = self.time_embedding(time_emb).unsqueeze(0)

        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()
        for k, v in model_kwargs.items():
            if model_kwargs[k] is None:
                continue
            model_kwargs[k] = torch.cat(
                (torch.zeros_like(model_kwargs[k], dtype=model_kwargs[k].dtype), model_kwargs[k]),
                dim=0)

        (speaker_audio_encodings,
         speaker_latent_embed,
         speaker_3dmm_encodings,
         speaker_emotion_encodings,
         past_listener_emotion) = (
            self.get_model_kwargs(
                bs,
                'test',
                sample,
                model_kwargs,
            )
        )

        # motion_length = torch.cat([motion_length] * 2, dim=0) if motion_length is not None else None
        prediction = self._forward(
            sample,
            time_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_emotion,
            model_kwargs.get('motion_length', None),
        )

        pred_uncond, pred_cond = prediction.chunk(2)
        if self.guidance_scale == 1.0:  # TODO to be updated
            return pred_cond

        # classifier-free guidance
        prediction = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        return prediction

    def forward(
            self,
            sample,  # noised x_t
            timesteps,
            model_kwargs,
            **kwargs,
    ):
        bs, _, _ = sample.shape
        sample = sample.permute(1, 0, 2).contiguous()

        # with embedding permutation: [batch_size, l, encoded_dim] => [l, batch_size, encoded_dim]
        time_emb = self.time_proj(timesteps)  # time_embedding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_embed = self.time_embedding(time_emb).unsqueeze(0)

        (speaker_audio_encodings,
         speaker_latent_embed,
         speaker_3dmm_encodings,
         speaker_emotion_encodings,
         past_listener_emotion) = (
            self.get_model_kwargs(
                bs,
                'train',
                sample,
                model_kwargs,
            )
        )

        output = self._forward(
            sample,
            time_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_emotion,
            model_kwargs.get('motion_length', None),
        )

        return output

    def get_model_name(self):
        return self.__class__.__name__

# listener_latent_embed = model_kwargs.get('listener_latent_embed')
# if listener_latent_embed is None or self.l_latent_embed_drop_prob >= 1.0:
#     listener_latent_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
# else:
#     # [1, bs, encoded_dim] => [1, bs, latent_dim]
#     listener_latent_embed = self.listener_latent_proj(listener_latent_embed)
#     listener_latent_embed = self.mask_cond(listener_latent_embed, mode, self.l_latent_embed_drop_prob)
# listener_latent_embed = listener_latent_embed.permute(1, 0, 2).contiguous()

# listener_personal_embed = model_kwargs.get('listener_personal_embed')
# if listener_personal_embed is None or self.l_personal_embed_drop_prob >= 1.0:
#     listener_personal_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
# else:
#     listener_personal_embed = self.listener_personal_proj(listener_personal_embed)
#     listener_personal_embed = self.mask_cond(listener_personal_embed, mode, self.l_personal_embed_drop_prob)
# listener_personal_embed = listener_personal_embed.permute(1, 0, 2).contiguous()

#     elif self.condition_concat == 'feat_concat':
#     assert speaker_emotion_encodings.shape[0] == speaker_3dmm_encodings.shape[0], \
#         "we need to use temporal emotion encodings or raw AU features."
#
#     emb_latent = torch.cat((
#         speaker_audio_encodings,
#         speaker_3dmm_encodings,
#         speaker_emotion_encodings,
#     ), dim=-1)  # [seq_len, bs, latent_dim * 3]
#
#     emb_latent = self.condition_proj(emb_latent)  # (seq_len, bs, latent_dim)
#     # append time embedding
#     emb_latent = torch.cat((time_embed, emb_latent), dim=0)
#     embed_seq_len = emb_latent.shape[0]
#
#     if self.arch == "trans_enc":
#         if self.concat == "concat_first":
#             xseq = torch.cat((emb_latent, sample), dim=0)
#             xseq = self.query_pos(xseq)
#             tokens = self.encoder(xseq)
#             sample = tokens[embed_seq_len:]
#         elif self.concat == "concat_last":
#             xseq = torch.cat((sample, emb_latent), dim=0)
#             xseq = self.query_pos(xseq)
#             tokens = self.encoder(xseq)
#             sample = tokens[:embed_seq_len]
#         else:
#             raise NotImplementedError(f"{self.concat} is not supported.")
#
#     elif self.arch == "trans_dec":
#         sample = self.query_pos(sample)
#         emb_latent = self.mem_pos(emb_latent)
#         sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)
#
#     else:
#         raise NotImplementedError(f"{self.arch} is not supported.")
#
# elif self.condition_concat == 'cascade':
# sample = self.query_pos(sample)
# memory = torch.cat((
#     time_embed,
#     speaker_3dmm_encodings,
#     speaker_latent_embed,
# ), dim=0)
#
# sample = self.transformer_fusion_ns2sv(tgt=sample, memory=memory).squeeze(0)
# memory = torch.cat((time_embed, speaker_audio_encodings), dim=0)
# sample = self.transformer_fusion_ns2sa(tgt=sample, memory=memory).squeeze(0)
# sample = self.transformer_fusion_final(tgt=sample, memory=sample).squeeze(0)
#
# # map back to original dim
# sample = self.to_emotion_feat(sample)
# sample = sample.permute(1, 0, 2)

# if self.condition_concat == 'cascade':
#     num_layer_ns2sa = 3  # noise sample <---interact---> speaker audio
#     num_layer_ns2sv = 3  # noise sample <---interact---> speaker visual
#
#     self.transformer_fusion_ns2sa = TransformerDecoder(
#         decoder_layer, num_layer_ns2sa, nn.LayerNorm(self.latent_dim),
#         return_intermediate=return_intermediate_dec,
#     )
#
#     # noised sample <--> speaker visual
#     self.transformer_fusion_ns2sv = TransformerDecoder(
#         decoder_layer, num_layer_ns2sv, nn.LayerNorm(self.latent_dim),
#         return_intermediate=return_intermediate_dec,
#     )
#
#     # final interaction
#     self.transformer_fusion_final = TransformerDecoder(
#         decoder_layer, 1, nn.LayerNorm(self.latent_dim),
#         return_intermediate=return_intermediate_dec,
#     )

# if self.arch == "trans_enc":
#     if self.concat == "concat_first":
#         xseq = torch.cat((emb_latent, sample), dim=0)
#         xseq = self.query_pos(xseq)
#         tokens = self.encoder(xseq, src_key_padding_mask=src_key_padding_mask)
#         sample = tokens[embed_seq_len:]
#     elif self.concat == "concat_last":
#         xseq = torch.cat((sample, emb_latent), dim=0)
#         xseq = self.query_pos(xseq)
#         tokens = self.encoder(xseq, src_key_padding_mask=src_key_padding_mask)
#         sample = tokens[:embed_seq_len]
#     else:
#         raise NotImplementedError(f"{self.concat} is not supported.")
