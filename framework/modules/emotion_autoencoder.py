import torch
import torch.nn as nn
from einops import repeat
import numpy as np
from framework.motion_diffusion.diffusion.utils.temos_utils import lengths_to_mask


def sequence_slice(emb, start_indices, end_indices, max_seq_length):
    """
    :param emb: positional token embeddings | time query token embeddings
     (bz, l=5000, d_model)
    """
    B, L, D = emb.shape

    lengths = end_indices - start_indices + 1  # [B]
    M = int(lengths.max())  # Python int

    rel_pos = torch.arange(M, device=emb.device).unsqueeze(0).expand(B, M)  # [B, M]
    abs_pos = start_indices.unsqueeze(1) + rel_pos  # [B, M]
    abs_pos = abs_pos.clamp(0, L - 1)  # will mask out if out of bounds

    slice_batched = emb.gather(
        dim=1,
        index=abs_pos.unsqueeze(-1).expand(-1, -1, D)
    )

    mask = rel_pos < lengths.unsqueeze(1)
    slice_batched = slice_batched * mask.unsqueeze(-1)  # [B, M, D]

    assert M <= max_seq_length, "Length of sliced sequence exceeds max_seq_length"
    slice_batched = torch.cat((slice_batched,
                               torch.zeros(B, max_seq_length - M, D, device=emb.device)), dim=1)

    return slice_batched  # lengths


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, pe_type  = "absolute",
                 batch_first = True, max_len = 50000,):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        self.pe_type = pe_type

        if pe_type == "learnable":
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.uniform_(self.pe)
        elif pe_type == "absolute":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
        else:
            raise ValueError("Unknown positional encoding type: {}".format(pe_type))

    def forward(self, x, start_indices, end_indices, clip_length):
        if self.pe_type == "learnable":
            pe = repeat(self.pe, "1 l d -> b l d", b=x.shape[0])
        elif self.pe_type == "absolute":
            assert self.batch_first, "At the moment only batch_first=True"
            pe = repeat(self.pe.permute(1, 0, 2), "1 l d -> b l d", b=x.shape[0])
        else:
            raise ValueError("Unknown positional encoding type: {}".format(self.pe_type))

        pe = sequence_slice(pe, start_indices, end_indices, clip_length)
        x = x + pe[:, :x.shape[1], :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self,
                 d_model = 512,
                 nhead = 8,
                 num_layers = 6,
                 max_seq_len = 5000,
                 global_token_len = 128,
                 mlp_dist = False,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len

        if mlp_dist:
            self.global_token_len = global_token_len
        else:
            self.global_token_len = global_token_len * 2
        self.global_tokens = nn.Parameter(torch.zeros(1, self.global_token_len, d_model))
        self.reset_parameters()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers)

    def reset_parameters(self):
        nn.init.normal_(self.global_tokens, mean=0.0, std=0.04)  # 0.02?

    def forward(self,
                x: torch.Tensor,
                pe: nn.Module,
                start_indices: torch.Tensor,
                end_indices: torch.Tensor,
                clip_length: torch.Tensor,
                ):

        lengths = end_indices - start_indices + 1
        x = pe(x, start_indices, end_indices, clip_length)
        x = torch.cat((self.global_tokens.expand(x.shape[0], -1, -1), x), dim=1)
        mask = lengths_to_mask(lengths, device=x.device, max_len=clip_length)
        mask = torch.cat((torch.ones(x.shape[0], self.global_token_len, device=x.device), mask), dim=-1)
        src_key_padding_mask = ~(mask.bool())

        x = self.transformer_encoder(src=x, mask=None, src_key_padding_mask=src_key_padding_mask)
        return x[:, :self.global_token_len, :]


class Decoder(nn.Module):
    def __init__(self,
                 d_model = 512,
                 nhead = 8,
                 num_layers = 6,
                 max_seq_len = 5000,
                 query_type  = "learnable"):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.query_type = query_type

        if query_type == "learnable":
            self.time_query = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))
            # torch.nn.init.normal_(self.time_query, mean=0.0, std=0.04)
            nn.init.uniform_(self.time_query)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers)

    def forward(self, x: torch.Tensor,
                pe: nn.Module,
                start_indices: torch.Tensor,
                end_indices: torch.Tensor,
                clip_length: torch.Tensor,
                ):

        lengths = end_indices - start_indices + 1
        mask = lengths_to_mask(lengths, device=x.device, max_len=clip_length)

        if self.query_type == "learnable":
            time_query = sequence_slice(repeat(self.time_query, "1 l d -> b l d", b=x.shape[0]),
                                        start_indices, end_indices, clip_length)
        elif self.query_type == "zero":
            time_query = torch.zeros(x.shape[0], clip_length, self.d_model).to(x.device)
        else:
            raise ValueError("Unknown query type: {}".format(self.query_type))
        # (bsz, clip_len, d)
        time_query = pe(time_query, start_indices, end_indices, clip_length)

        x = self.transformer_decoder(tgt=time_query, memory=x, tgt_key_padding_mask=~mask)
        # padding_mask = mask.float().unsqueeze(-1).to(x.device)
        # x = x * padding_mask

        return x


class EmotionVAE(nn.Module):
    def __init__(self,
                 in_channels = 25,
                 out_channels = 25,
                 feature_dim = 512,
                 nhead = 8,
                 dropout = 0.1,
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 mlp_dist = False,  # expand mu & logvar
                 in_proj_type  = "linear",  # linear | mlp
                 out_proj_type  = "separate", # separate | shared
                 pe_type  = "learnable",
                 query_type  = "zero",
                 max_seq_len = 5000,
                 global_token_len = 128,
                 **kwargs,
                 ):

        super().__init__()
        self.feature_dim = feature_dim
        self.mlp_dist = mlp_dist
        self.in_proj_type = in_proj_type
        self.out_proj_type = out_proj_type
        self.pe_type = pe_type
        self.global_token_len = global_token_len

        self.PE = PositionalEncoding(d_model=feature_dim,
                                     dropout=dropout,
                                     pe_type=pe_type,
                                     max_len=max_seq_len)

        if self.in_proj_type == "mlp":
            self.in_proj = nn.Sequential(
                nn.Linear(in_channels, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        elif self.in_proj_type == "linear":
            self.in_proj = nn.Linear(in_channels, feature_dim)
        else:
            raise ValueError("Unknown input projection type: {}".format(self.in_proj_type))

        if self.out_proj_type == "separate":
            self.au_out_proj  = nn.Linear(feature_dim, 15)
            self.va_out_proj  = nn.Linear(feature_dim, 2)
            self.emo_out_proj = nn.Linear(feature_dim, 8)
        elif self.out_proj_type == "shared":
            self.out_proj = nn.Linear(feature_dim, out_channels)
        else:
            raise ValueError("Unknown output projection type: {}".format(self.out_proj_type))

        self.encoder = Encoder(
            d_model=feature_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            global_token_len=global_token_len,
            mlp_dist=mlp_dist,
        )

        if mlp_dist:
            self.mu_head = nn.Linear(feature_dim, feature_dim)
            self.logvar_head = nn.Linear(feature_dim, feature_dim)

        self.decoder = Decoder(
            d_model=feature_dim,
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            query_type=query_type,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x M x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x M x D]
        """
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)

        if deterministic:
            return mu, None

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = self.sample_from_distribution(dist).to(mu.device)
        return latent, dist

    def sample_from_distribution(self, distribution):
        return distribution.rsample()

    def forward(self, x: torch.Tensor,
                start_e: torch.Tensor,
                end_e: torch.Tensor,
                start_d: torch.Tensor,
                end_d: torch.Tensor,
                reparameterization: str = "random"):
        """
        :param x: input sequence, (batch_size, token_len, feature_dim)
        """
        B, L, D = x.shape
        x = self.in_proj(x)
        z = self.encoder(x, self.PE, start_e, end_e, L)  # L: seq_len

        if self.mlp_dist:
            mu = self.mu_head(z)
            logvar = self.logvar_head(z)
        else:
            mu = z[:, :self.global_token_len, :]
            logvar = z[:, self.global_token_len:, :]

        deterministic = reparameterization == "deterministic"
        latent, dist = self.reparameterize(mu, logvar, deterministic=deterministic)

        x = self.decoder(latent, self.PE, start_d, end_d, L)

        if self.out_proj_type == "shared":
            x = self.out_proj(x)
            au_out = x[:, :, :15]  # F.sigmoid(x[:, :, :15])
            va_out = x[:, :, 15:17]
            emo_out = x[:, :, 17:]  # F.softmax(x[:, :, 17:], dim=-1)
            out = (au_out, va_out, emo_out)  # torch.cat((au_out, va_out, emo_out), dim=-1)
        elif self.out_proj_type == "separate":
            au_out = self.au_out_proj(x)  # F.sigmoid(self.au_out_proj(x))
            va_out = self.va_out_proj(x)
            emo_out = self.emo_out_proj(x)  # F.softmax(self.emo_out_proj(x), dim=-1)
            out = (au_out, va_out, emo_out)  # torch.cat((au_out, va_out, emo_out), dim=-1)
        else:
            raise ValueError("Unknown output projection type: {}".format(self.out_proj_type))

        out_padding_mask = lengths_to_mask(
            lengths=(end_d - start_d + 1),
            device=x.device,
            max_len=L,
        ).float().unsqueeze(-1)
        out = (o * out_padding_mask for o in out)

        return out, latent, dist, out_padding_mask

    def get_model_name(self):
        return self.__class__.__name__