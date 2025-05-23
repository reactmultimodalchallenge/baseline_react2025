import torch
import torch.nn as nn


class AudioEmbedder(nn.Module):
    def __init__(self, skip_norm=False, *args, **kwargs):
        super().__init__()
        self.skip_norm = skip_norm

    def _encode(self, x):
        if not self.skip_norm:
            x_min = torch.min(x)
            x_max = torch.max(x)
            return (x - x_min) / (x_max - x_min)
        else:
            return x
