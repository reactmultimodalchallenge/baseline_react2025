from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import os
from framework.utils.util import from_pretrained_checkpoint


class Processor:
    def __init__(self,
                 config_name: str = "configs/model/emotion_autoencoder.yaml",
                 ckpt_dir: str = "pretrained_models/post_processor",
                 device: Optional[torch.device] = None,
                 clip_len_test: int = 1000,
                 cfg_dir: str = None,
                 **kwargs):
        if cfg_dir is None:
            cfg_dir = hydra.utils.get_original_cwd()
        cfg = OmegaConf.load(os.path.join(cfg_dir, config_name))
        self.model = instantiate(cfg, _recursive_=False)
        self.clip_len_test = clip_len_test
        self.num_preds = kwargs.get("num_preds", 10)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        ckpt_path = self.get_ckpt_path(ckpt_dir)
        from_pretrained_checkpoint(ckpt_path, self.model, device=device)
        self.model.eval()

    def get_ckpt_path(self, ckpt_dir):
        ckpt_path = Path(hydra.utils.to_absolute_path(ckpt_dir)) / 'checkpoint.pth'
        assert ckpt_path.is_file(), f"Checkpoint file not found at {ckpt_dir}"
        return ckpt_path

    def forward(self, prediction_list, target_list):
        processed_target_list = []

        for predictions, targets in zip(prediction_list, target_list):
            if len(predictions.shape) == 2:
                predictions = repeat(predictions, 'l d -> n l d', n=self.num_preds)

            # predictions: Tensor([num_preds, l, d])
            # targets: List: [(l', d), (l'', d), ...]
            _, pred_seq_len, dim = predictions.shape

            processed_targets = []
            for tgt in targets:  # tgt: Tensor([l', 25])
                tgt_seq_len = tgt.shape[0]
                max_len = self.clip_len_test

                if pred_seq_len == tgt_seq_len:
                    processed_targets.append(tgt)
                    continue

                if pred_seq_len > tgt_seq_len:
                    num_segments = math.ceil(pred_seq_len / max_len)
                    assert tgt_seq_len >= num_segments
                    min_len = math.ceil(tgt_seq_len / num_segments)
                    total_len = num_segments * min_len

                    tgt = torch.cat(
                        (tgt,
                         torch.zeros(size=(int(total_len - tgt_seq_len), dim))
                         ), dim=0)
                    tgt = rearrange(tgt, '(b l) d -> b l d', b=num_segments)
                    tgt = torch.cat((tgt, torch.zeros(num_segments, max_len - min_len, dim)), dim=1)

                    out_start_indices = torch.zeros(size=(num_segments,))
                    input_start_indices = torch.zeros(size=(num_segments,))
                    if num_segments == 1:
                        out_end_indices = torch.tensor([pred_seq_len - 1])
                        input_end_indices = torch.tensor([tgt_seq_len - 1])
                    else:
                        if pred_seq_len % max_len == 0:
                            out_end_indices = torch.tensor([max_len - 1] * num_segments)
                        else:
                            out_end_indices = torch.cat((torch.tensor([max_len - 1] * (num_segments - 1)),
                                                         torch.tensor([pred_seq_len % max_len - 1])))
                        if tgt_seq_len % min_len == 0:
                            input_end_indices = torch.tensor([min_len - 1] * num_segments)
                        else:
                            input_end_indices = torch.cat((torch.tensor([min_len - 1] * (num_segments - 1)),
                                                           torch.tensor([tgt_seq_len % min_len - 1])))

                else:
                    num_segments = math.ceil(tgt_seq_len / max_len)
                    assert pred_seq_len >= num_segments
                    total_len = num_segments * max_len
                    min_len = math.ceil(pred_seq_len / num_segments)

                    tgt = torch.cat(
                        (tgt,
                         torch.zeros(size=(int(total_len - tgt_seq_len), dim))
                         ), dim=0)
                    tgt = rearrange(tgt, '(b l) d -> b l d', b=num_segments)

                    out_start_indices = torch.zeros(size=(num_segments,))
                    input_start_indices = torch.zeros(size=(num_segments,))
                    if num_segments == 1:
                        out_end_indices = torch.tensor([pred_seq_len - 1])
                        input_end_indices = torch.tensor([tgt_seq_len - 1])
                    else:
                        if tgt_seq_len % max_len == 0:
                            input_end_indices = torch.tensor([max_len - 1] * num_segments)
                        else:
                            input_end_indices = torch.cat((torch.tensor([max_len - 1] * (num_segments - 1)),
                                                           torch.tensor([tgt_seq_len % max_len - 1])))
                        if pred_seq_len % min_len == 0:
                            out_end_indices = torch.tensor([min_len - 1] * num_segments)
                        else:
                            out_end_indices = torch.cat((torch.tensor([min_len - 1] * (num_segments - 1)),
                                                         torch.tensor([pred_seq_len % min_len - 1])))

                lengths = (out_end_indices - out_start_indices + 1).long()
                inputs, input_start_indices, input_end_indices, out_start_indices, out_end_indices = \
                    tgt.to(self.device), input_start_indices.to(self.device), input_end_indices.to(self.device), \
                        out_start_indices.to(self.device), out_end_indices.to(self.device)

                outputs = self.model(
                    inputs,
                    input_start_indices.long(),
                    input_end_indices.long(),
                    out_start_indices.long(),
                    out_end_indices.long(),
                )[0]  # (bsz, total_len, d)

                processed_target = []
                for i, (out_au, out_va, out_em) in enumerate(zip(*outputs)):
                    _len = lengths[i]
                    out_au =  (F.sigmoid(out_au) >= 0.5).float()
                    out_em = F.softmax(out_em, dim=-1).float()
                    out_all = torch.cat((out_au, out_va, out_em), dim=-1)[:_len].detach().cpu()
                    processed_target.append(out_all)
                processed_target = torch.cat(processed_target, dim=0)
                processed_targets.append(processed_target)  # len equal to speaker's

            processed_targets = torch.stack(processed_targets, dim=0)
            processed_target_list.append(processed_targets)
            # prediction_list, processed_target_list
            # List [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]

        return processed_target_list
