# import math
# import os
# from pathlib import Path
# import random
# import hydra
# from omegaconf import DictConfig
# import torch
# import torch.nn as nn
# from framework.utils.util import from_pretrained_checkpoint
# from utils.util import get_tensorboard_path, AverageMeter, get_lr
# from torch import optim
# from torch.nn.utils import clip_grad_norm_
# from tqdm import tqdm
# from hydra.utils import instantiate
# from torch.utils.tensorboard import SummaryWriter
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# class Trainer:
#     def __init__(self,
#                  resumed_training: bool = False,
#                  model: DictConfig = None,
#                  criterion: DictConfig = None,
#                  **kwargs):
#         super().__init__()
#         self.resumed_training = resumed_training
#         self.model_cfg = model
#         self.criterion_cfg = criterion
#
#         if torch.cuda.device_count() > 0:
#             device = torch.device('cuda:0')
#         else:
#             device = torch.device('cpu')
#         self.device = device
#
#         self.clip_length = kwargs.pop("clip_length")
#         self.window_size = kwargs.pop("window_size") * kwargs.pop("s_ratio")
#         self.start_epoch = kwargs.pop("start_epoch")
#         self.epochs = kwargs.pop("epochs")
#         self.tb_dir = kwargs.pop("tb_dir")
#         self.val_period = kwargs.pop("val_period")
#         self.lr = kwargs.pop("lr")
#         self.weight_decay = kwargs.pop("weight_decay")
#         self.beta = kwargs.pop("beta")
#         self.kwargs = kwargs
#
#     def set_data_module(self, data_module):
#         self.data_module = data_module
#
#     def get_ckpt_path(self, model, runid="current_runid", epoch=None, best=False, last=False):
#         ckpt_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
#         run_id = Path(self.kwargs.get(runid))
#         ckpt_dir = str(ckpt_dir / run_id / model.get_model_name())
#         os.makedirs(ckpt_dir, exist_ok=True)
#
#         ckpt_path = None
#         if epoch is not None:
#             ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
#         if best:
#             ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
#         if last:
#             ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
#         assert ckpt_path is not None, "No checkpoint path is provided."
#         return ckpt_path
#
#     def data_resample(self, emotion_clips, params_clips, seq_lengths):
#         emotion_clip_list = []
#         params_clip_list = []
#         for emotion_clip, params_clip, seq_length in zip(emotion_clips, params_clips, seq_lengths):
#             emotion_clip = emotion_clip[:seq_length]
#             params_clip = params_clip[:seq_length]
#
#             if seq_length < self.clip_length:
#                 cp = random.randint(0, seq_length - self.window_size) if seq_length > self.window_size else 0
#             else:
#                 cp = random.randint(0, self.clip_length - self.window_size)
#
#             if seq_length < self.window_size:
#                 emotion_clip = torch.cat((emotion_clip, torch.zeros(size=(self.window_size - seq_length,
#                                                                           emotion_clip.shape[-1]))), dim=0)
#                 params_clip = torch.cat((params_clip, torch.zeros(size=(self.window_size - seq_length,
#                                                                         params_clip.shape[-1]))), dim=0)
#
#             emotion_clip = emotion_clip[cp:cp + self.window_size]
#             emotion_clip_list.append(emotion_clip)
#             params_clip = params_clip[cp:cp + self.window_size]
#             params_clip_list.append(params_clip)
#
#         emotion_clips = torch.stack(emotion_clip_list, dim=0)  # (bs, w, d)
#         params_clips = torch.stack(params_clip_list, dim=0)
#
#         return emotion_clips, params_clips
#
#     def fit(self):
#         stage = "fit"
#
#         logger.info("Loading data module")
#         self.train_loader, self.val_loader = self.data_module.get_dataloader(stage=stage)
#         logger.info("Data module loaded")
#
#         logger.info("Loading criterion")
#         self.criterion = instantiate(self.criterion_cfg)
#         logger.info("Criterion loaded")
#
#         logger.info("Loading writer")
#         self.writer = SummaryWriter(self.tb_dir)
#         logger.info(f"Writer loaded: {self.tb_dir}")
#
#         self.main_autoencoder()
#
#     def compute_metrics(self, original, reconstructed):
#         """
#         Compute evaluation metrics between original and reconstructed emotion data
#         """
#         # Example metrics: MSE, MAE
#         total_mse = 0
#         total_mae = 0
#         count = 0
#
#         for orig, recon in zip(original, reconstructed):
#             # Make sure they have the same length
#             min_len = min(orig.size(0), recon.size(0))
#             orig = orig[:min_len]
#             recon = recon[:min_len]
#
#             mse = torch.mean((orig - recon) ** 2)
#             mae = torch.mean(torch.abs(orig - recon))
#
#             total_mse += mse.item()
#             total_mae += mae.item()
#             count += 1
#
#         avg_mse = total_mse / count if count > 0 else 0
#         avg_mae = total_mae / count if count > 0 else 0
#
#         logger.info(f"Test MSE: {avg_mse:.5f}")
#         logger.info(f"Test MAE: {avg_mae:.5f}")
#
#     def main_autoencoder(self):
#         model = instantiate(self.model_cfg,
#                             _recursive_=False)
#         model.to(self.device)
#
#         optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr,
#                                       weight_decay=self.weight_decay, betas=self.beta)
#         if self.resumed_training:
#             checkpoint_path = self.get_ckpt_path(model.autoencoder, runid="resume_runid", best=True)
#             from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
#
#         best_recon_loss = float('inf')
#         for epoch in range(self.start_epoch, self.epochs):
#             train_loss, rec_loss, kld_loss, coeff_loss = self.train(
#                 model, self.train_loader, optimizer, self.criterion, epoch, self.writer, self.device
#             )
#             logging.info(
#                 "Epoch: {} train_whole_loss: {:.5f} train_rec_loss: {:.5f} train_kld_loss: {:.5f} train_coeff_loss: {:.5f}"
#                 .format(epoch + 1, train_loss, rec_loss, kld_loss, coeff_loss))
#
#             if (epoch + 1) % self.val_period == 0:
#                 val_loss, rec_loss, kld_loss, coeff_loss = (
#                     self.val(model, self.val_loader, self.criterion, self.device))
#                 checkpoint_path = self.get_ckpt_path(model, epoch=(epoch + 1))
#                 model.save_ckpt(checkpoint_path, optimizer)
#
#                 logging.info(
#                     "Epoch: {} val_whole_loss: {:.5f} val_rec_loss: {:.5f} val_kld_loss: {:.5f} val_coeff_loss: {:.5f}"
#                     .format(epoch + 1, val_loss, rec_loss, kld_loss, coeff_loss))
#
#                 if val_loss < best_recon_loss:
#                     best_recon_loss = val_loss
#                     logging.info(
#                         f"New best reconstruction loss ({best_recon_loss:.5f}) at epoch {epoch + 1}, saving checkpoint."
#                     )
#                     checkpoint_path = self.get_ckpt_path(model, best=True)
#                     model.save_ckpt(checkpoint_path, optimizer)
#
#         checkpoint_path = self.get_ckpt_path(model, last=True)
#         model.save_ckpt(checkpoint_path, optimizer)
#
#     def train(self, model, data_loader, optimizer, criterion, epoch, writer, device):
#         whole_losses = AverageMeter()
#         rec_losses = AverageMeter()
#         kld_losses = AverageMeter()
#         div_losses = AverageMeter()
#
#         model.train()
#         for batch_idx, (_, _,
#                         emotion_clips,
#                         params_clips,
#                         _, _, _, _,
#                         seq_lengths) in enumerate(tqdm(data_loader)):
#
#             emotion_data, params_data = self.data_resample(emotion_clips, params_clips, seq_lengths)
#             emotion_data = emotion_data.to(device)
#             params_data = params_data.to(device)
#             batch_size = emotion_data.shape[0]
#
#             outputs = model(emotion=emotion_data, _3dmm=params_data)
#             loss_output = criterion(**outputs)
#             loss, rec_loss, kld_loss, div_loss = (
#                 loss_output["loss"], loss_output["mse"], loss_output["kld"], loss_output["coeff"]
#             )
#
#             # Log metrics
#             iteration = batch_idx + len(data_loader) * epoch
#             if writer is not None:
#                 writer.add_scalar("Train/rec_loss", rec_loss.data.item(), iteration)
#                 writer.add_scalar("Train/kld_loss", kld_loss.data.item(), iteration)
#                 writer.add_scalar("Train/div_loss", div_loss.data.item(), iteration)
#
#             whole_losses.update(loss.data.item(), batch_size)
#             rec_losses.update(rec_loss.data.item(), batch_size)
#             kld_losses.update(kld_loss.data.item(), batch_size)
#             div_losses.update(div_loss.data.item(), batch_size)
#
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         return whole_losses.avg, rec_losses.avg, kld_losses.avg, div_losses.avg
#
#     def val(self, model, data_loader, criterion, device):
#         whole_losses = AverageMeter()
#         rec_losses = AverageMeter()
#         kld_losses = AverageMeter()
#         div_losses = AverageMeter()
#
#         model.eval()
#         for batch_idx, (_, _,
#                         emotion_clips,
#                         params_clips,
#                         _, _, _, _,
#                         seq_lengths) in enumerate(tqdm(data_loader)):
#             emotion_data, params_data = self.data_resample(emotion_clips, params_clips, seq_lengths)
#             emotion_data = emotion_data.to(device)
#             params_data = params_data.to(device)
#             batch_size = emotion_data.shape[0]
#
#             with torch.no_grad():
#                 outputs = model(emotion=emotion_data, _3dmm=params_data)
#                 loss_output = criterion(**outputs)
#                 loss, rec_loss, kld_loss, div_loss = (
#                     loss_output["loss"], loss_output["mse"], loss_output["kld"], loss_output["coeff"]
#                 )
#             # print(f"batch {batch_idx}, loss = {loss.data.item():.6f}")
#
#             whole_losses.update(loss.data.item(), batch_size)
#             rec_losses.update(rec_loss.data.item(), batch_size)
#             kld_losses.update(kld_loss.data.item(), batch_size)
#             div_losses.update(div_loss.data.item(), batch_size)
#
#         return whole_losses.avg, rec_losses.avg, kld_losses.avg, div_losses.avg
