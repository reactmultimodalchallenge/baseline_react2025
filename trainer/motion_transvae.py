import math
from pathlib import Path
from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_metrics
from framework.utils.losses import div_loss
from framework.utils.util import AverageMeter, from_pretrained_checkpoint

os.environ["NUMEXPR_MAX_THREADS"] = '16'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 renderer: DictConfig = None,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):

        self.renderer_cfg = renderer
        self.model_cfg = model
        self.criterion_cfg = criterion
        self.resumed_training = resumed_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = kwargs.pop('lr')
        self.optim_cfg = kwargs.pop("optim")
        self.epochs = kwargs.pop("epochs")
        self.gpu_ids = kwargs.pop("gpu_ids")
        self.j = kwargs.pop("j")
        self.max_seq_len = kwargs.pop("max_seq_len")
        self.window_size = kwargs.pop("window_size")
        self.div_p = kwargs.pop("div_p")
        self.task = kwargs.pop("task")
        self.kwargs = kwargs

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

    def set_data_module(self, data_module):
        self.data_module = data_module

    def data_resample(self, speaker_audio_clips, speaker_video_clips, listener_emotion_clips,
                      listener_3dmm_clips, speaker_seq_lengths, listener_seq_lengths):
        speaker_audios = [audio[:L] for audio, L in zip(speaker_audio_clips, speaker_seq_lengths)]
        speaker_videos = [video[:L] for video, L in zip(speaker_video_clips, speaker_seq_lengths)]
        listener_emotions = [emo[:L] for emo, L in zip(listener_emotion_clips, listener_seq_lengths)]
        listener_3dmm = [param[:L] for param, L in zip(listener_3dmm_clips, listener_seq_lengths)]
        return speaker_audios, speaker_videos, listener_emotions, listener_3dmm

    def fit(self):
        """
        # relative directory
        root_dir = save/${trainer.task_name}/${data.data_name}/${folder_name}
        # absolute directory
        saving_dir = Path(hydra.utils.to_absolute_path(root_dir))
        # get saving path
        saving_path = str(saving_dir / ...)
        """
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = (
            self.data_module.get_dataloader(stage=stage))
        logger.info("Data module loaded")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        logger.info("Criterion loaded")

        self.main()

    def main(self):
        model = instantiate(self.model_cfg, _recursive_=False)
        model.to(self.device)
        optimizer = instantiate(self.optim_cfg, lr=self.lr, params=model.parameters())

        if self.resumed_training:
            checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", last=True)
            from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            lowest_val_loss, start_epoch = from_pretrained_checkpoint(checkpoint_path, model, self.device)
            logger.info(f"Resume training from epoch {start_epoch}")
        else:
            start_epoch = 0
            lowest_val_loss = float('inf')
        print(f"Best validation loss: {lowest_val_loss}")

        for epoch in range(start_epoch, self.epochs):
            train_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, div_loss = self.train(model, optimizer)
            logger.info("Epoch: {}  train_loss: {:.5f}  rec_all_loss: {:.5f}  rec_emo_loss: {:.5f}  "
                        "rec_parma_loss: {:.5f}  kld_loss: {:.5f}  div_loss: {:.5f}"
                  .format(epoch + 1, train_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, div_loss))

            if (epoch + 1) % 5 == 0:
                val_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss = self.val(model)
                logger.info("Epoch: {}  val_loss: {:.5f}  rec_all_loss: {:.5f}  rec_emo_loss: {:.5f}  "
                            "rec_param_loss: {:.5f}  kld_loss: {:.5f}"
                      .format(epoch + 1, val_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss))

                checkpoint = {
                    'epoch': epoch + 1,
                    'best_loss': lowest_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    ckpt_path = self.get_ckpt_path(model, best=True)
                    logger.info(f"Saving best checkpoint, val_loss: {lowest_val_loss:.5f}, ckpt_path: {ckpt_path}")
                    checkpoint['best_loss'] = lowest_val_loss
                    torch.save(checkpoint, ckpt_path)

                ckpt_path = self.get_ckpt_path(model, epoch=(epoch + 1))
                torch.save(checkpoint, ckpt_path)
                ckpt_path = self.get_ckpt_path(model, last=True)
                torch.save(checkpoint, ckpt_path)

    # Train
    def train(self, model, optimizer):
        losses = AverageMeter()
        rec_losses = AverageMeter()
        rec_emo_losses = AverageMeter()
        rec_param_losses = AverageMeter()
        kld_losses = AverageMeter()
        div_losses = AverageMeter()

        model.train()
        for batch_idx, (speaker_audio_clip,
                        speaker_video_clip,
                        _, _, _,
                        listener_emotion,
                        listener_3dmm,
                        speaker_clip_length,
                        listener_clip_length) in enumerate(tqdm(self.train_loader)):

            if self.model_cfg.task == 'offline':
                (speaker_audio_clip, speaker_video_clip, listener_emotion, listener_3dmm) = self.data_resample(
                    speaker_audio_clip, speaker_video_clip, listener_emotion,
                    listener_3dmm, speaker_clip_length, listener_clip_length,)
            optimizer.zero_grad()
            listener_3dmm_out, listener_emotion_out, distribution = model(
                speaker_video_clip, speaker_audio_clip, motion_lengths=speaker_clip_length)

            loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss = self.criterion(
                listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out, distribution)
            with torch.no_grad():
                listener_3dmm_out_, listener_emotion_out_, _ = model(speaker_video_clip, speaker_audio_clip)

            d_loss = (div_loss(listener_3dmm_out_, listener_3dmm_out) +
                      div_loss(listener_emotion_out_, listener_emotion_out))
            loss = loss + self.div_p * d_loss

            batch_size = len(speaker_video_clip)
            losses.update(loss.data.item(), batch_size)
            rec_losses.update(rec_loss.data.item(), batch_size)
            rec_emo_losses.update(rec_emo_loss.data.item(), batch_size)
            rec_param_losses.update(rec_param_loss.data.item(), batch_size)
            kld_losses.update(kld_loss.data.item(), batch_size)
            div_losses.update(d_loss.data.item(), batch_size)

            loss.backward()
            optimizer.step()
        return losses.avg, rec_losses.avg, rec_emo_losses.avg, rec_param_losses.avg, kld_losses.avg, div_losses.avg

    # Validation
    def val(self, model):
        losses = AverageMeter()
        rec_losses = AverageMeter()
        rec_emo_losses = AverageMeter()
        rec_param_losses = AverageMeter()
        kld_losses = AverageMeter()
        model.eval()
        model.reset_window_size(8)

        for batch_idx, (speaker_audio_clip,
                        speaker_video_clip,
                        _, _, _,
                        listener_emotion,
                        listener_3dmm,
                        speaker_clip_length,
                        listener_clip_length) in enumerate(tqdm(self.val_loader)):
            if self.model_cfg.task == 'offline':
                (speaker_audio_clip, speaker_video_clip, listener_emotion, listener_3dmm) = self.data_resample(
                    speaker_audio_clip, speaker_video_clip, listener_emotion,
                    listener_3dmm, speaker_clip_length, listener_clip_length,)

            with (torch.no_grad()):
                listener_3dmm_out, listener_emotion_out, distribution = model(
                    speaker_video_clip, speaker_audio_clip, motion_lengths=speaker_clip_length)
                loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss = \
                    self.criterion(listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out, distribution)

                batch_size = len(speaker_video_clip)
                losses.update(loss.data.item(), batch_size)
                rec_losses.update(rec_loss.data.item(), batch_size)
                rec_emo_losses.update(rec_emo_loss.data.item(), batch_size)
                rec_param_losses.update(rec_param_loss.data.item(), batch_size)
                kld_losses.update(kld_loss.data.item(), batch_size)

        model.reset_window_size(self.window_size)
        return losses.avg, rec_losses.avg, rec_emo_losses.avg, rec_param_losses.avg, kld_losses.avg

    def pad_to(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        L = seq.shape[0]
        if L < length:
            pad_shape = (length - L, *seq.shape[1:])
            return torch.cat([seq, seq.new_zeros(pad_shape)], dim=0)
        return seq

    def test(self):
        stage = "test"
        data_clamp = self.kwargs.pop("data_clamp")

        model = instantiate(self.model_cfg, _recursive_=False)
        checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", best=True)
        # checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", epoch=30)
        from_pretrained_checkpoint(checkpoint_path, model, self.device)
        model.eval()

        # instantiate renderer
        renderer = instantiate(self.renderer_cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Test data module loaded")

        logger.info("Loading post processor")
        post_processor = Processor(config_name=self.kwargs.pop("post_config_name"),
                                   clip_len_test=self.kwargs.pop("post_clip_length"),
                                   device=self.device)
        logger.info("Post processor loaded")

        speaker_emotions_input_all = []
        listener_3dmm_preds_lists_all = []
        listener_emotion_preds_lists_all = []
        listener_3dmm_GTs_all = []
        listener_emotion_GTs_all = []
        max_seq_len = self.max_seq_len
        num_preds = 10

        for batch_idx, (speaker_audio_clips, speaker_video_clips, speaker_emotion_clips, _,
                        listener_video_clips, listener_emotions, listener_3dmms, speaker_clip_lengths,
                        listener_clip_lengths) in enumerate(tqdm(test_loader)):

            listener_3dmm_preds = []
            listener_emotion_preds = []
            for i in range(num_preds):
                if i == 0:
                    listener_emotion_GTs_all.extend(listener_emotions)
                    listener_3dmm_GTs_all.extend(listener_3dmms)
                    speaker_emotions_input_all.extend(speaker_emotion_clips)

                for j, (speaker_audio_clip, speaker_video_clip, listener_video_clip, speaker_clip_length) in (
                        enumerate(zip(speaker_audio_clips, speaker_video_clips, listener_video_clips, speaker_clip_lengths))):

                    speaker_audio_clip_list = []
                    speaker_video_clip_list = []
                    motion_length_list = []

                    # split into sub-clips
                    for k in range(math.ceil(speaker_clip_length / max_seq_len)):
                        start_idx = k * max_seq_len
                        end_idx = min((k + 1) * max_seq_len, speaker_clip_length)
                        motion_length = end_idx - start_idx
                        motion_length_list.append(motion_length)
                        speaker_audio_clip_list.append(speaker_audio_clip[start_idx:end_idx])
                        speaker_video_clip_list.append(speaker_video_clip[start_idx:end_idx])

                    if self.model_cfg.task == 'online':
                        speaker_audio_clip_list[-1] = self.pad_to(speaker_audio_clip_list[-1], max_seq_len)
                        speaker_video_clip_list[-1] = self.pad_to(speaker_video_clip_list[-1], max_seq_len)
                    speaker_audio_clip_inputs = speaker_audio_clip_list  # List: [tensor([l, d_audio]), ...]
                    speaker_video_clip_inputs = speaker_video_clip_list  # List: [tensor([l, 3, 224, 224]), ...]

                    with torch.no_grad():
                        listener_3dmm_out, listener_emotion_out, _ = model(
                            speaker_video_clip_inputs,
                            speaker_audio_clip_inputs,
                            motion_lengths=torch.tensor(motion_length_list)
                        )
                        # List: [tensor([l, 58]), ...]
                        # List: [tensor([l, 25]), ...]

                        listener_3dmm_out = torch.cat(listener_3dmm_out, dim=0)[:speaker_clip_length]
                        listener_3dmm_out = listener_3dmm_out.detach().cpu().unsqueeze(0)
                        # one data sample (origin_len, d)
                        listener_emotion_out = torch.cat(listener_emotion_out, dim=0)[:speaker_clip_length]
                        listener_emotion_out = listener_emotion_out.detach().cpu().unsqueeze(0)

                        if data_clamp:
                            listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])

                        if self.renderer_cfg.do_render and i == 0:  # (batch_idx % 20) == 0
                            listener_video_clip = listener_video_clip[0].to(self.device)
                            val_path = os.path.join('results_videos', 'test')
                            os.makedirs(val_path, exist_ok=True)

                            perm = torch.randperm(listener_video_clip.shape[0])
                            listener_references = listener_video_clip[perm[0]]
                            assert len(listener_references.shape) == 3, \
                                "listener_references.shape should be (3, 224, 224)"
                            renderer.rendering(val_path,
                                               f"batch{str(batch_idx + 1)}",
                                               listener_3dmm_out.to(self.device),
                                               speaker_video_clip,
                                               listener_references,
                                               listener_video_clip)

                    if i == 0:
                        listener_3dmm_preds.append(listener_3dmm_out)
                        listener_emotion_preds.append(listener_emotion_out)
                    else:
                        listener_3dmm_preds[j] = torch.cat(
                            (listener_3dmm_preds[j], listener_3dmm_out), dim=0)
                        listener_emotion_preds[j] = torch.cat(
                            (listener_emotion_preds[j], listener_emotion_out), dim=0)

            # listener_3dmm_preds: (num_preds, l, ...)
            listener_3dmm_preds_lists_all.extend(listener_3dmm_preds)
            # listener_emotion_preds: (num_preds, l, ...)
            listener_emotion_preds_lists_all.extend(listener_emotion_preds)

        # listener_emotion_preds_lists_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]
        # listener_emotion_GTs_all
        # List: 750 [List: [(l', 25), (l'', 25), ...], List: [(l''', 25), (l'''', 25)], ...]
        listener_emotion_GTs_all = post_processor.forward(
            prediction_list=listener_emotion_preds_lists_all,
            target_list=listener_emotion_GTs_all,)
        # listener_emotion_GTs_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]

        try:
            torch.save({'GT': listener_emotion_GTs_all, 'PRED': listener_emotion_preds_lists_all}, 'results.pt')
            print("Successfully saved Tensor List")
        except Exception:
            print("Failed to save Tensor List")

        results = compute_metrics(
            speaker_emotions_input_all,
            listener_emotion_preds_lists_all,
            listener_emotion_GTs_all,
        )
        logger.info(results)