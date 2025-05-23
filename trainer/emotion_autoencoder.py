import os
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from framework.utils.util import from_pretrained_checkpoint
from utils.util import AverageMeter, get_lr
from tqdm import tqdm
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):
        super().__init__()
        self.resumed_training = resumed_training
        self.model_cfg = model
        self.criterion_cfg = criterion

        if torch.cuda.device_count() > 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device

        self.clip_length = kwargs.pop("clip_length")
        self.start_epoch = kwargs.pop("start_epoch")
        self.epochs = kwargs.pop("epochs")
        self.tb_dir = kwargs.pop("tb_dir")
        self.val_period = kwargs.pop("val_period")
        self.lr = kwargs.pop("lr")
        self.weight_decay = kwargs.pop("weight_decay")
        self.beta = kwargs.pop("beta")
        self.kwargs = kwargs

    def set_data_module(self, data_module):
        self.data_module = data_module

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

    def compute_metrics(self, original, reconstructed):
        """
        Compute evaluation metrics between original and reconstructed emotion data
        """
        # Example metrics: MSE, MAE
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))

        return {
            "mse": mse,
            "mae": mae,
        }

    def eval_au_binary(self, pred_list, tgt_list):
        assert len(pred_list) == len(tgt_list)
        accs, precs, recs, f1s = [], [], [], []

        for preds, tgts in zip(pred_list, tgt_list):
            preds_bin = preds.astype(int)

            print("number of positive samples: ", np.sum(tgts))
            print("number of negative samples: ", np.sum(1-tgts))
            print("number of positive samples predicted: ", np.sum(preds_bin))
            print("number of negative samples predicted: ", np.sum(1-preds_bin))

            accs.append(accuracy_score(tgts, preds_bin))
            p, r, f1, _ = precision_recall_fscore_support(
                tgts,
                preds_bin,
                average='binary',
                zero_division=0
            )
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

        results = {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1': float(np.mean(f1s)),
        }
        # 'micro_f1': micro_f1,
        # 'macro_f1': macro_f1,
        return results

    def test(self):
        stage = "test"
        device = self.device
        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(
            stage=stage, collate_fn='none')
        logger.info("Test data module loaded")

        model = instantiate(self.model_cfg,
                            _recursive_=False)
        ckpt_path = self.get_ckpt_path(model, runid="resume_runid", best=True)
        from_pretrained_checkpoint(ckpt_path, model, device)
        model.eval()

        au_predictions_all = []
        va_predictions_all = []
        emotion_predictions_all = []
        emotion_targets_all = []
        au_targets_all = []
        va_targets_all = []

        logger.info("Model testing started ...")
        for batch_idx, (input_emotion_clips, e_start_indices, e_end_indices,
                        output_emotion_clips, d_start_indices, d_end_indices) \
                in enumerate(tqdm(test_loader)):

            (input_emotion_clips, e_start_indices, e_end_indices, d_start_indices, d_end_indices) = \
            (input_emotion_clips.to(device), e_start_indices.to(device), e_end_indices.to(device),
             d_start_indices.to(device), d_end_indices.to(device))

            with torch.no_grad():
                out, _, dist, mask = (
                    model(input_emotion_clips, e_start_indices, e_end_indices, d_start_indices, d_end_indices,
                          reparameterization="deterministic")
                )

            lengths = (d_end_indices - d_start_indices + 1).detach().cpu().numpy()

            au_logits, va_logits, emotion_logits = out
            au_predictions = (F.sigmoid(au_logits) >= 0.5).float()
            au_predictions = (au_predictions * mask).detach().cpu().numpy()
            au_prediction_list = np.array([])
            for au_pred, length in zip(au_predictions, lengths):
                au_prediction_list = np.concatenate([au_prediction_list, au_pred[:length].reshape(-1)], axis=0)
            au_predictions_all.append(au_prediction_list)

            va_predictions_all.append((va_logits * mask).detach().cpu().numpy())
            emotion_predictions_all.append((F.softmax(emotion_logits, dim=-1) * mask).detach().cpu().numpy())

            au_targets, va_targets, emotion_targets = \
                (output_emotion_clips[:, :, :15].numpy(),
                 output_emotion_clips[:, :, 15:17].numpy(),
                 output_emotion_clips[:, :, 17:].numpy())

            au_target_list = np.array([])
            for au_target, length in zip(au_targets, lengths):
                au_target_list = np.concatenate([au_target_list, au_target[:length].reshape(-1)], axis=0)
            au_targets_all.append(au_target_list)

            va_targets_all.append(va_targets)
            emotion_targets_all.append(emotion_targets)

        va_predictions_all = np.concatenate(va_predictions_all, axis=0)
        va_targets_all = np.concatenate(va_targets_all, axis=0)
        emotion_predictions_all = np.concatenate(emotion_predictions_all, axis=0)
        emotion_targets_all = np.concatenate(emotion_targets_all, axis=0)

        au_results = self.eval_au_binary(au_predictions_all, au_targets_all)
        logger.info(f"AU results: {au_results}")
        va_mse_results = self.compute_metrics(va_targets_all, va_predictions_all)
        logger.info(f"VA results: {va_mse_results}")
        emotion_mse_results = self.compute_metrics(emotion_targets_all, emotion_predictions_all)
        logger.info(f"Emotion results: {emotion_mse_results}")

    def fit(self):
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = self.data_module.get_dataloader(
            stage=stage, collate_fn='none')
        logger.info("Data module loaded")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        logger.info("Criterion loaded")

        logger.info("Loading writer")
        self.writer = SummaryWriter(self.tb_dir)
        logger.info(f"Writer loaded: {self.tb_dir}")
        self.main_autoencoder()

    def main_autoencoder(self):
        model = instantiate(self.model_cfg,
                            _recursive_=False)
        model.to(self.device)

        # Load optimizer
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay, betas=self.beta)

        if self.resumed_training:
            checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", last=True)
            from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            best_val_loss, self.start_epoch = from_pretrained_checkpoint(checkpoint_path, model, self.device)
            logger.info(f"Resume training from epoch {self.start_epoch}")
        else:
            best_val_loss = float('inf')
        print(f"Best validation loss: {best_val_loss}")

        for epoch in range(self.start_epoch, self.epochs):
            train_loss, kld_loss, au_loss, va_loss, em_loss = self.train_autoencoder(
                model, self.train_loader, optimizer,
                self.criterion, epoch, self.writer, self.device
            )
            logging.info(f"Epoch: {epoch + 1} train_loss: {train_loss:.5f} kld_loss: {kld_loss:.5f} "
                         f"au_loss: {au_loss:.5f} va_loss: {va_loss:.5f} em_loss: {em_loss:.5f}")

            if (epoch + 1) % self.val_period == 0:
                val_loss, kld_loss, au_loss, va_loss, em_loss = self.val_autoencoder(
                    model, self.val_loader, self.criterion, self.device
                )
                logging.info(f"Epoch: {epoch + 1} val_loss: {val_loss:.5f} kld_loss: {kld_loss:.5f} "
                             f"au_loss: {au_loss:.5f} va_loss: {va_loss:.5f} em_loss: {em_loss:.5f}")

                checkpoint = {
                    'epoch': epoch + 1,
                    'best_loss': best_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                ckpt_path = self.get_ckpt_path(model, epoch=(epoch + 1))
                torch.save(checkpoint, ckpt_path)
                ckpt_path = self.get_ckpt_path(model, last=True)
                torch.save(checkpoint, ckpt_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logging.info(
                        f"New best loss ({best_val_loss:.5f}) at epoch {epoch + 1}, saving checkpoint."
                    )
                    ckpt_path = self.get_ckpt_path(model, best=True)
                    torch.save(checkpoint, ckpt_path)

    def train_autoencoder(self, model, data_loader, optimizer,
                          criterion, epoch, writer, device):
        whole_losses = AverageMeter()
        kld_losses = AverageMeter()
        au_losses = AverageMeter()
        va_losses = AverageMeter()
        em_losses = AverageMeter()

        model.train()
        for batch_idx, (input_emotion_clips, e_start_indices, e_end_indices,
                        output_emotion_clips, d_start_indices, d_end_indices) \
                in enumerate(tqdm(data_loader)):

            (input_emotion_clips, output_emotion_clips, e_start_indices,
             e_end_indices, d_start_indices, d_end_indices) = \
            (input_emotion_clips.to(device), output_emotion_clips.to(device), e_start_indices.to(device),
             e_end_indices.to(device), d_start_indices.to(device), d_end_indices.to(device))
            batch_size = input_emotion_clips.shape[0]

            out, _, dist, mask = (
                model(input_emotion_clips, e_start_indices, e_end_indices, d_start_indices, d_end_indices)
            )
            # out: ([B x L x 15], [B x L x 2], [B x L x 8])

            loss, kld_loss, au_loss, va_loss, em_loss = \
                criterion(predictions=out, targets=output_emotion_clips, distribution=dist, mask=mask)

            # Log metrics
            iteration = batch_idx + len(data_loader) * epoch
            if writer is not None:
                writer.add_scalar("Train/whole_loss", loss.data.item(), iteration)
                writer.add_scalar("Train/kld_loss", kld_loss.data.item(), iteration)
                writer.add_scalar("Train/au_loss", au_loss.data.item(), iteration)
                writer.add_scalar("Train/va_loss", va_loss.data.item(), iteration)
                writer.add_scalar("Train/em_loss", em_loss.data.item(), iteration)

            whole_losses.update(loss.data.item(), batch_size)
            kld_losses.update(kld_loss.data.item(), batch_size)
            au_losses.update(au_loss.data.item(), batch_size)
            va_losses.update(va_loss.data.item(), batch_size)
            em_losses.update(em_loss.data.item(), batch_size)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get learning rate
        lr = get_lr(optimizer=optimizer)
        if writer is not None:
            writer.add_scalar("Train/lr", lr, epoch)

        return whole_losses.avg, kld_losses.avg, au_losses.avg, va_losses.avg, em_losses.avg

    def val_autoencoder(self, model, val_loader, criterion, device):
        whole_losses = AverageMeter()
        kld_losses = AverageMeter()
        au_losses = AverageMeter()
        va_losses = AverageMeter()
        em_losses = AverageMeter()

        model.eval()
        for batch_idx, (input_emotion_clips, e_start_indices, e_end_indices,
                        output_emotion_clips, d_start_indices, d_end_indices) \
                in enumerate(tqdm(val_loader)):
            (input_emotion_clips, output_emotion_clips, e_start_indices,
             e_end_indices, d_start_indices, d_end_indices) = \
                (input_emotion_clips.to(device), output_emotion_clips.to(device), e_start_indices.to(device),
                 e_end_indices.to(device), d_start_indices.to(device), d_end_indices.to(device))
            batch_size = input_emotion_clips.shape[0]

            with torch.no_grad():
                out, _, dist, mask = (
                    model(input_emotion_clips, e_start_indices, e_end_indices, d_start_indices, d_end_indices)
                )
                loss, kld_loss, au_loss, va_loss, em_loss = \
                    criterion(predictions=out, targets=output_emotion_clips, distribution=dist, mask=mask)

            whole_losses.update(loss.data.item(), batch_size)
            kld_losses.update(kld_loss.data.item(), batch_size)
            au_losses.update(au_loss.data.item(), batch_size)
            va_losses.update(va_loss.data.item(), batch_size)
            em_losses.update(em_loss.data.item(), batch_size)

        return whole_losses.avg, kld_losses.avg, au_losses.avg, va_losses.avg, em_losses.avg