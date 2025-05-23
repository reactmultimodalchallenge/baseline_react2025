import math
import torch
import time
from einops import rearrange
from tqdm import tqdm
from utils.meters import AverageMeter
from utils.loss import DistributionLoss, AllThreMseLoss, MidLoss
# PCC; PearsonCC; AllMseLoss; ThreMseLoss; S_MSE
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_metrics


class Trainer(object):
    def __init__(self, model, loss_name='MSE', cal_logdets=True, no_inverse=False,
                 neighbor_pattern='nearest', num_frames=50, loss_mid=False, num_preds=10, batch_size=16,):

        self.model = model
        if loss_name == 'Distribution':
            self.criterion = DistributionLoss()
        else:
            print('Use Neighbor Pattern: ' + neighbor_pattern)
            if neighbor_pattern == 'nearest':
                self.threshold = None
                self.criterion = AllThreMseLoss()
                self.mid_criterion = MidLoss()
            elif neighbor_pattern == 'all':
                self.threshold = None
                self.criterion = AllThreMseLoss(cal_type='min')
                self.mid_criterion = MidLoss(loss_type='L2')

        self.cal_logdets = cal_logdets
        self.no_inverse = no_inverse
        self.neighbor_pattern = neighbor_pattern
        self.num_frames = num_frames
        self.loss_mid = loss_mid
        self.num_preds = num_preds
        self.data_parser = Processor(
            ckpt_dir="../pretrained_models/post_processor",
            num_preds=num_preds,
            cfg_dir="../",
        )
        self.bsz = batch_size

    def _parse_data(self,
                    speaker_video_inputs,
                    speaker_audio_inputs,
                    listener_param_gts,
                    indices,
                    seq_lens,
                    test=False):
        inputs = [torch.zeros(size=(seq_len, 25)) for seq_len in seq_lens]
        listener_param_gts = self.data_parser.forward(inputs, listener_param_gts)

        if test:
            # speaker_video_inputs -> List: [Tensor([N, n_frames, d_v]), Tensor([N', n_frames, d_v]), ...]'])]
            # listener_param_gts -> List: [Tensor([num_preds, l, d_e]), Tensor([num_preds, l', d_e]), ...]
            num_clip_list = [s.shape[0] for s in speaker_video_inputs]
            speaker_video_inputs = torch.cat(speaker_video_inputs, dim=0)  # [B, n_frames, d_v]
            speaker_audio_inputs = torch.cat(speaker_audio_inputs, dim=0)  # [B, n_frames, d_a]
            lengths = (seq_lens, num_clip_list)
        else:
            # speaker_video_inputs, speaker_audio_inputs -> List: [Tensor([n_frames, d]), ...]
            speaker_audio_inputs = torch.stack(speaker_audio_inputs, dim=0)  # [bsz, n_frames, d_v]
            speaker_video_inputs = torch.stack(speaker_video_inputs, dim=0)  # [bsz, n_frames, d_a]

            lengths = []
            listener_param_gts_list = []
            for idx, l_param_gt in zip(indices, listener_param_gts):
                listener_param_gts_list.append(l_param_gt[:, idx:idx + self.num_frames])
                lengths.append(l_param_gt.shape[0])
            listener_param_gts = torch.cat(listener_param_gts_list, dim=0).transpose(2, 1)
            lengths = torch.tensor(lengths)
            # listener_param_gts = [l_param_gt[:, idx:idx + self.num_frames] \
            #                       for idx, l_param_gt in zip(indices, listener_param_gts)]
            # lengths = torch.tensor([self.num_preds] * len(speaker_video_inputs))
            # [bsz * num_preds, n_frames, d_e]

        return speaker_video_inputs, speaker_audio_inputs, listener_param_gts, lengths

    def train(self, epoch, dataloader, optimizer, print_freq=1, train_iters=100):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        process_time = AverageMeter()
        losses_dtw = AverageMeter()
        losses_mid = AverageMeter()
        losses_det = AverageMeter()
        since = time.time()

        for i, (_, v_inputs, a_inputs, targets, indices, num_frames) in tqdm(enumerate(dataloader)):
            data_time.update(time.time() - since)

            v_inputs, a_inputs, targets, lengths = self._parse_data(
                v_inputs, a_inputs, targets, indices, num_frames, test=False,
            )
            # v_inputs: [bsz, n_frames, d_v]; a_inputs: [bsz, n_frames, d_a]; targets: [bsz * num_preds, n_frames, d_e]
            v_inputs, a_inputs, targets, lengths = v_inputs.cuda(), a_inputs.cuda(), targets.cuda(), lengths.cuda()
            process_time.update(time.time() - since)

            torch.autograd.set_detect_anomaly(True)
            if not self.no_inverse:
                speaker_features, listener_features, params, edge, nearest_targets, loss_det = \
                    self.model(v_inputs, a_inputs, targets, lengths)
            else:
                # Perform alignment after the inversion of samples
                speaker_features, listener_features, loss_det = self.model(v_inputs, a_inputs, targets, lengths)

            if self.neighbor_pattern == 'all':
                loss_dtw = self.criterion(speaker_features, listener_features, lengths, threshold=self.threshold)
                if self.loss_mid:
                    loss_mid = self.mid_criterion(listener_features, lengths)
                else:
                    loss_mid = 0.0
            else:
                loss_dtw = self.criterion(speaker_features, listener_features, lengths, threshold=self.threshold)

            loss = loss_dtw + (loss_det if self.cal_logdets else 0.) + loss_mid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_dtw.update(loss_dtw.item())
            losses_mid.update(loss_mid.item() if not type(loss_mid) == float else 0.)
            losses_det.update(loss_det.item() if not type(loss_det) == float else 0.)
            batch_time.update(time.time() - since)
            since = time.time()

            torch.set_printoptions(precision=4, sci_mode=False)
            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Whole Time {:.3f} ({:.3f})\t'
                      'Process Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_DTW {:.3f} ({:.3f})\t'
                      'Loss_MID {:.3f} ({:.3f})\t'
                      'Loss_DET {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              process_time.val, process_time.avg,
                              data_time.val, data_time.avg,
                              losses_dtw.val, losses_dtw.avg,
                              losses_mid.val, losses_mid.avg,
                              losses_det.val, losses_det.avg,
                              ))
                print('-' * 160)

            if i == train_iters:
                self.threshold = torch.tensor([(losses_dtw.avg / 3)]).cuda()
                print('='*10 + ' threshold ' + '='*10)
                print(self.threshold)
                break


    def test(self, testloader, modify=False):
        self.model.eval()
        test_iters = len(testloader)
        print(f"Length of testloader: {test_iters}")
        self.modify = modify

        speaker_params_all = []
        predictions_all = []
        targets_all = []
        for bsz_idx, (e_inputs, v_inputs, a_inputs, targets, indices, num_frames) in tqdm(enumerate(testloader)):
            # TODO debug: ==========
            # if bsz_idx > 5:
            #     break
            # TODO debug: ==========

            speaker_params_all.extend(e_inputs)
            v_inputs, a_inputs, targets, (seq_lens, num_clip_list) = self._parse_data(
                v_inputs, a_inputs, targets, indices, num_frames, test=True,
            )
            v_inputs, a_inputs = v_inputs.cuda(), a_inputs.cuda()

            SAMPLE_NUMS = 10
            pred_list = []
            with torch.no_grad():
                for k in range(SAMPLE_NUMS):
                    predictions = torch.zeros(size=(0, self.num_frames, 25))
                    for i in range(math.ceil(v_inputs.shape[0] / self.bsz)):
                        v_input_data = v_inputs[i*self.bsz:(i+1)*self.bsz]  # [bsz, n_frames, d_v]
                        a_input_data = a_inputs[i*self.bsz:(i+1)*self.bsz]  # [bsz, n_frames, d_a]

                        pred = self.model.inverse(v_input_data, a_input_data,
                                                  cal_norm=True, threshold=self.threshold).detach().cpu()
                        if self.modify:
                            pred = torch.where(pred[:, :15, :] >= 1.0, 1.0, 0.0)
                        predictions = torch.cat((predictions, pred), dim=0)  # [B, n_frames, d_e]

                    indices = [0] + list(np.cumsum(num_clip_list))
                    for j, (start_idx, end_idx) in enumerate(zip(indices[:-1], indices[1:])):
                        pred = predictions[start_idx:end_idx]
                        pred = rearrange(pred, 'b l d -> (b l) d')[:seq_lens[j]].unsqueeze(0)  # [1, L, d_e]

                        if k == 0:
                            pred_list.append(pred)
                        else:
                            pred_list[j] = torch.cat((pred_list[j], pred), dim=0)  # [num_preds, L, d_e]

            predictions_all.extend(pred_list)
            targets_all.extend(targets)
            # List: [Tensor([num_preds, L, d_e]), Tensor([num_preds, L', d_e]), ...]

        # TODO saving Tensor List
        try:
            torch.save({'GT': targets_all, 'PRED': predictions_all}, 'results.pt')
            print("Successfully saved Tensor List")
        except Exception:
            print("Failed to save Tensor List")

        results = compute_metrics(
            speaker_params_all,
            predictions_all,
            targets_all,
        )

        print(results)
