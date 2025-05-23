import os
import random
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def custom_collate(batch):
    speaker_param_inputs = [item[0] for item in batch]
    speaker_video_inputs = [item[1] for item in batch]
    speaker_audio_inputs = [item[2] for item in batch]
    listener_param_gts = [item[3] for item in batch]
    cp = [item[4] for item in batch]
    num_frames = [item[5] for item in batch]

    return speaker_param_inputs, speaker_video_inputs, speaker_audio_inputs, listener_param_gts, cp, num_frames


class ActionData(Dataset):
    def __init__(self, root, data_type, num_frames=50, neighbor_pattern='all', k_select=None):
        self.root_dir = Path(root)
        self.split = data_type
        self.data_dir = self.root_dir / self.split
        self.video_dir = self.data_dir / 'video-features'
        self.audio_dir = self.data_dir / 'audio-features'
        self.param_dir = self.data_dir / 'facial-attributes'
        self.num_frames = num_frames  # 50
        self.neighbor_pattern = neighbor_pattern  # 'all'
        self.k_select = 10 if k_select is None else k_select

        gt_path_dict = {}
        for root, _, files in os.walk(self.video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                if ext.lower() != '.pth':
                    continue

                session_id = Path(*Path(root).parts[-2:])  # listener/session0
                file_path = session_id / file
                if session_id not in gt_path_dict:
                    gt_path_dict[session_id] = [file_path]
                else:
                    gt_path_dict[session_id].append(file_path)

        speaker_path_list = []
        listener_path_list = []
        gt_path_list = []

        for root, _, files in os.walk(self.video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                if ext.lower() != '.pth':
                    continue

                parts = Path(root).parts
                file_path = Path(*parts[-2:]) / file
                role = parts[-2]
                session_id = Path(parts[-1])
                gt_session_id = 'speaker' / session_id if role == 'listener' else 'listener' / session_id
                listener_file_path = gt_session_id / file

                speaker_path_list.append(file_path)
                listener_path_list.append(listener_file_path)
                listener_gt_paths = gt_path_dict[gt_session_id]
                gt_path_list.append(listener_gt_paths)

        self.speaker_path_list = speaker_path_list.copy()
        self.listener_path_list = listener_path_list.copy()
        self.gt_path_list = gt_path_list.copy()
        self._len = len(self.speaker_path_list)

    def segment_torch2d(self, t: torch.Tensor, seg_len: int) -> torch.Tensor:
        L, dim = t.size()
        pad = (-L) % seg_len
        padded = F.pad(t, (0, 0, 0, pad))
        return padded.view(-1, seg_len, dim)

    def __getitem__(self, index):
        speaker_path = self.speaker_path_list[index]
        # e.g., speaker/session*/Camera-2024-06-21-103121-103102

        # speaker's video features
        s_video_path = Path(self.video_dir) / speaker_path.with_suffix('.pth')
        s_video_inputs = torch.load(s_video_path)

        # speaker's audio features
        s_audio_path = Path(self.audio_dir) / speaker_path.with_suffix('.npy')
        s_audio_inputs = torch.from_numpy(np.load(s_audio_path))

        # speaker's emotion features
        s_param_path = Path(self.param_dir) / speaker_path.with_suffix('.npy')
        s_param_inputs = torch.from_numpy(np.load(s_param_path))

        listener_path = self.listener_path_list[index]
        # e.g., listener/session*/Camera-2024-06-21-103121-103102
        listener_paths = self.gt_path_list[index]
        if len(listener_paths) >= self.k_select:
            listener_paths = ([listener_path] +
                              random.sample([p for p in listener_paths if p != listener_path], self.k_select - 1)) \
                if len(listener_paths) >= self.k_select else random.choices(listener_paths, k=self.k_select)

        total_length = s_video_inputs.size(0)
        num_frames = total_length if self.split == "test" else self.num_frames

        cp = random.randint(0, total_length - num_frames)
        # get speaker's audio and video data inputs
        s_video_inputs = s_video_inputs[cp: cp + num_frames]
        s_audio_inputs = s_audio_inputs[cp: cp + num_frames]
        # 'train' [n_frames, d]

        if self.split == "test":
            s_video_inputs = self.segment_torch2d(s_video_inputs, self.num_frames)
            s_audio_inputs = self.segment_torch2d(s_audio_inputs, self.num_frames)
            # 'test' [N, n_frames, d]

        # listener's emotion features
        l_param_gts = [torch.from_numpy(np.load(Path(self.param_dir) / l_path.with_suffix('.npy')))
                       for l_path in listener_paths]

        return s_param_inputs, s_video_inputs, s_audio_inputs, l_param_gts, cp, total_length

    def __len__(self):
        return self._len


class ActionDataloader:
    def __init__(self,
                 root: str = '../data',
                 num_frames: int = 50,
                 neighbor_pattern: str = 'all',
                 batch_size: int = 16,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 k_select: int = 10,
                 **kwargs):
        self.data_dir = root
        self.num_frames = num_frames
        self.neighbor_pattern = neighbor_pattern
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.k_select = k_select
        self.collate_fn_dict = {'none': None,
                                'custom': custom_collate}

    def get_dataloader(self, data_type: str = 'train',
                       collate_fn: str = 'custom', **kwargs):
        dataset = ActionData(
            root=self.data_dir,
            data_type=data_type,
            num_frames=self.num_frames,
            neighbor_pattern=self.neighbor_pattern,
            k_select=self.k_select,
        )

        dataloader = DataLoader(dataset=dataset,
                                collate_fn=self.collate_fn_dict[collate_fn],
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,)
        return dataloader
