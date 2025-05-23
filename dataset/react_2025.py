import os
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
from decord import VideoReader
from decord import cpu
from torch.utils.data import DataLoader
from dataset.tools.util import Transform, extract_audio_features
import torchaudio

torchaudio.set_audio_backend("sox_io")


def custom_collate(batch):
    speaker_audio_clip = [item[0] for item in batch if len(item[0]) > 0]
    speaker_video_clip = [item[1] for item in batch if len(item[1]) > 0]
    speaker_emotion_clip = [item[2] for item in batch if len(item[2]) > 0]
    speaker_params_clip = [item[3] for item in batch if len(item[3]) > 0]
    listener_video_clip = [item[4] for item in batch if len(item[4]) > 0]
    listener_emotion_clip = [item[5] for item in batch if len(item[5]) > 0]
    listener_params_clip = [item[6] for item in batch if len(item[6]) > 0]
    speaker_clip_length = [item[7] for item in batch]
    listener_clip_length = [item[8] for item in batch]

    return (
        speaker_audio_clip,
        speaker_video_clip,
        speaker_emotion_clip,
        speaker_params_clip,
        listener_video_clip,
        listener_emotion_clip,
        listener_params_clip,
        speaker_clip_length,
        listener_clip_length,
    )


class ReactionAutoEncoderDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 clip_length: int = 1000,
                 **kwargs):

        self._root_dir = root_dir
        self._split = split
        self._clip_length = clip_length

        dataset_dir = os.path.join(root_dir, self._split)
        self._emotion_dir = os.path.join(dataset_dir, 'facial-attributes')

        data_list = []
        for root, _, files in os.walk(self._emotion_dir):
            for path in files:
                path = Path(path)

                if path.suffix.lower() != '.npy':
                    continue

                file_path = os.path.join(root, path)
                data_list.append(file_path)
        self._data_list = data_list

    def __getitem__(self, index):
        data_path = self._data_list[index]
        emotion = np.load(data_path)
        speaker_emotion_clip = torch.from_numpy(emotion)
        total_length = len(emotion)

        global_cp = random.randint(0, total_length - self._clip_length) \
            if total_length > self._clip_length else 0
        speaker_emotion_clip = speaker_emotion_clip[global_cp: global_cp + self._clip_length]  # (25-d)

        # padding_length = self._clip_length - total_length if total_length < self._clip_length else 0
        clip_length = self._clip_length if total_length >= self._clip_length else total_length

        input_length = random.randint(1, clip_length)
        input_start_idx = random.randint(0, clip_length - input_length)
        input_end_idx = input_start_idx + input_length - 1
        input_emotion_clip = speaker_emotion_clip[input_start_idx: input_start_idx + input_length]
        input_emotion_clip = torch.cat(
            (input_emotion_clip, torch.zeros(size=(self._clip_length - input_length, 25))), dim=0)

        output_length = random.randint(1, clip_length)
        output_start_idx = random.randint(0, clip_length - output_length)
        output_end_idx = output_start_idx + output_length - 1
        output_emotion_clip = speaker_emotion_clip[output_start_idx: output_start_idx + output_length]
        output_emotion_clip = torch.cat(
            (output_emotion_clip, torch.zeros(size=(self._clip_length - output_length, 25))), dim=0)

        return (input_emotion_clip, input_start_idx, input_end_idx,
                output_emotion_clip, output_start_idx, output_end_idx)

    def __len__(self):
        return len(self._data_list)


class ReactionDataset(data.Dataset):
    def __init__(self,
                 root_dir: str = './data',  # /path/to/dataset
                 split: str = 'train',
                 clip_length: int = None,
                 target_size: int = 224,
                 crop_size: int = 224,
                 fps: int = 30,
                 audio_feature_type: str = 'wav2vec',
                 load_video_s: bool = False,
                 load_video_l: bool = False,
                 load_audio: bool = True,
                 load_emotion_s: bool = True,
                 load_emotion_l: bool = True,
                 load_3dmm_s: bool = True,
                 load_3dmm_l: bool = True,
                 normalize_3dmm: str = 'standard',  # standard | zero_center
                 **kwargs,
                 ):

        self._root_dir = root_dir
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_audio = load_audio
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l

        dataset_dir = os.path.join(root_dir, self._split)
        self.audio_feature_type = audio_feature_type
        if self.audio_feature_type == 'wav2vec':
            self._audio_dir = os.path.join(dataset_dir, 'audio-features')
        elif self.audio_feature_type == 'mfcc':
            self._audio_dir = os.path.join(dataset_dir, 'audio')
        self._video_dir = os.path.join(dataset_dir, 'video-face-crop')
        self._emotion_dir = os.path.join(dataset_dir, 'facial-attributes')
        self._3dmm_dir = os.path.join(dataset_dir, 'coefficients')

        gt_path_dict = {}
        for root, _, files in os.walk(self._video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix

                if ext.lower() != '.mp4':
                    continue

                session_id = Path(*Path(root).parts[-2:]) # listener/session0

                file_path = session_id / file
                if session_id not in gt_path_dict:
                    gt_path_dict[session_id] = [file_path]
                else:
                    gt_path_dict[session_id].append(file_path)

        speaker_path_list = []
        listener_path_list = []
        gt_path_list = []

        for root, _, files in os.walk(self._video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                # file, ext = os.path.splitext(path)
                # e.g., 'Camera-2024-06-21-103121-103102', '.mp4'
                if ext.lower() != '.mp4':
                    continue

                #  role: listener | speaker
                #  session id: session*
                #  file: Camera-2024-06-21-103121-103102
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

        mean_face_path = os.path.join(hydra.utils.get_original_cwd(), 'external/FaceVerse/mean_face.npy')
        self.mean_face = torch.FloatTensor(
            np.load(mean_face_path).astype(np.float32)).view(1, 1, -1)
        std_face_path = os.path.join(hydra.utils.get_original_cwd(), 'external/FaceVerse/std_face.npy')
        self.std_face = torch.FloatTensor(
            np.load(std_face_path).astype(np.float32)).view(1, 1, -1)

        self.normalize_3dmm = normalize_3dmm
        if normalize_3dmm == 'standard':
            self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face) / self.std_face)
        elif normalize_3dmm == 'zero_center':
            self._transform_3dmm = transforms.Lambda(lambda e: e - self.mean_face)
        else:
            raise ValueError(f"Unknown normalize_3dmm: {normalize_3dmm}")

        self._transform = Transform(target_size, crop_size)
        self._len = len(self.speaker_path_list)

    def __getitem__(self, index):
        speaker_path = self.speaker_path_list[index]  # e.g., speaker/session*/Camera-2024-06-21-103121-103102
        listener_path = self.listener_path_list[index]  # e.g., listener/session*/Camera-2024-06-21-103121-103102

        video_path = os.fspath(Path(self._video_dir) / speaker_path.with_suffix('.mp4'))
        vr = VideoReader(video_path, ctx=cpu(0))
        total_length = len(vr)  # length from 58 to more than 20000
        if not self.load_video_s:
            del vr

        if self._split == "test":
            k_select = 9
            gt_paths = self.gt_path_list[index]
            listener_paths = []
            listener_paths.append(listener_path)
            c = gt_paths if len(gt_paths) <= 1 else [p for p in gt_paths if p != listener_path]
            listener_paths.extend(random.sample(c, k_select) \
                                      if len(c) >= k_select else random.choices(c, k=k_select))
            self._clip_length = total_length

        cp = random.randint(0, total_length - self._clip_length) \
            if total_length > self._clip_length else 0

        # ========== Load Speaker Data ==========
        # speaker's face video clip
        speaker_video_clip = torch.zeros(size=(0,))
        if self.load_video_s:
            clip = []
            for i in range(cp, cp + self._clip_length):
                if i >= len(vr):
                    break
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            del vr
            speaker_video_clip = torch.stack(clip, dim=0)
            if total_length < self._clip_length:
                speaker_video_clip = torch.cat((speaker_video_clip,
                                                torch.zeros(size=(self._clip_length - total_length,
                                                                  *speaker_video_clip.shape[1:]))), dim=0)

        # speaker's facial attribute (emotion) clip
        speaker_emotion_clip = torch.zeros(size=(0,))
        if self.load_emotion_s:
            emotion_path = os.fspath(Path(self._emotion_dir) / speaker_path.with_suffix('.npy'))
            emotion = np.load(emotion_path)
            speaker_emotion_clip = torch.from_numpy(emotion)
            speaker_emotion_clip = speaker_emotion_clip[cp: cp + self._clip_length]  # (25-d)

        # speaker's 3DMM coefficient (facial motion) clip
        speaker_params_clip = torch.zeros(size=(0,))
        if self.load_3dmm_s:
            params_path = os.fspath(Path(self._3dmm_dir) / speaker_path.with_suffix('.npy'))
            params = torch.FloatTensor(np.load(params_path)).squeeze()
            params = params[cp: cp + self._clip_length]
            speaker_params_clip = self._transform_3dmm(params)[0]  # (58-d)

        # speaker's audio feature clip
        speaker_audio_clip = torch.zeros(size=(0,))
        if self.load_audio:
            if self.audio_feature_type == 'wav2vec':
                audio_path = os.fspath(Path(self._audio_dir) / speaker_path.with_suffix('.npy'))
                speaker_audio_clip = np.load(audio_path)  # (768-d)
            elif self.audio_feature_type == 'mfcc':
                audio_path = os.fspath(Path(self._audio_dir) / speaker_path.with_suffix('.wav'))
                speaker_audio_clip = extract_audio_features(audio_path, self._fps, total_length)  # (78-d)
            else:
                raise ValueError(f"Unknown audio feature type: {self.audio_feature_type}")
            speaker_audio_clip = torch.from_numpy(speaker_audio_clip)[cp:cp + self._clip_length]

        if total_length < self._clip_length:
            speaker_audio_clip = torch.cat((speaker_audio_clip,
                                            torch.zeros(size=(self._clip_length - total_length,
                                                              speaker_audio_clip.shape[-1]))), dim=0) \
                if self.load_audio else speaker_audio_clip
            speaker_emotion_clip = torch.cat((speaker_emotion_clip,
                                            torch.zeros(size=(self._clip_length - total_length,
                                                              speaker_emotion_clip.shape[-1]))), dim=0) \
                if self.load_emotion_s else speaker_emotion_clip
            speaker_params_clip = torch.cat((speaker_params_clip,
                                            torch.zeros(size=(self._clip_length - total_length,
                                                              speaker_params_clip.shape[-1]))), dim=0) \
                if self.load_3dmm_s else speaker_params_clip

        # ========== Load Listener Data ==========
        if self._split == "test":
            # listener's (ground-truth) face video clip
            listener_clip_length = []
            for k, listener_path in enumerate(listener_paths):
                if k != 0:
                    continue

                video_path = os.fspath(Path(self._video_dir) / listener_path.with_suffix('.mp4'))
                vr = VideoReader(video_path, ctx=cpu(0))
                clip = []
                for f in range(len(vr)):
                    frame = vr[f]
                    img = Image.fromarray(frame.asnumpy())
                    img = self._transform(img)
                    clip.append(img)
                del vr
                listener_video_clip = [torch.stack(clip, dim=0)]
            listener_video_clip = (listener_video_clip +
                                   [listener_video_clip[:1]] * (len(listener_paths) - 1))  # TODO to be modified

            listener_emotion_clip = torch.zeros(size=(0,))
            # listener's emotion ground-truths
            if self.load_emotion_l:
                listener_emotion_clips = []
                for listener_path in listener_paths:
                    emotion_path = os.fspath(Path(self._emotion_dir) / listener_path.with_suffix('.npy'))
                    emotion = np.load(emotion_path)
                    listener_clip_length.append(emotion.shape[0])
                    listener_emotion_clip = torch.from_numpy(emotion)
                    listener_emotion_clips.append(listener_emotion_clip)
                listener_emotion_clip = listener_emotion_clips

            listener_params_clip = torch.zeros(size=(0,))
            # listener's 3DMM coefficients ground-truths
            if self.load_3dmm_l:
                listener_params_clips = []
                for listener_path in listener_paths:
                    params_path = os.fspath(Path(self._3dmm_dir) / listener_path.with_suffix('.npy'))
                    params = torch.FloatTensor(np.load(params_path)).squeeze()
                    listener_params_clip = self._transform_3dmm(params)[0]
                    listener_params_clips.append(listener_params_clip)
                listener_params_clip = listener_params_clips

            speaker_clip_length = total_length
            listener_clip_length = torch.tensor(listener_clip_length)
        else:
            # listener's (ground-truth) face video clip
            listener_video_clip = torch.zeros(size=(0,))
            if self.load_video_l:
                video_path = os.fspath(Path(self._video_dir) / listener_path.with_suffix('.mp4'))
                vr = VideoReader(video_path, ctx=cpu(0))

                clip = []
                for i in range(cp, cp + self._clip_length):
                    if i >= len(vr):
                        break
                    frame = vr[i]
                    img = Image.fromarray(frame.asnumpy())
                    img = self._transform(img)
                    clip.append(img)
                del vr
                listener_video_clip = torch.stack(clip, dim=0)

                _clip_length = len(listener_video_clip)
                if self.load_video_s:
                    listener_video_clip = torch.cat(
                        (listener_video_clip, speaker_video_clip[(_clip_length - self._clip_length):]), dim=0) \
                        if _clip_length < self._clip_length else listener_video_clip[:self._clip_length]
                else:
                    listener_video_clip = torch.cat(
                        (listener_video_clip,
                         torch.zeros(size=(self._clip_length - _clip_length, *listener_video_clip.shape[1:]))), dim=0) \
                        if _clip_length < self._clip_length else listener_video_clip[:self._clip_length]

            # listener's (ground-truth) facial attribute (emotion) clip
            listener_emotion_clip = torch.zeros(size=(0,))
            if self.load_emotion_l:
                emotion_path = os.fspath(Path(self._emotion_dir) / listener_path.with_suffix('.npy'))
                emotion = np.load(emotion_path)
                assert self.load_emotion_s, "Loading speaker's emotion is required for listener's emotion at the moment"
                listener_emotion_clip = torch.from_numpy(emotion)[cp: cp + self._clip_length]
                _clip_length = len(listener_emotion_clip)
                listener_emotion_clip = torch.cat(
                    (listener_emotion_clip, speaker_emotion_clip[(_clip_length - self._clip_length):]), dim=0) \
                    if _clip_length < self._clip_length else listener_emotion_clip[:self._clip_length]

            # speaker's (ground-truth) 3DMM coefficient (facial motion) clip
            listener_params_clip = torch.zeros(size=(0,))
            if self.load_3dmm_l:
                params_path = os.fspath(Path(self._3dmm_dir) / listener_path.with_suffix('.npy'))
                params = torch.FloatTensor(np.load(params_path)).squeeze()
                assert self.load_3dmm_s, "Loading speaker's 3dmm is required for listener's 3dmm at the moment"
                params = params[cp: cp + self._clip_length]
                listener_params_clip = self._transform_3dmm(params)[0]
                _clip_length = len(listener_params_clip)
                listener_params_clip = torch.cat(
                    (listener_params_clip, speaker_params_clip[(_clip_length - self._clip_length):]), dim=0) \
                    if _clip_length < self._clip_length else listener_params_clip[:self._clip_length]

            speaker_clip_length = torch.tensor(total_length)
            listener_clip_length = torch.tensor(total_length)

        return (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_params_clip,
            listener_video_clip,
            listener_emotion_clip,
            listener_params_clip,
            speaker_clip_length,
            listener_clip_length,
        )

    def __len__(self):
        return self._len


class ReactionDataloader:
    def __init__(self,
                 train_dataset: DictConfig = None,
                 validation_dataset: DictConfig = None,
                 test_dataset: DictConfig = None,
                 **kwargs):

        self.seed = kwargs.pop('seed')
        self.train_set_cfg = train_dataset
        self.val_set_cfg = validation_dataset
        self.test_set_cfg = test_dataset
        self.clip_length = train_dataset.clip_length

        self.collate_fn_dict = {'none': None,
                                'custom': custom_collate,}

    def get_dataloader(self, stage, collate_fn: str = 'custom'):
        def worker_init_fn(worker_id):
            seed = self.seed + worker_id
            random.seed(seed)
            np.random.seed(seed)
            # torch.manual_seed(seed)

        if stage == 'fit':
            train_dataset = instantiate(self.train_set_cfg)
            train_loader = DataLoader(dataset=train_dataset,
                                      collate_fn=self.collate_fn_dict[collate_fn],
                                      batch_size=self.train_set_cfg.batch_size,
                                      shuffle=self.train_set_cfg.shuffle,
                                      num_workers=self.train_set_cfg.num_workers,
                                      worker_init_fn=worker_init_fn)

            val_dataset = instantiate(self.val_set_cfg)
            val_loader = DataLoader(dataset=val_dataset,
                                    collate_fn=self.collate_fn_dict[collate_fn],
                                    batch_size=self.val_set_cfg.batch_size,
                                    shuffle=self.val_set_cfg.shuffle,
                                    num_workers=self.val_set_cfg.num_workers,
                                    worker_init_fn=worker_init_fn)
            return train_loader, val_loader

        else:
            test_dataset = instantiate(self.test_set_cfg)
            test_loader = DataLoader(dataset=test_dataset,
                                     collate_fn=self.collate_fn_dict[collate_fn],
                                     batch_size=self.test_set_cfg.batch_size,
                                     shuffle=self.test_set_cfg.shuffle,
                                     num_workers=self.test_set_cfg.num_workers,
                                     worker_init_fn=worker_init_fn)
            return test_loader