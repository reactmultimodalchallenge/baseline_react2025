import subprocess
import soundfile as sf
import torch
import numpy as np
import torchaudio
from torchvision import transforms


class Transform(object):
    def __init__(self, target_size=224, crop_size=224):
        self.target_size = target_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int):
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    assert ret == 0, "Resample audio failed!"
    return output_audio_file


def extract_audio_features(audio_path, fps, n_frames):
    audio, sr = sf.read(audio_path)

    if audio.ndim == 2:
        audio = audio.mean(-1)

    frame_n_samples = int(sr / fps)
    curr_length = len(audio)

    target_length = frame_n_samples * n_frames

    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])

    shifted_n_samples = 0
    curr_feats = []

    for i in range(n_frames):
        curr_samples = audio[i * frame_n_samples:shifted_n_samples + i * frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1),
                                                     sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1)  # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)

    return curr_feats