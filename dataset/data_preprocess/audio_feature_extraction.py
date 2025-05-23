import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from dataset.modules.audio_processor import AudioProcessor


def main(args):
    video_paths = []
    input_paths = []
    output_paths = []

    for root, _, files in os.walk(args.root_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue

            input_dir = os.path.join(root, file)
            input_paths.append(input_dir)

            output_dir = root.replace("audio", "audio-features")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file.replace(".wav", ".npy"))
            output_paths.append(output_path)

            video_dir = root.replace("audio", "video-raw")
            video_path = os.path.join(video_dir, file.replace(".wav", ".mp4"))
            video_paths.append(video_path)
    print(f"Read {len(input_paths)} audio files, {len(output_paths)} output files, {len(video_paths)} video files")

    sample_rate = args.sample_rate
    fps = args.fps
    wav2vec_model_path = args.wav2vec_model_path
    wav2vec_only_last_features = args.wav2vec_only_last_features == "last"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with AudioProcessor(
            sample_rate,
            fps,
            wav2vec_model_path,
            wav2vec_only_last_features,
            device=device,
    ) as audio_processor:

        for input_path, output_path, video_path in tqdm(zip(input_paths, output_paths, video_paths)):
            cap = cv2.VideoCapture(video_path)
            seq_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            audio_emb, audio_length = audio_processor.preprocess(wav_file=input_path, seq_len=seq_len)

            try:
                np.save(output_path, audio_emb)
                print(f"Successfully saved audio embedding {input_path} -> {output_path}")
            except Exception as e:
                print(f"Error saving {input_path} -> {output_path}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Feature Extraction")
    parser.add_argument('--root_dir', type=str,
                        default='./data',
                        help="root directory of REACT2025 dataset")
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help="original sampling rate of audio data")
    parser.add_argument('--fps', type=int, default=30, help="original fps of video data")
    parser.add_argument('--wav2vec_model_path', type=str,
                        default='./pretrained_models/wav2vec/wav2vec2-base-960h',
                        help="wav2vec model path")
    parser.add_argument('--wav2vec_only_last_features', type=str, default='last',
                        help="wav2vec only last features")

    args = parser.parse_args()

    main(args)