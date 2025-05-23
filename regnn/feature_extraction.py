import math
import torch
from models import SwinTransformer, VGGish
import pandas as pd
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
from PIL import Image
from decord import VideoReader
from decord import cpu
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--data-dir', default="../data", type=str,
                        help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset",
                        choices=["train", "val", "test"], default='train')
    parser.add_argument('--type', type=str, help="type of features to extract",
                        default='video')
    parser.add_argument('--ckpt-path', type=str,
                        default="../pretrained_models/swin_transformer/swin_fer.pth",
                        help="path of saved swin-transformer ckpt")
    args = parser.parse_args()
    return args


class Transform(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img


def extract_video_features(args):
    max_seq_len = 1000

    model = SwinTransformer(
        embed_dim=96, depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        window_size=7, drop_path_rate=0.2,
        num_classes=7,
    )
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model = model.cuda().eval()
    transform = Transform()

    base_dir = os.path.join(args.data_dir, args.split)
    in_root  = os.path.join(base_dir, 'video-face-crop')
    out_root = os.path.join(base_dir, 'video-features')

    in_paths = []
    for role in sorted(os.listdir(in_root)):
        role_dir = os.path.join(in_root, role)
        if not os.path.isdir(role_dir): continue
        for sess in sorted(os.listdir(role_dir)):
            sess_dir = os.path.join(role_dir, sess)
            if not os.path.isdir(sess_dir): continue
            for fname in sorted(os.listdir(sess_dir)):
                if fname.lower().endswith('.mp4'):
                    in_paths.append(os.path.join(sess_dir, fname))

    out_paths = []
    for vid in in_paths:
        rel   = os.path.relpath(vid, in_root)          # e.g. speaker/session0/xxx.mp4
        rel_p = rel.replace('.mp4', '.pth')
        op    = os.path.join(out_root, rel_p)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        out_paths.append(op)


    for vid_path, out_path in zip(in_paths, out_paths):
        vr = VideoReader(vid_path, ctx=cpu(0))
        frames = [ transform(Image.fromarray(f.asnumpy())).unsqueeze(0)
                   for f in vr ]
        if not frames:
            continue
        clip = torch.cat(frames, dim=0).cuda()
        with torch.no_grad():
            if clip.shape[0] <= max_seq_len:
                feats = model.forward_features(clip).detach().cpu()
            else:
                feats = torch.zeros(size=(0, 768))
                for i in range(math.ceil(clip.shape[0]/max_seq_len)):
                    input = clip[i*max_seq_len:(i+1)*max_seq_len]
                    feats = torch.cat([feats, model.forward_features(input).detach().cpu()], dim=0)
        torch.save(feats, out_path)
        assert feats.shape[0] == len(vr), (
            f"Frame count mismatch for {vid_path}: "
            f"{feats.shape[0]} vs {len(vr)}"
        )
        print(f"Saved {feats.shape} → {out_path}")


def main(args):
    # if args.type == 'video':
    args.split = 'train'
    extract_video_features(args)
    # args.split = 'test'
    # extract_video_features(args)
    args.split = 'val'
    extract_video_features(args)


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '32'
    main(args)