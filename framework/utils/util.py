import os
import numpy as np
import torch
from datetime import datetime
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.backends import cudnn
import hydra


def init_seed(seed, rank=0):
    process_seed = seed + rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    np.random.seed(process_seed)
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_config_from_file(path):
    return OmegaConf.load(path)


def load_config(args=None, config_path=None):
    if args is not None:
        config_from_args = OmegaConf.create(vars(args))
    else:
        config_from_args = OmegaConf.from_cli()
    # config_from_file = OmegaConf.load(cli_conf.pop('config') if config_path is None else config_path)
    config_from_file = load_config_from_file(config_path)
    return OmegaConf.merge(config_from_file, config_from_args)


def store_config(config):
    dir = config.trainer.out_dir
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img


def _fix_image(image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]
    return image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collect_grad_value_(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        grad_values.append(p.grad.data.abs().mean().item())
    grad_values = np.array(grad_values)
    return grad_values


def save_checkpoint(checkpoint_path, net, optimizer=None, epoch=None, best_loss=None):
    checkpoint = {
        'epoch': epoch if epoch is not None else None,
        'best_loss': best_loss if best_loss is not None else None,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(checkpoint, checkpoint_path)
    

def from_pretrained_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(model, torch.optim.Optimizer):
        model.load_state_dict(checkpoint['optimizer'])
        print(f'Successfully load optimizer checkpoint: {checkpoint_path}')
    else:
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f'Successfully load model checkpoint: {checkpoint_path}')
    return checkpoint.get('best_loss', float('inf')), checkpoint.get('epoch', 0)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def collect_grad_stats(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        # Store the absolute values of gradients
        grad_values.extend(p.grad.data.abs().view(-1).cpu().numpy())

    # Convert to a numpy array for statistical computation
    grad_values = np.array(grad_values)
    if grad_values.size == 0:
        return {"min": None, "max": None, "mean": None}

    # Compute min, max, and mean
    grad_stats = {
        "min": grad_values.min(),
        "max": grad_values.max(),
        "mean": grad_values.mean()
    }
    return grad_stats
