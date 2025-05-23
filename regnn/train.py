from __future__ import print_function, absolute_import
import os
import sys
import torch
import random
import argparse
import numpy as np
import os.path as osp
from datasets import ActionDataloader  # from datasets import ActionData
from trainers import Trainer
from utils.logging import Logger
from torch.backends import cudnn
from utils.meters import AverageMeter
from utils.lr_scheduler import WarmupMultiStepLR
from models import CognitiveProcessor, PercepProcessor, MHP, LipschitzGraph


def set_seed(seed):
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def load_model(args):
    Cog = CognitiveProcessor(input_dim=64, convert_type=args.convert_type, num_features=args.num_frames,
                             n_channels=args.edge_dim, k=args.neighbors)
    Per = PercepProcessor(only_fuse=True)
    Mot = LipschitzGraph(edge_channel=args.edge_dim, n_layers=args.layers, act_type=args.act,
                         num_features=args.num_frames, norm=args.norm, get_logdets=args.get_logdets)
    model = MHP(p=Per, c=Cog, m=Mot, no_inverse=args.no_inverse, neighbor_pattern=args.neighbor_pattern)
    model = model.cuda()

    return model


def train(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    set_seed(args.seed)

    trainloader = ActionDataloader(
        root=args.data_dir,
        num_frames=args.num_frames,
        neighbor_pattern=args.neighbor_pattern,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        k_select=args.k_select,
    ).get_dataloader(data_type='train')
    train_iters = len(trainloader)

    model = load_model(args)
    trainer = Trainer(model=model, loss_name=args.loss_name, no_inverse=args.no_inverse,
                      neighbor_pattern=args.neighbor_pattern, num_frames=args.num_frames,
                      loss_mid=args.loss_mid, cal_logdets=args.get_logdets,
                      num_preds=args.num_preds, batch_size=args.batch_size)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, gamma=args.gamma, warmup_factor=args.warmup_factor,
                                     milestones=args.milestones, warmup_iters=args.warmup_step)

    resume_training = args.resume_training
    if resume_training:
        checkpoint = torch.load(osp.join(args.logs_dir, f"mhp-last-seed{args.seed}.pth"), map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.to(torch.device('cuda'))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get("epoch", 0)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step(epoch)
        print('Epoch [{}] LR [{:.6f}]'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        trainer.train(epoch=epoch, dataloader=trainloader, optimizer=optimizer, train_iters=train_iters)
        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, osp.join(args.logs_dir, f"mhp-epoch{(epoch+1)}-seed{args.seed}.pth"))
            torch.save(checkpoint, osp.join(args.logs_dir, f"mhp-last-seed{args.seed}.pth"))
            print(f"Saving the checkpoint of epoch {epoch+1} at {args.logs_dir}")

def test():
    sys.stdout = Logger(osp.join(args.logs_dir, 'test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    set_seed(args.seed)

    testloader = ActionDataloader(
        root=args.data_dir,
        num_frames=args.num_frames,
        neighbor_pattern=args.neighbor_pattern,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
    ).get_dataloader(data_type='test')

    model_pth = args.model_pth
    model = load_model(args)
    ckpt = torch.load(model_pth)
    model.load_state_dict(ckpt['state_dict'])
    model.to(torch.device('cuda'))
    print(f"Checkpoint loaded from {model_pth}")

    trainer = Trainer(model=model,
                      neighbor_pattern=args.neighbor_pattern,
                      no_inverse=args.no_inverse)
    trainer.threshold = 0.06
    trainer.test(testloader, modify=args.modify)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Actiton Generation")
    # pattern
    parser.add_argument('--test', action='store_true')
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    # model
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--act', type=str, default='ELU')
    parser.add_argument('--no-inverse', action='store_true')
    parser.add_argument('--convert-type', type=str, default='indirect')
    parser.add_argument('--edge-dim', type=int, default=8)
    parser.add_argument('--neighbors', type=int, default=6)
    # optimizer
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 15])
    parser.add_argument('--warmup-factor', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100, help="training epochs")
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--loss-name', type=str, default='MSE')
    parser.add_argument('--train-iters', type=int, default=100)
    parser.add_argument('--get-logdets', action='store_true')
    parser.add_argument('--loss-mid', action='store_true')
    parser.add_argument('--neighbor-pattern', type=str, default='nearest',
                        choices=['nearest', 'pair', 'all'], help="neighbor pattern")
    parser.add_argument('--num-frames', type=int, default=50)
    # parser.add_argument('--stride', type=int, default=25)
    parser.add_argument('--num-preds', type=int, default=10)
    parser.add_argument('--resume-training', action='store_true',
                        help="resume training from the saved checkpoint if provided")
    parser.add_argument('--k-select', type=int, default=2)
    # testing configs
    parser.add_argument('--modify', action='store_true')
    parser.add_argument('--model-pth', type=str, metavar='PATH', default=' ')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    """
    Training:
        python train.py --logs-dir='Gmm-logs' --milestones=9 --batch-size=64 --layers=2 --norm \
        --neighbor-pattern='all' --convert-type='direct' --loss-mid --data-dir=/home/x/xk18/REACT2025
    
    Testing:
        python train.py --test --layers=2 --norm --model-pth="Gmm-logs/mhp-epoch100-seed1.pth" --neighbor-pattern='all' \
        --convert-type='direct' --seed=1 --data-dir=/home/x/xk18/REACT2025
    """

    args = parser.parse_args()
    if args.test:
        test()
    else:
        train(args)