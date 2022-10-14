import sys
import torch
from torch import nn, optim
import torchvision
from IPython import embed
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import ToTensor, Normalize
from torchvision.utils import save_image
from torch.utils.data.distributed import DistributedSampler
import os
import commons
import utils
import json
import argparse
import itertools
import math

torch.backends.cudnn.benchmark = True
global_step = 0

def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8928'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
                           
def run(rank, n_gpus,hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # print(len(train_dataset))
    train_sampler = DistributedSampler(train_dataset)
    data_loader = DataLoader(train_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=4,
                    sampler = train_sampler)
    if rank == 0:
        eval_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
        eval_loader = DataLoader(eval_dataset,
                        batch_size=8,
                        shuffle=False,
                        num_workers=4)
    

    data_loader = [next(iter(data_loader))]
    # for batch_idx, (x, y) in enumerate(data_loader):
    #     print(x.shape)
    #     save_image(x[0],'1.png')


if __name__ == "__main__":
  main()
