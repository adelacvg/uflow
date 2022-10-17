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
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data.distributed import DistributedSampler
import os
import commons
from model import Decoder, Decoder_test, GeneratorTrn, Morton_decode, Morton_encode
import utils
from torch.profiler import profile, record_function, ProfilerActivity
import json
import argparse
import itertools
import math
from morton_code import morton_encode,morton_decode

#   _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
#   _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
#   global_step = (epoch_str - 1) * len(train_loader)