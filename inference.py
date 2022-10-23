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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
torch.backends.cudnn.benchmark = True

hps = utils.get_hparams()
input_size = hps.train.input_size
hps.net_g.input_length = (input_size//(int(math.sqrt(hps.net_g.scale_factor))**hps.net_g.n_uplayers))**2
hps.net_d.input_length = (input_size//(int(math.sqrt(hps.net_d.scale_factor))**hps.net_d.n_uplayers))**2

net_g = GeneratorTrn(**hps.net_g).to(device)
optim_g = torch.optim.AdamW(
net_g.parameters(), 
hps.train.learning_rate, 
betas=hps.train.betas, 
eps=hps.train.eps)

net_d = Decoder(**hps.net_d).to(device)
optim_d = torch.optim.AdamW(
net_d.parameters(), 
hps.train.learning_rate, 
betas=hps.train.betas, 
eps=hps.train.eps)

_, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g)
_, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d)

num_samples = 8
net_g.eval()
net_d.eval()
with torch.no_grad():
  samples = net_g.sample(num_samples,device)

# image_dict = {
#     "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
# }
# audio_dict = {
#     "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
# }
# utils.summarize(
#     writer=writer_eval,
#     global_step=global_step, 
#     images=image_dict,
#     audios=audio_dict,
#     audio_sampling_rate=hps.data.sampling_rate
# )
net_g.train()
net_d.train()


