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
import numpy as np


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
torch.backends.cudnn.benchmark = True
global_step = 0
profile1 = False

def main():
  assert torch.cuda.is_available(), "CPU training is not allowed."

  hps = utils.get_hparams()
  # with profile(activities=[
  #       ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
  run(hps)
  # print(prof.table())
  # prof.export_chrome_trace('./profile.json')

def run(hps):
  global global_step
  logger = utils.get_logger(hps.model_dir)
  logger.info(hps)
  utils.check_git_hash(hps.model_dir)
  writer = SummaryWriter(log_dir=hps.model_dir)
  writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  torch.manual_seed(hps.train.seed)

  train_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(32)])
  )

  train_loader = DataLoader(train_dataset,
          batch_size=16,
          shuffle=False,
          num_workers=4)
  eval_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(32)]))
  eval_loader = DataLoader(eval_dataset,
          batch_size=16,
          shuffle=False,
          num_workers=4)
  
  net_g = GeneratorTrn(**hps.model).to(device)
  optim_g = torch.optim.AdamW(
    net_g.parameters(), 
    hps.train.learning_rate, 
    betas=hps.train.betas, 
    eps=hps.train.eps)
  
  net_d = Decoder(2,2).to(device)
  optim_d = torch.optim.AdamW(
    net_d.parameters(), 
    hps.train.learning_rate, 
    betas=hps.train.betas, 
    eps=hps.train.eps)

  # try:
  #   _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
  #   _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
  #   global_step = (epoch_str - 1) * len(train_loader)
  # except:
  #   epoch_str = 1
  #   global_step = 0
  epoch_str=1
  global_step=0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  # hps.train.epochs=1
  # train_loader = [next(iter(train_loader))]
  


  torch.autograd.set_detect_anomaly(True)
  for epoch in range(epoch_str, hps.train.epochs+1):
    net_g.train()
    for batch_idx, (x, _) in enumerate(train_loader):
      x = x.to(device)
      x = x*2-1
      with autocast(enabled=hps.train.fp16_run):
        enc_z_gt,z_norm,y,enc_pyramid,x_enc,x,mu,log_var = net_g(x.detach())
        enc_pyramid_t = [morton_decode(i).clone().detach() for i in enc_pyramid]
        enc_pyramid_t.reverse()
        # print(enc_pyramid[0].shape)
        with autocast(enabled=False):
          loss_recon = F.mse_loss(y,x_enc.detach())
          mu = mu.view(mu.shape[0]*mu.shape[1],mu.shape[2])
          log_var = log_var.view(log_var.shape[0]*log_var.shape[1],log_var.shape[2])
          loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
          loss_g_all = loss_recon+hps.train.kld_weight*loss_kld

      optim_g.zero_grad()
      loss_g_all.backward(retain_graph=True)

      with autocast(enabled=hps.train.fp16_run):
        enc_z,z_norm_r,flow_pyramid = net_g(x_enc.detach(),reverse=True)
        with autocast(enabled=False):
          flow_pyramid_t = [morton_decode(i).to(device).detach() for i in flow_pyramid]
          flow_pyramid_t.reverse()
          loss_z_recon =F.mse_loss(enc_z,enc_z_gt.detach())

      loss_z_recon.backward(retain_graph=True)

      loss_pyramid=0
      loss_pyramid_gt=0
      with autocast(enabled=hps.train.fp16_run):
        enc_z = morton_decode(enc_z).to(device).clone()
        enc_z_gt = morton_decode(enc_z_gt).to(device).clone()
        y_dec,y_pyramid = net_d(enc_z,flow_pyramid_t)
        y_dec_gt,y_pyramid_gt = net_d(enc_z_gt,flow_pyramid_t)
        with autocast(enabled=False):
          for i,yy in enumerate(y_pyramid):
            loss_pyramid=loss_pyramid + torch.abs(yy-enc_pyramid_t[i].detach()).mean()
          for i,yy in enumerate(y_pyramid_gt):
            loss_pyramid_gt=loss_pyramid_gt + torch.abs(yy-enc_pyramid_t[i].detach()).mean()
          loss_y_dec_recon = F.mse_loss(y_dec,x.detach())
          loss_y_dec_recon_gt =  F.mse_loss(y_dec_gt,x.detach())
          loss_d_total = loss_pyramid+loss_y_dec_recon+loss_pyramid_gt+loss_y_dec_recon_gt
      optim_d.zero_grad()
      loss_d_total.backward()
      optim_d.step()
      optim_g.step()

      if global_step%hps.train.log_interval == 0:
        pic_list=[]
        y0,y1=torch.split(y,[1,1],dim=1)
        y0 = (y0+1)/2
        y0 = morton_decode(y0)

        lr = optim_g.param_groups[0]['lr']
        losses = [loss_recon,loss_kld,loss_g_all,loss_y_dec_recon,loss_pyramid,loss_d_total]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([i.item() for i in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_g_all, "learning_rate": lr}
        scalar_dict.update({"loss/g/recon": loss_recon, "loss/g/kld": loss_kld,"loss/g/z_recon":loss_z_recon})
        scalar_dict.update({'loss/d/loss_y_dec_recon':loss_y_dec_recon,'loss/d/loss_y_dec_recon_gt':loss_y_dec_recon_gt,
        'loss/d/loss_pyramid':loss_pyramid,'loss/d/loss_pyramid_gt':loss_pyramid_gt,'loss/d/total':loss_d_total})

        image_dict = { 
            "img/flow_gen" : utils.plot_image_to_numpy(y0[0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)

        for i in enc_pyramid_t:
            i0,i1=torch.split(i,[1,1],dim=1)
            image=i0[0].data.cpu().numpy()
            image=transforms.ToPILImage()(torch.from_numpy(image))
            image=transforms.functional.resize(image, [32,32], interpolation=2)
            image=transforms.ToTensor()(image)
            image=image.numpy()
            pic_list.append(image)
        for i in y_pyramid:
            i0,i1=torch.split(i,[1,1],dim=1)
            image=i0[0].data.cpu().numpy()
            image=transforms.ToPILImage()(torch.from_numpy(image))
            image=transforms.functional.resize(image, [32,32], interpolation=2)
            image=transforms.ToTensor()(image)
            image=image.numpy()
            pic_list.append(image)
        for i in y_pyramid_gt:
            i0,i1=torch.split(i,[1,1],dim=1)
            image=i0[0].data.cpu().numpy()
            image=transforms.ToPILImage()(torch.from_numpy(image))
            image=transforms.functional.resize(image, [32,32], interpolation=2)
            image=transforms.ToTensor()(image)
            image=image.numpy()
            pic_list.append(image)

        output_list=[enc_z, enc_z_gt, y_dec, y_dec_gt]

        pic_y=np.reshape(y0[0].data.cpu().numpy(),(1,32,32))
        pic_list.append(pic_y)

        for i in output_list:
            i0,i1=torch.split(i,[1,1],dim=1)
            image=i0[0].data.cpu().numpy()
            image=transforms.ToPILImage()(torch.from_numpy(image))
            image=transforms.functional.resize(image, [32,32], interpolation=2)
            image=transforms.ToTensor()(image)
            image=image.numpy()
            pic_list.append(image)
        pic_list=np.array(pic_list)
        images = np.reshape(pic_list[0:], (len(pic_list), 1, 32, 32))
        writer.add_images('test__', images, global_step=global_step)

      if global_step%hps.train.eval_interval==0 and global_step!=0:
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
      #   pass
      global_step += 1
      if profile1==True:
        return
    logger.info('====> Epoch: {}'.format(epoch))
    scheduler_g.step()


if __name__ == "__main__":
  main()
