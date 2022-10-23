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
from dataloader import FFHQDataset, FangtanDataset
from model import Decoder, GeneratorTrn
import utils
from torch.profiler import profile, record_function, ProfilerActivity
import json
import argparse
import itertools
import math
from morton_code import morton_encode,morton_decode
import numpy as np
import gc



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
torch.backends.cudnn.benchmark = True
global_step = 0
profile1 = False

def main():
  assert torch.cuda.is_available(), "CPU training is not allowed."

  hps = utils.get_hparams()
  run(hps)

def get_datasets(hps):
  train_dataset=None
  eval_dataset=None
  input_size = hps.train.input_size
  hps.net_g.input_length = (input_size//(int(math.sqrt(hps.net_g.scale_factor))**hps.net_g.n_uplayers))**2
  hps.net_d.input_length = (input_size//(int(math.sqrt(hps.net_d.scale_factor))**hps.net_d.n_uplayers))**2
  if hps.data.dataset=='celeba':
    train_dataset = torchvision.datasets.CelebA(
      root="data",
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((input_size,input_size))])
    )
    eval_dataset = torchvision.datasets.CelebA(
      root="data",
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((input_size,input_size))])
    )
  elif hps.data.dataset == 'cifar10':
    train_dataset = torchvision.datasets.CIFAR10(
      root="data",
      train=True,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((input_size,input_size))])
    )
    eval_dataset = torchvision.datasets.CIFAR10(
      root="data",
      train=False,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((input_size,input_size))])
    )
  elif hps.data.dataset=='mnist':
    train_dataset = torchvision.datasets.MNIST(
      root="data",
      train=True,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(input_size)])
    )
    eval_dataset = torchvision.datasets.MNIST(
      root="data", 
      train=False, 
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(input_size)]))
  elif hps.data.dataset=='fangtan':
    train_dataset = FangtanDataset('./data/fangtan/train',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(input_size)]))
    eval_dataset = FangtanDataset('./data/fangtan/eval',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(input_size)]))
  elif hps.data.dataset =='ffhq':
    train_dataset = FFHQDataset('./data/ffhq',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(input_size)]))
    eval_dataset = FFHQDataset('./data/ffhq',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(input_size)]))
  else:
    print("dataset type is not supported, add by yourself")
  return train_dataset,eval_dataset

def run(hps):
  global global_step
  logger = utils.get_logger(hps.model_dir)
  logger.info(hps)
  utils.check_git_hash(hps.model_dir)
  writer = SummaryWriter(log_dir=hps.model_dir)
  writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  torch.manual_seed(hps.train.seed)
  train_dataset,eval_dataset = get_datasets(hps)
  train_loader = DataLoader(train_dataset,
          batch_size=16,
          shuffle=False,
          num_workers=8)

  eval_loader = DataLoader(eval_dataset,
          batch_size=16,
          shuffle=False,
          num_workers=8)
  
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
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  # train_loader = [next(iter(train_loader))]
  

  torch.autograd.set_detect_anomaly(True)
  if hps.data.number==1:
    train_ae(hps,net_g,optim_g,net_d,optim_d,train_loader,eval_loader,scheduler_g,scheduler_d,epoch_str,logger,writer,writer_eval,scaler)
  elif hps.data.number==2:
    train_sr(hps,net_g,optim_g,net_d,optim_d,train_loader,eval_loader,scheduler_g,scheduler_d,epoch_str,logger,writer,writer_eval,scaler)
  elif hps.data.number==3:
    train_contrasive(hps,net_g,optim_g,net_d,optim_d,train_loader,eval_loader,scheduler_g,scheduler_d,epoch_str,logger,writer,writer_eval,scaler)

def evaluate(hps, generator, decoder,eval_loader, writer_eval):
  global global_step
  generator.eval()
  decoder.eval()
  with torch.no_grad():
    for batch_idx, input_dict in enumerate(eval_loader):
      if hps.data.number ==1:
        x,_ = input_dict
      elif hps.data.number==2:
        _,x= input_dict
      elif hps.data.number==3:
        x,_,_ = input_dict
      x = x.to(device)
      x = x*2-1
      x = torch.cat([x,-x],dim=1)
      _,enc_z,_,_,flow2_pyramid_reverse = generator(morton_encode(x),reverse=True)
      flow2_pyramid_reverse.reverse()
      y_dec,_,y_dec_post = decoder(enc_z,flow2_pyramid_reverse)

      samples = generator.sample(8,device)
      samples = torch.unsqueeze(samples,dim=1)
      samples = [img for img in samples]

      break
  # print("eval")
  image_dict = {
    "gen/gt" : utils.plot_image_to_numpy(x),
    "gen/y_dec" : utils.plot_image_to_numpy(y_dec),
    "gen/y_dec_post" : utils.plot_image_to_numpy(y_dec_post),
    "gen/flow1_sampled":utils.plot_images_to_numpy(samples),
  }

  utils.summarize(
    writer=writer_eval,
    global_step=global_step, 
    images=image_dict,
  )
  generator.train()
  decoder.train()

def train_ae(hps,net_g,optim_g,net_d,optim_d,train_loader,eval_loader,scheduler_g,scheduler_d,epoch_str,logger,writer,writer_eval,scaler):
  global global_step
  for epoch in range(epoch_str, hps.train.epochs+1):
    net_g.train()
    for batch_idx, input_dict in enumerate(train_loader):
      x,_ = input_dict
      x = x.to(device)
      x = x*2-1
      with autocast(enabled=hps.train.fp16_run):
        enc_z_gt,z_norm,y,flow1_pyramid_reverse,x_enc,x,mu,log_var,flow2_pyramid_forward = net_g(x.detach())
        flow1_pyramid_reverse.reverse()

        # with autocast(enabled=False):
        loss_recon = F.mse_loss(y,x_enc.detach())
        mu = mu.view(mu.shape[0]*mu.shape[1],mu.shape[2])
        log_var = log_var.view(log_var.shape[0]*log_var.shape[1],log_var.shape[2])
        loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss_g_all = loss_recon+hps.train.kld_weight*loss_kld

      optim_g.zero_grad()
      scaler.scale(loss_g_all).backward()

      with autocast(enabled=hps.train.fp16_run):
        x_enc_reverse,enc_z,z_norm,flow1_pyramid_forward,flow2_pyramid_reverse = net_g(x_enc.detach(),reverse=True)
        # enc_z,z_norm_r,flow2_pyramid_2 = net_g(x_enc.detach(),reverse=True)
        with autocast(enabled=False):
          flow2_pyramid_reverse.reverse()
          loss_x_rv_recon =F.mse_loss(x_enc_reverse,x_enc.detach())

      scaler.scale(loss_x_rv_recon).backward()
      scaler.unscale_(optim_g)
      grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
      scaler.step(optim_g)

      loss_pyramid=0
      loss_pyramid_gt=0
      with autocast(enabled=hps.train.fp16_run):
        flow2_pyramid_reverse = [layer.detach() for layer in flow2_pyramid_reverse]
        y_dec,flow3_pyramid_forward,y_dec_post = net_d(enc_z.detach(),flow2_pyramid_reverse)
        y_dec_gt,flow3_pyramid_forward_gt,y_dec_gt_post = net_d(enc_z_gt.detach(),flow2_pyramid_reverse)
        # with autocast(enabled=False):
          # for i,yy in enumerate(flow3_pyramid_forward):
          #   loss_pyramid=loss_pyramid + torch.abs(yy-flow1_pyramid_reverse[i].detach()).mean()
          # for i,yy in enumerate(y_pyramid_gt):
          #   loss_pyramid_gt=loss_pyramid_gt + torch.abs(yy-flow1_pyramid_2[i].detach()).mean()
        loss_y_dec_recon = F.mse_loss(y_dec,x_enc.detach())
        loss_y_dec_recon_gt =  F.mse_loss(y_dec_gt,x_enc.detach())
        loss_y_dec_post_recon = F.mse_loss(y_dec_post,x.detach())
        loss_y_dec_recon_gt_post =  F.mse_loss(y_dec_gt_post,x.detach())
        loss_d_total = loss_y_dec_recon+loss_y_dec_post_recon+loss_y_dec_recon_gt+loss_y_dec_recon_gt_post
      optim_d.zero_grad()
      scaler.scale(loss_d_total).backward()
      scaler.unscale_(optim_d)
      grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
      scaler.step(optim_d)
      scaler.update()
      
      # torch.cuda.empty_cache()
      # torch.cuda.synchronize()
      # print(torch.cuda.memory_summary(device))
      if global_step%hps.train.log_interval == 0:

        lr = optim_g.param_groups[0]['lr']
        losses = [loss_recon,loss_kld,loss_g_all,loss_y_dec_recon,loss_d_total]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([i.item() for i in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_g_all, "learning_rate": lr}
        scalar_dict.update({"loss/g/recon": loss_recon, "loss/g/kld": loss_kld,"loss/g/x_rv_recon":loss_x_rv_recon})
        scalar_dict.update({
        'loss/d/loss_y_dec_recon':loss_y_dec_recon,
        'loss/d/loss_y_dec_recon_gt':loss_y_dec_recon_gt,
        "loss/d/loss_y_dec_post_recon":loss_y_dec_post_recon,
        "loss/d/loss_y_dec_recon_gt_post":loss_y_dec_recon_gt_post,
        # "loss/d/loss_pyramid":loss_pyramid,
        # 'loss/d/loss_pyramid_gt':loss_pyramid_gt,
        'loss/d/total':loss_d_total})

        #1 represent not reverse, 2 represent reverse
        image_dict = { 
            "img/x" : utils.plot_image_to_numpy(x),
            "img/y" : utils.plot_image_to_numpy(y),
            "img/y_dec" : utils.plot_image_to_numpy(y_dec),
            "img/y_dec_gt" : utils.plot_image_to_numpy(y_dec_gt),
            "img/y_dec_post" : utils.plot_image_to_numpy(y_dec_post),
            "img/y_dec_gt_post" : utils.plot_image_to_numpy(y_dec_gt_post),
            "img/flow1_pyramid_reverse" : utils.plot_images_to_numpy(flow1_pyramid_reverse),
            "img/flow2_pyramid_forward" : utils.plot_images_to_numpy(flow2_pyramid_forward),
            "img/flow2_pyramid_reverse" : utils.plot_images_to_numpy(flow2_pyramid_reverse),
            "img/flow1_pyramid_forward" : utils.plot_images_to_numpy(flow1_pyramid_forward),
            "img/flow3_pyramid_forward" : utils.plot_images_to_numpy(flow3_pyramid_forward),
            "img/flow3_pyramid_forward_gt" : utils.plot_images_to_numpy(flow3_pyramid_forward_gt),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)



      if global_step%hps.train.eval_interval==0 and global_step!=0:
        evaluate(hps, net_g, net_d, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
      #   pass
      global_step += 1
      if profile1==True:
        return
    logger.info('====> Epoch: {}'.format(epoch))
    scheduler_g.step()
    scheduler_d.step()

def train_contrasive(hps,net_g,optim_g,net_d,optim_d,train_loader,eval_loader,scheduler_g,scheduler_d,epoch_str,logger,writer,writer_eval,scaler):
  global global_step
  for epoch in range(epoch_str, hps.train.epochs+1):
    net_g.train()
    for batch_idx, input_dict in enumerate(train_loader):
      frame1,frame2,interval = input_dict
      frame1 = frame1.to(device)
      frame2 = frame2.to(device)
      frame1 = frame1*2-1
      frame2 = frame2*2-1
      frame1 = torch.cat([frame1,-frame1],dim=1)
      frame1_enc=morton_encode(frame1)
      frame2 = torch.cat([frame2,-frame2],dim=1)
      frame2_enc=morton_encode(frame2)
      loss_weight = torch.tensor([1/itv for itv in interval])
      interval_loss=0
      with autocast(enabled=hps.train.fp16_run):
        frame1_enc_reverse,enc_z_frame1,z_norm,flow1_pyramid_forward_frame1,flow2_pyramid_reverse_frame1 = net_g(frame1_enc.detach(),reverse=True)
        frame2_enc_reverse,enc_z_frame2,z_norm,flow1_pyramid_forward_frame2,flow2_pyramid_reverse_frame2 = net_g(frame2_enc.detach(),reverse=True)
        # enc_z,z_norm_r,flow2_pyramid_2 = net_g(x_enc.detach(),reverse=True)
        with autocast(enabled=False):
          flow2_pyramid_reverse_frame1.reverse()
          flow2_pyramid_reverse_frame2.reverse()
          interval_loss =F.mse_loss(enc_z_frame1,enc_z_frame2,reduce=False)
          interval_loss = (loss_weight*interval_loss).mean()

      optim_g.zero_grad()
      interval_loss.backward()
      optim_g.step()

      loss_pyramid=0
      loss_pyramid_gt=0
      with autocast(enabled=hps.train.fp16_run):
        flow2_pyramid_reverse_frame1 = [layer.detach() for layer in flow2_pyramid_reverse_frame1]
        flow2_pyramid_reverse_frame2 = [layer.detach() for layer in flow2_pyramid_reverse_frame2]
        frame1_dec,flow3_pyramid_forward_frame1,frame1_dec_post = net_d(enc_z_frame1.detach(),flow2_pyramid_reverse_frame1)
        frame2_dec,flow3_pyramid_forward_frame2,frame2_dec_post = net_d(enc_z_frame2.detach(),flow2_pyramid_reverse_frame2)
        with autocast(enabled=False):
          recon_interval_loss = F.mse_loss(frame1_dec,frame2_dec,reduce=False)
          recon_interval_loss = (loss_weight*recon_interval_loss).mean()
          post_recon_interval_loss = F.mse_loss(frame1_dec_post,frame2_dec_post,reduce=False)
          post_recon_interval_loss = (loss_weight*post_recon_interval_loss).mean()
          loss_recon_interval_total = recon_interval_loss+post_recon_interval_loss
      optim_d.zero_grad()
      loss_recon_interval_total.backward()
      optim_d.step()
      
      # torch.cuda.empty_cache()
      # torch.cuda.synchronize()
      # print(torch.cuda.memory_summary(device))
      if global_step%hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [interval_loss,recon_interval_loss,post_recon_interval_loss,loss_recon_interval_total]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([i.item() for i in losses] + [global_step, lr])
        
        scalar_dict = {"learning_rate": lr}
        scalar_dict.update({
        'loss/interval_loss':interval_loss,
        'loss/recon_interval_loss':recon_interval_loss,
        "loss/post_recon_interval_loss":post_recon_interval_loss,
        "loss/loss_recon_interval_total":loss_recon_interval_total})

        #1 represent not reverse, 2 represent reverse
        image_dict = { 
          "img/enc_z_frame1" : utils.plot_image_to_numpy(enc_z_frame1),
          "img/enc_z_frame2" : utils.plot_image_to_numpy(enc_z_frame2),
          "img/frame1_dec" : utils.plot_image_to_numpy(frame1_dec),
          "img/frame2_dec" : utils.plot_image_to_numpy(frame2_dec),
          "img/frame1_dec_post" : utils.plot_image_to_numpy(frame1_dec_post),
          "img/frame2_dec_post" : utils.plot_image_to_numpy(frame2_dec_post),
          "img/flow3_pyramid_forward_frame1" : utils.plot_images_to_numpy(flow3_pyramid_forward_frame1),
          "img/flow3_pyramid_forward_frame2" : utils.plot_images_to_numpy(flow3_pyramid_forward_frame2),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)

      if global_step%hps.train.eval_interval==0 and global_step!=0:
        evaluate(hps, net_g, net_d, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
      #   pass
      global_step += 1
      if profile1==True:
        return
    logger.info('====> Epoch: {}'.format(epoch))
    scheduler_g.step()
    scheduler_d.step()

def train_sr(hps,net_g,optim_g,net_d,optim_d,train_loader,eval_loader,scheduler_g,scheduler_d,epoch_str,logger,writer,writer_eval,scaler):
  global global_step
  for epoch in range(epoch_str, hps.train.epochs+1):
    net_g.train()
    for batch_idx, input_dict in enumerate(train_loader):
      lq,hq = input_dict
      lq = lq.to(device)
      lq = lq*2-1
      hq = hq.to(device)
      hq = hq*2-1
      
      lq = torch.cat([lq,-lq],dim=1)
      lq_enc=morton_encode(lq)
      with autocast(enabled=hps.train.fp16_run):
        enc_z_gt,z_norm,lq_gen,flow1_pyramid_reverse,hq_enc,hq,mu,log_var,flow2_pyramid_forward = net_g(hq.detach())
        flow1_pyramid_reverse.reverse()

        loss_recon = F.mse_loss(lq_gen,lq_enc.detach())
        mu = mu.view(mu.shape[0]*mu.shape[1],mu.shape[2])
        log_var = log_var.view(log_var.shape[0]*log_var.shape[1],log_var.shape[2])
        loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # loss_kld = F.l1_loss(z_norm,torch.zeros_like(z_norm))
        loss_g_all = loss_recon+hps.train.kld_weight*loss_kld

      optim_g.zero_grad()
      loss_g_all.backward()

      with autocast(enabled=hps.train.fp16_run):
        hq_gen,enc_z,z_norm,flow1_pyramid_forward,flow2_pyramid_reverse = net_g(lq_enc.detach(),reverse=True)
        # enc_z,z_norm_r,flow2_pyramid_2 = net_g(x_enc.detach(),reverse=True)
        flow2_pyramid_reverse.reverse()
        loss_x_rv_recon =F.mse_loss(hq_gen,hq_enc.detach())

      loss_x_rv_recon.backward()
      optim_g.step()

      loss_pyramid=0
      loss_pyramid_gt=0
      with autocast(enabled=hps.train.fp16_run):
        flow2_pyramid_reverse = [layer.detach() for layer in flow2_pyramid_reverse]
        hq_dec,flow3_pyramid_forward,hq_dec_post = net_d(enc_z.detach(),flow2_pyramid_reverse)
        hq_dec_gt,flow3_pyramid_forward_gt,hq_dec_gt_post = net_d(enc_z_gt.detach(),flow2_pyramid_reverse)

        loss_hq_dec_recon = F.mse_loss(hq_dec,hq_enc.detach())
        loss_hq_dec_recon_gt =  F.mse_loss(hq_dec_gt,hq_enc.detach())
        loss_hq_dec_post_recon = F.mse_loss(hq_dec_post,hq.detach())
        loss_hq_dec_recon_gt_post =  F.mse_loss(hq_dec_gt_post,hq.detach())
        loss_d_total = loss_hq_dec_recon+loss_hq_dec_post_recon+loss_hq_dec_recon_gt+loss_hq_dec_recon_gt_post
      optim_d.zero_grad()
      loss_d_total.backward()
      optim_d.step()
      
      if global_step%hps.train.log_interval == 0:

        lr = optim_g.param_groups[0]['lr']
        losses = [loss_recon,loss_kld,loss_g_all,loss_hq_dec_recon,loss_d_total]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([i.item() for i in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_g_all, "learning_rate": lr}
        scalar_dict.update({"loss/g/recon": loss_recon, "loss/g/kld": loss_kld,"loss/g/x_rv_recon":loss_x_rv_recon})
        scalar_dict.update({
        'loss/d/loss_hq_dec_recon':loss_hq_dec_recon,
        'loss/d/loss_hq_dec_recon_gt':loss_hq_dec_recon_gt,
        "loss/d/loss_hq_dec_post_recon":loss_hq_dec_post_recon,
        "loss/d/loss_hq_dec_recon_gt_post":loss_hq_dec_recon_gt_post,
        'loss/d/total':loss_d_total})

        image_dict = { 
            "img/hq" : utils.plot_image_to_numpy(hq),
            "img/lq" : utils.plot_image_to_numpy(lq),
            # "img/hq_gen" : utils.plot_image_to_numpy(hq_gen),
            # "img/lq_gen" : utils.plot_image_to_numpy(lq_gen),
            "img/hq_dec" : utils.plot_image_to_numpy(hq_dec),
            "img/hq_dec_gt" : utils.plot_image_to_numpy(hq_dec_gt),
            "img/hq_dec_post" : utils.plot_image_to_numpy(hq_dec_post),
            "img/hq_dec_gt_post" : utils.plot_image_to_numpy(hq_dec_gt_post),
            "img/flow1_pyramid_reverse" : utils.plot_images_to_numpy(flow1_pyramid_reverse),
            "img/flow2_pyramid_forward" : utils.plot_images_to_numpy(flow2_pyramid_forward),
            "img/flow2_pyramid_reverse" : utils.plot_images_to_numpy(flow2_pyramid_reverse),
            "img/flow1_pyramid_forward" : utils.plot_images_to_numpy(flow1_pyramid_forward),
            "img/flow3_pyramid_forward" : utils.plot_images_to_numpy(flow3_pyramid_forward),
            "img/flow3_pyramid_forward_gt" : utils.plot_images_to_numpy(flow3_pyramid_forward_gt),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)



      if global_step%hps.train.eval_interval==0 and global_step!=0:
        evaluate(hps, net_g, net_d, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
      #   pass
      global_step += 1
      if profile1==True:
        return
    logger.info('====> Epoch: {}'.format(epoch))
    scheduler_g.step()
    scheduler_d.step()
if __name__ == "__main__":
  main()
