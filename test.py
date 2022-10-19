import math
from torchvision.utils import save_image

import torch
import torchvision
from torch.nn import functional as F
from torch import nn
from model import Decoder, Encoder, GeneratorTrn, Hierarchy_flow, ResidualConv1d, ResidualCouplingBlock, ResidualCouplingHierarchyBlock, ResidualUpsampleCouplingLayer,ResidualUpsampleCouplingBlock, SimpleDecoder
import pymorton as pm

import utils

# input x   std s


#√
def hierarchy_flow(x, s, inverse=False):
  l = x.shape[0]
  assert((l & (l-torch.tensor(1)) == 0) and l != 0)
  assert(s.shape[0] == 2*l-1)
  bit_l = l.bit_length()

  if inverse==True:
    now = 0
    for i in range(bit_l):
      a = s[now:now+(1 << i)]
      now = now+(1 << i)
      a = a.repeat_interleave(l//(1 << i), dim=0)
      x=x-a
  else:
    now = 0
    for i in reversed(range(bit_l)):
      a = s[now:now+(1 << i)]
      now = now+(1 << i)
      a = a.repeat_interleave(l//(1 << i), dim=0)
      x = x+a
  return x

#√
def reverse_hierarchy_flow(x, s, inverse=False):
  l = x.shape[0]
  assert((l & (l-torch.tensor(1)) == 0) and l != 0)
  assert(s.shape[0] == 2*l-1)
  bit_l = l.bit_length()

  if inverse==True:
    now = 0
    for i in reversed(range(bit_l)):
      a = s[now:now+(1 << i)]
      now = now+(1 << i)
      a = a.repeat_interleave(l//(1 << i), dim=0)
      x = x-a
  else:
    now = 0
    for i in range(bit_l):
      a = s[now:now+(1 << i)]
      now = now+(1 << i)
      a = a.repeat_interleave(l//(1 << i), dim=0)
      x = x+a
  return x

#√
def morton_encode(xx):
  l = xx.shape[0]**2
  bit_l = l.bit_length()
  ret = torch.zeros(l)
  for i in range(xx.shape[0]):
    for j in range(xx.shape[0]):
      ij=0
      ij = pm.interleave2(i,j)
      # for bit in range(bit_l):
      #   if bit%2==1:
      #     ij=ij+(((i>>(bit//2))&1)<<bit)
      #   else:
      #     ij=ij+(((j>>(bit//2))&1)<<bit)
      # print(i,j,ij)
      ret[ij]=xx[i][j]
  return ret


#√
def morton_decode(x):
  l = x.shape[0]
  bit_l = l.bit_length()
  ret = torch.zeros_like(x)
  ret=ret.view(int(math.sqrt(l)),int(math.sqrt(l)))
  # print(ret.shape)
  for ij in range(l):
    i=0
    j=0
    for bit in range(bit_l):
      if bit % 2==1:
        i=i+(((ij>>bit)&1)<<(bit//2))
      else:
        j=j+(((ij>>bit)&1)<<(bit//2))
    # print(i,j,ij)
    ret[i][j]=x[ij]
  return ret

class Upsample_flow(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.pool = nn.AvgPool1d(4,stride=4)

  def forward(self,x, s, scale_factor=4, inverse=False):
    l = x.shape[0]
    # assert(s.shape[0] == l*scale_factor)
    if inverse==True:
      x=x-s
      downsampled = self.pool(x)
      ret = downsampled
    else:
      upsampled = x.repeat_interleave(scale_factor,dim=1)
      upsampled = upsampled+s
      ret = upsampled
    return ret

class Test(nn.Module):
  def __init__(self,in_channel,out_channel):
    super().__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.l1 = nn.Conv1d(in_channel, out_channel, 1)
    self.l1.weight.data.zero_()
    self.l1.bias.data.zero_()
  def forward(self, x):
    return self.l1(x)

w=4
###how to inversable downsample???###
###one way is to use loss clamp four value to the same###
# def downsample_flow(x, s, scale_factor=4,inverse=False):
#   downsample_layer = nn.AvgPool1d(scale_factor, stride=scale_factor)
#   downsampled = downsample_layer(x)
#   m = downsampled.repeat_interleave(scale_factor) - x 
#   # self.register_buffer("mean", m) 
#   return downsampled

# x = torch.zeros(256*256)
# s = torch.ones(2*256*256-1)


### hierachy_flow test ###
# x = torch.zeros(w*w)
# s = torch.ones(2*w*w-1)
# # print(hierarchy_flow(x,s).shape)
# print(hierarchy_flow(x,s))
# y=hierarchy_flow(x,s)
# print(x)
# print(hierarchy_flow(y,s,inverse=True))

### morton code test ###
# img = torch.rand(w, w)
# save_image(img,'2.png')
# print(morton_encode(img))
# a= morton_encode(img)
# print(morton_decode(a))

### upsample test ###
# img = torch.rand(w, w)
# save_image(img,'2.png')
# s = torch.zeros(w*w*4)
# img_encoded = morton_encode(img)
# uf = Upsample_flow()
# img_encoded = img_encoded.unsqueeze(dim=0)
# img1 = uf(img_encoded,s)
# img2 = morton_decode(img1[0])
# save_image(img2,'3.png')
# img3 = uf(img1,s,inverse=True)
# img4 = morton_decode(img3[0])
# save_image(img4,'4.png')

# from model import WN,ResidualCouplingBlock,ResidualCouplingLayer


# rcb = ResidualCouplingBlock(4, 192, 5, 1, 4)
# # rcl = ResidualCouplingLayer(192, 192, 5, 1, 4, mean_only=True)
# x = torch.rand(16,4,192)
# x = rcb(x,torch.ones(16,1,192))
# print(x.shape)



# l = torch.nn.Conv1d(1,1,5,4,2)
# l = torch.nn.Conv1d(1,1,3,2,1)
# x = torch.rand(1,512)
# print(l(x).shape)

# g = Encoder(2)
# x =  sum(p.numel() for p in g.parameters() if p.requires_grad)
# print(x)
# x = torch.rand(1,2,32*32)
# print(g(x).shape)

# g = ResidualUpsampleCouplingBlock(2,256,3,4,1,4,3,16,4)
# x = torch.rand(4,2,16)
# print(g(x).shape)
# x =  sum(p.numel() for p in g.parameters() if p.requires_grad)
# print(x)

# x = torch.rand(2,1,1024)
# x = x*2-1
# print(x)
# x1 = -x
# print(x1)
# x = torch.cat([x,x],dim=1)
# print(x.shape)

# m = Morton_encode()
# x=torch.rand(1,1,32,32)
# print(m(x).shape)

# self.flow1 = ResidualCouplingBlock(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers)
# channels,
# hidden_channels,
# kernel_size,
# dilation_rate,
# n_layers,
# n_flows=4,

# g = Hierarchy_flow(4,32,3,1,2,16)
# x = torch.rand(6,4,16)
# print(g(x).shape)
# g = ResidualCouplingHierarchyBlock(2,32,3,1,2,1,16)
# x = torch.rand(1,2,16)
# y = g(x)
# print(x)
# y = g(y,reverse=True)
# print(y)


#todo list 
# 1. hierarchy flow add √
# 2. add diffusion decoder or cnn decoder √
# 3. add dimension of z B,4C,T -> B,C,4T √
# 4. understand WN and replace by unet framework 
# 5. pyramid loss with Encoder √
# 6. unet skip connection with upsample flow generator √

# B,2,16 -> B,2,4,4 -> B,2,8,8 ->B,2,16,16 -> B,2,32,32

# sd = nn.ModuleList([SimpleDecoder(2,2),SimpleDecoder(2,2),SimpleDecoder(2,2)])
# x = torch.rand(4,2,4,4)
# for sdd in sd:
#   x=sdd(x)
#   print(x.shape)

# img=torchvision.io.read_image("./1.png")
# for i in range(img.shape[0]):
#   img[i] = morton_encode(img[i]/255)
# print(img.shape)
# torchvision.utils.save_image(img.float(),'1_encoded.png')
# img1 = torch.ones_like(img)-img/255
# torchvision.utils.save_image(img1.float(),'1_reverse.png')

# g = GeneratorTrn(2,256,3,4,1,4,3,False,16)
# x = torch.rand(1,2,32*32)
# enc_z,z_norm,y,enc_pyramid = g(x)
# for i in enc_pyramid:
#   print(i.shape)

# print('!')
# enc_z,z_norm,flow_pyramid=g.reverse(y)
# for i in flow_pyramid:
#   print(i.shape)

# print('!')
# d = Decoder(2,2)
# x = torch.rand(1,2,4,4)
# flow_pyramid=[]
# for i in [4,8,16,32]:
#   flow_pyramid.append(torch.rand(1,2,i,i))
# y,y_pyramid = d(x,flow_pyramid)
# print(y.shape)
# for i in y_pyramid:
#   print(i.shape)

# import time
# x=torch.rand(512,512)
# print(x)
# time_start=time.time()
# # import pymorton as pm

# # # pm.deinterleave2(mortoncode)             # (100, 200)
# # for i in range(512):
# #   for j in range(512):
# #     pm.interleave2(i,j)
# x = morton_encode(x)
# print(x)
# time_end=time.time()
# print('time cost',time_end-time_start,'s')

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='logs/test')
# netd = Decoder(2,2)
# writer.add_graph(netg,input_to_model = torch.rand(1,2,32*32))
# # writer.add_graph(netd,input_to_model = [torch.rand(1,2,4,4),flow_pyramid])
# writer.close()

# morton_dec = Morton_decode()
# morton_enc = Morton_encode()
# net_g = GeneratorTrn(2,256,3,4,1,4,3,False,16)
# optim_g = torch.optim.AdamW(
#     net_g.parameters())

# x = torch.rand(1,1,32,32)
# x_enc=morton_enc(x)
# x = x*2-1
# x = torch.cat([x,-x],dim=1)
# x_enc = x_enc*2-1
# x_enc = torch.cat([x_enc,-x_enc],dim=1)
# enc_z_gt,z_norm,y,enc_pyramid = net_g(x_enc.detach())
# enc_pyramid_t = [morton_dec(i).clone().detach() for i in enc_pyramid]
# enc_pyramid_t.reverse()
# loss_recon = F.mse_loss(y,x_enc.detach())
# loss_kld = 0.01*F.l1_loss(z_norm,torch.zeros_like(z_norm))
# loss_g_all = loss_recon+loss_kld

# optim_g.zero_grad()
# loss_g_all.backward(retain_graph=True)

# loss_recon_1 = F.mse_loss(y,x_enc.detach())
# loss_recon_1.backward()
# optim_g.step()

# import time
# import morton_code
# x = torch.rand(16,2,256,256)
# # print(x)
# time_start=time.time()
# encoded = morton_code.morton_encode(x)
# # print(encoded)
# time_end=time.time()
# print('time cost',time_end-time_start,'s')
# decoded = morton_code.morton_decode(encoded)
# print(decoded)

#1.wgan loss or kld loss √
#2.tensorboard vidulization √

#1.post unet 
#2.decoder unet √
#3.inference

# from torchvision import models
# from torchsummary import summary

# net_g = GeneratorTrn(
#     in_channels= 2,
#     hidden_channels= 128,
#     n_layers= 4,
#     n_uplayers= 3,
#     kernel_size= 3,
#     scale_factor= 4,
#     dilation_rate= 1,
#     expand=False,
#     input_length=16)

# net_d = Decoder(in_channels= 2,
#     hidden_channels= 128,
#     n_layers= 4,
#     n_uplayers= 3,
#     kernel_size= 3,
#     scale_factor= 4,
#     dilation_rate= 1,
#     input_length=16)
# for name, p in net_g.named_parameters():
#   if p.requires_grad:
#     if p.numel()>1000000:
#       print(name)
#       print(p.numel())
# x =  sum(p.numel() for p in net_g.parameters() if p.requires_grad)
# # print(x)

# for name, p in net_d.named_parameters():
#   if p.requires_grad:
#     if p.numel()>1000000:
#       print(name)
#       print(p.numel())
# x =  sum(p.numel() for p in net_d.parameters() if p.requires_grad)
# print(x)

from torchvision import transforms

train_dataset = torchvision.datasets.CelebA(
  root="data",
  download=True,
  transform=transforms.Compose([transforms.ToTensor()])
)
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4)

train_loader = [next(iter(train_loader))]
for x,_ in train_loader:
  print(x.shape)
  print(torch.max(x[0]))
  print(torch.min(x[0]))
  print(x[0])
  # save_image(x,'celeba1.png')
