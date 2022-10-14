import copy
import math
from tkinter import E
from turtle import forward
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm
import commons

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) 
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)

LRELU_SLOPE = 0.1
class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    # self.post.weight.data.zero_()
    # self.post.bias.data.zero_()

  # Shapes:
  #     - x: :math:`[B, C, T]`
  #     - x_mask: :math:`[B, 1, T]`
  #     - g: :math:`[B, C, 1]`
  def forward(self, x, g=None, reverse=False):
    # print(x.shape)
    # print(self.half_channels)
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) 
    h = self.enc(h, g=g)
    stats = self.post(h)
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs)
      x = torch.cat([x0, x1], 1)
      return x 
    else:
      x1 = (x1 - m) * torch.exp(-logs)
      x = torch.cat([x0, x1], 1)
      return x

class ResidualUpsampleCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      scale_factor,
      dilation_rate,
      n_layers,
      input_length,
      p_dropout=0):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.scale_factor = scale_factor
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.input_length = input_length


    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
    self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()
    self.proj = nn.Linear(self.input_length,self.input_length*self.scale_factor)

    self.pre1 = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc1 = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
    self.post1 = nn.Conv1d(hidden_channels, self.half_channels, 1)
    self.post1.weight.data.zero_()
    self.post1.bias.data.zero_()

    self.pool0 = nn.AvgPool1d(self.scale_factor,stride=self.scale_factor)
    self.pool1 = nn.AvgPool1d(self.scale_factor,stride=self.scale_factor)

  #x: B,C,T
  def forward(self, x, scale_factor=4, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    # assert(s.shape[2] == l*self.scale_factor)
    if reverse==True:
      s2 = self.pre1(x1)
      s2 = self.enc1(s2)
      s2 = self.post1(s2)
      downsampled_x0 = self.pool0(x0-s2)
      s1 = self.pre(downsampled_x0)
      s1 = self.enc(s1)
      s1 = self.post(s1)
      s1 = self.proj(s1)
      # s1 = s1.tranpose(0,1)
      downsampled_x1 = self.pool1(x1-s1)
      ret = torch.cat([downsampled_x0,downsampled_x1],1)
    else:
      s1 = self.pre(x0)
      s1 = self.enc(s1)
      s1 = self.post(s1)
      s1 = self.proj(s1)
      # s1 = s1.tranpose(0,1)
      upsampled_x1 = x1.repeat_interleave(scale_factor,dim=2) + s1
      s2 = self.pre1(upsampled_x1)
      s2 = self.enc1(s2)
      s2 = self.post1(s2)
      upsampled_x0 = x0.repeat_interleave(scale_factor,dim=2) + s2
      ret = torch.cat([upsampled_x0,upsampled_x1],1)
    return ret

class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    return x

class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(Flip())

  def forward(self, x, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x = flow(x, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, g=g, reverse=reverse)
    return x



class ResidualUpsampleCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      scale_factor,
      dilation_rate,
      n_layers,
      n_uplayers,
      input_length,
      n_flows=4
      ):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.scale_factor=scale_factor
    self.input_length=input_length

    self.flows = nn.ModuleList()
    for i in range(n_uplayers):
      self.flows.append(ResidualUpsampleCouplingLayer(channels,hidden_channels,kernel_size,scale_factor,dilation_rate,n_layers,input_length*(scale_factor**i)))
      self.flows.append(Flip())
      self.flows.append(ResidualCouplingHierarchyBlock(channels,hidden_channels,kernel_size,dilation_rate,n_layers,1,input_length*(scale_factor**(i+1))))

  def forward(self, x, reverse=False):
    flow_pyramid=[]
    if not reverse:
      # print(self.flows.size())
      for flow in self.flows:
        x = flow(x, reverse=reverse)
        # print(x.shape)
      return x
    else:
      for i,flow in enumerate(reversed(self.flows)):
        if i%3==0:
          flow_pyramid.append(x)
        x = flow(x, reverse=reverse)
      flow_pyramid.append(x)
      return x,flow_pyramid

class ResidualConv1d(nn.Module):
  def __init__(self,
      input_channels,
      output_channels,
      kernel_size,
      stride,
      padding) -> None:
    super().__init__()
    self.input_channels=input_channels
    self.output_channels=output_channels
    self.stride=stride
    self.conv = nn.Sequential(
      nn.Conv1d(input_channels, output_channels, kernel_size, stride=1, padding=padding, bias=False),
      nn.LeakyReLU(negative_slope=LRELU_SLOPE,inplace=True),
    )
    self.pool1 = nn.AvgPool1d(stride,stride)
    self.pool2 = nn.AvgPool1d(stride,stride)
  def forward(self, x):
    # print(x.shape)
    if self.output_channels>self.input_channels:
      y = self.conv(self.pool1(x))
      x = self.pool2(x).repeat_interleave(self.output_channels//self.input_channels,dim=1)
      return x+y
    else:
      y = self.conv(x)
      return y

#input mnist B,2,1024
class Encoder(nn.Module):
  def __init__(self,
      in_channels,
      expand
      ) -> None:
    super().__init__()
    self.in_channels=in_channels
    self.expand=expand
    modules = []
    t = in_channels
    hidden_dims=[32,64,128,256]
    self.pre = nn.Sequential(ResidualConv1d(in_channels,4*in_channels,25,4,12),
                            ResidualConv1d(in_channels*4,in_channels,25,1,12))
    for h_dim in hidden_dims:
      modules.append(
        nn.Sequential(
          ResidualConv1d(in_channels,h_dim,9,2,4),
          ResidualConv1d(h_dim,in_channels,9,1,4),
        )
      )
    # modules.append(ResidualConv1d(in_channels,1,9,1,4))
    self.convs = nn.ModuleList(modules)
  def forward(self, x):
    #output B,256,16
    #B,C,T
    enc_pyramid = []
    enc_pyramid.append(x)
    x = self.pre(x)
    enc_pyramid.append(x)
    for i,c in enumerate(self.convs):
      # print(i)
      x = c(x)
      if i%2==1:
        # print(len(enc_pyramid))
        enc_pyramid.append(x)
    return x,enc_pyramid
    

class FlowGenerator(nn.Module):
  """
  Flow Based Invertable Encoder
  """
  def __init__(self,
      inter_channels,
      hidden_channels,
      kernel_size,
      scale_factor,
      dilation_rate,
      n_layers,
      n_uplayers,
      input_length
      ):
    super().__init__()
    self.flow1 = ResidualCouplingBlock(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers)
    self.flow2 = ResidualCouplingBlock(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers)
    self.upflow = ResidualUpsampleCouplingBlock(inter_channels,hidden_channels,kernel_size,scale_factor,dilation_rate,n_layers,n_uplayers,input_length)
    # self.fc_mu = nn.Linear(input_length, input_length)
    # self.fc_var = nn.Linear(input_length, input_length)
  def forward(self,x,reverse=False):
    """
    z : B,C,L
    """
    l = x.shape
    flow_pyramid=None
    if reverse==False:
      z = self.flow1(x)
      z_norm = z
      # mu = self.fc_mu(z)
      # log_var = self.fc_var(z)
      z = self.flow2(z)
      z = self.upflow(z)
      # print(z.shape)
      return z,z_norm
    else:
      z,flow_pyramid = self.upflow(x,reverse=True)
      z = self.flow2(z,reverse=True)
      z_norm=z
      z = self.flow1(z,reverse=True)
      return z,z_norm,flow_pyramid

class GeneratorTrn(nn.Module):
  def __init__(self,
      in_channels,
      hidden_channels,
      kernel_size,
      scale_factor,
      dilation_rate,
      n_layers,
      n_uplayers,
      expand,
      input_length
    ) -> None:
    super().__init__()
    self.enc = Encoder(in_channels,expand)
    self.flow_g = FlowGenerator(in_channels,hidden_channels,kernel_size,scale_factor,dilation_rate,n_layers,n_uplayers,input_length)
  def forward(self,x):
    enc_z,enc_pyramid = self.enc(x)
    y,z_norm = self.flow_g(enc_z)
    # print(y.shape)
    return enc_z,z_norm,y,enc_pyramid
  def reverse(self,x):
    enc_z,z_norm,flow_pyramid = self.flow_g(x,reverse=True)
    return enc_z,z_norm,flow_pyramid

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
def Normalize(in_channels, num_groups=1):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x

class Decoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      ) -> None:
     super().__init__()
     self.model = nn.ModuleList([
      SimpleDecoder(in_channels*2,in_channels),
      SimpleDecoder(in_channels*2,in_channels),
      SimpleDecoder(in_channels*2,in_channels)
     ])
  def forward(self, x,unet_pyramid):
    x_pyramid=[]
    x_pyramid.append(x)
    for i,layer in enumerate(self.model):
      # print(x.shape)
      # print(unet_pyramid[i].shape)
      x = torch.cat([x,unet_pyramid[i]],dim=1)
      x = layer(x)
      x_pyramid.append(x)
    return x,x_pyramid


#√
class Hierarchy_flow(nn.Module):
  def __init__(self,
      input_channel,
      hidden_channel,
      kernel_size,
      dilation_rate,
      n_layers,
      input_length,
      p_dropout=0
      ) -> None:
    super().__init__()
    self.input_channel = input_channel
    self.hidden_channel = hidden_channel
    self.input_length = input_length
    self.half_channel = input_channel//2
    self.pre = nn.Conv1d(self.half_channel,hidden_channel,1)
    self.enc = WN(hidden_channel, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
    self.proj = nn.Conv1d(self.hidden_channel, self.half_channel, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()
    self.proj_layer = nn.Linear(self.input_length,self.input_length*2-1)

  #x: B,C,T
  def forward(self, x, reverse=False):
    #x0:B,C/2,T
    x0,x1 = torch.split(x,[self.half_channel]*2,1)
    # print(x0)
    l = x1.shape[2]
    s = x0
    s = self.pre(s)
    s = self.enc(s)
    s = self.proj(s)
    s = self.proj_layer(s)
    #s: B,C/2,2*T-1
    # assert((l & (l-torch.tensor(1)) == 0) and l != 0)
    # assert(s.shape[0] == 2*l-1)
    bit_l = l.bit_length()

    if reverse==True:
      # print(s)
      now = 2*self.input_length-1
      for i in range(bit_l):
        # print(now-(1<<i),now)
        a = s[:,:,now-(1<<i):now]
        now = now-(1 << i)
        a = a.repeat_interleave(l//(1 << i), dim=2)
        # print(a)
        x1=x1-a
    else:
      # print(s)
      now = 0
      for i in reversed(range(bit_l)):
        # print(now,now+(1<<i))
        a = s[:,:,now:now+(1 << i)]
        now = now+(1 << i)
        a = a.repeat_interleave(l//(1 << i), dim=2)
        # print(a)
        x1 = x1+a
    x = torch.cat([x0,x1],1)
    return x

class ResidualCouplingHierarchyBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows,
      input_length
      ):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(Hierarchy_flow(channels, hidden_channels, kernel_size, dilation_rate, n_layers,input_length))
      self.flows.append(Flip())

  def forward(self, x, reverse=False):
    if reverse==False:
      for flow in self.flows:
        x = flow(x, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, reverse=reverse)
    return x

#√
class reverse_hierarchy_flow(nn.Module): 
  def __init__(self,
      input_channel,
      hidden_channel,
      input_length,
      scale_factor) -> None:
    super().__init__()
    self.input_channel = input_channel
    self.hidden_channel = hidden_channel
    self.input_length = input_length
    self.scale_factor = scale_factor
    self.half_channel = input_channel//2
    self.enc = nn.ModuleList([
      nn.Conv1d(self.input_channel,self.hidden_channel,4,1,3), 
      nn.Conv1d(self.input_channel,self.hidden_channel,4,1,3)
    ])
    self.proj = nn.Conv1d(self.hidden_channel, self.half_channels*2, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()
    self.proj_weight = nn.Parameter(torch.rand(self.half_channel,self.input_length,self.input_length*2-1))
    self.proj_bias = nn.Parameter(torch.rand(self.half_channel,1,self.input_length*2-1))
  def __init__(self) -> None:
    super().__init__()
  def forward(self, x, s, inverse=False):
    #x0:B,C/2,T
    x0,x1 = torch.split(x,[self.half_channels]*2,1)
    l = x1.shape[2]
    s = self.proj(self.enc(x0))
    s = torch.bmm(s.transpose(0, 1),self.proj_weight)+self.proj_bias
    s = s.tranpose(0,1)
    #s: B,C/2,2*T-1
    assert((l & (l-torch.tensor(1)) == 0) and l != 0)
    assert(s.shape[0] == 2*l-1)
    bit_l = l.bit_length()

    if inverse==True:
      now = 0
      for i in reversed(range(bit_l)):
        a = s[:,:,now:now+(1 << i)]
        now = now+(1 << i)
        a = a.repeat_interleave(l//(1 << i), dim=2)
        x = x-a
    else:
      now = 0
      for i in range(bit_l):
        a = s[:,:,now:now+(1 << i)]
        now = now+(1 << i)
        a = a.repeat_interleave(l//(1 << i), dim=2)
        x = x+a
    return x

#√
class Morton_encode(nn.Module):
  def forward(self, x):
    l = x.shape[2]**2
    bit_l = l.bit_length()
    ret = torch.zeros(x.shape[0],x.shape[1],l)
    for b in range(x.shape[0]):
      for c in range(x.shape[1]):
        for i in range(x.shape[2]):
          for j in range(x.shape[2]):
            ij=0
            for bit in range(bit_l):
              if bit%2==1:
                ij=ij+(((i>>(bit//2))&1)<<bit)
              else:
                ij=ij+(((j>>(bit//2))&1)<<bit)
            ret[b][c][ij]=x[b][c][i][j]
    return ret

#√
class Morton_decode(nn.Module):
  def forward(self, x):
    # print(x.shape)
    l = x.shape[2]
    bit_l = l.bit_length()
    ret = torch.zeros_like(x)
    ret=ret.view(x.shape[0],x.shape[1],int(math.sqrt(l)),int(math.sqrt(l)))
    for b in range(x.shape[0]):
      for c in range(x.shape[1]):
        for ij in range(l):
          i=0
          j=0
          for bit in range(bit_l):
            if bit % 2==1:
              i=i+(((ij>>bit)&1)<<(bit//2))
            else:
              j=j+(((ij>>bit)&1)<<(bit//2))
          ret[b][c][i][j]=x[b][c][ij]
    return ret

#√
class Upsample_flow(nn.Module): 
  def forward(self, x, s, scale_factor=4, inverse=False):
    l = x.shape[0]
    assert(s.shape[0] == l*scale_factor)
    if inverse==True:
      x=x-s
      downsampled = nn.AvgPool1d(scale_factor,stride=scale_factor)
      ret = downsampled
    else:
      upsampled = x.repeat_interleave(scale_factor)
      upsampled = upsampled+s
      ret = upsampled
    return ret
  
