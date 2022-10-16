from math import sqrt
import numpy as np
import torch
from .morton_code.core import morton_encode_c,morton_decode_c



def morton_encode(img_batch):
  """
  img_batch: [b,c,w,h]
  """
  device = img_batch.device
  dtype = img_batch.dtype
  img_batch = img_batch.data.cpu().numpy().astype(np.float32)
  encoded_imgs = np.zeros((img_batch.shape[0],img_batch.shape[1],img_batch.shape[2]**2),dtype=np.float32)
  morton_encode_c(img_batch, encoded_imgs)

  return torch.from_numpy(encoded_imgs).to(device=device,dtype=dtype)

   
def morton_decode(img_batch):
  """
  img_batch: [b,c,w,h]
  """
  device = img_batch.device
  dtype = img_batch.dtype
  img_batch = img_batch.data.cpu().numpy().astype(np.float32)
  decoded_imgs = np.zeros((img_batch.shape[0],img_batch.shape[1],int(sqrt(img_batch.shape[2])),int(sqrt(img_batch.shape[2]))),dtype=np.float32)
  morton_decode_c(img_batch, decoded_imgs)

  return torch.from_numpy(decoded_imgs).to(device=device,dtype=dtype)
