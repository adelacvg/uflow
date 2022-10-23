import os
from subprocess import call
from tqdm import tqdm
from torch.utils.data import Dataset
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import random
from torchvision import transforms
import shutil


def extract_frames(src_path,target_path,fps=1):
  for video_name in (os.listdir(src_path)):
    video_path = src_path + video_name
    print(video_path)
    image_path = target_path+video_name.split('.mp4')[0]+'/'
    print(image_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    dest = image_path + video_name.split('.mp4')[0]+'-%05d.jpg'
    call(["ffmpeg", "-i", video_path, "-r", str(fps), dest])

class FFHQDataset(Dataset):
  def __init__(self,dataset_path,transform=None) -> None:
    super().__init__()
    self.root_path = dataset_path
    self.hq_imgs = sorted(os.listdir(dataset_path+'/hq'))
    self.lq_imgs = sorted(os.listdir(dataset_path+'/lq'))
    self.transform = transform
  def __len__(self):
    return len(self.hq_imgs)
  def __getitem__(self,idx):
    img_lq = io.imread(os.path.join(self.root_path+'/lq',self.lq_imgs[idx]))
    img_hq = io.imread(os.path.join(self.root_path+'/hq',self.hq_imgs[idx]))
    # print(img_hq)
    if self.transform:
      img_lq = self.transform(img_lq)
      img_hq = self.transform(img_hq)
    return img_lq,img_hq

class FangtanDataset(Dataset):
  def __init__(self, root_dir, transform=None, max_interval=60):
    super().__init__()
    self.image_lists=[]
    self.imgs_dirs=[]
    self.max_interval=max_interval
    self.transform = transform
    for root,dirs,files in os.walk(root_dir):
      for dir in dirs:
        self.imgs_dirs.append(os.path.join(root,dir))
        self.image_lists.append(sorted(os.listdir((os.path.join(root,dir)))))

  def __len__(self):
      return len(self.image_lists)

  def __getitem__(self, idx):
    image_list=self.image_lists[idx]
    frame1=random.randint(0,len(image_list)-2)
    frame2=random.randint(frame1, min(len(image_list)-1,frame1+self.max_interval))
    interval = frame2-frame1
    frame1 = io.imread(os.path.join(self.imgs_dirs[idx],image_list[frame1]))
    frame2 = io.imread(os.path.join(self.imgs_dirs[idx],image_list[frame2]))
    if self.transform:
      frame1 = self.transform(frame1)
      frame2 = self.transform(frame2)
    return frame1, frame2, interval


# extract_frames(src_path='/home/hyc/fangtan/train/',target_path='./data/fangtan/train/',fps=30)
# extract_frames(src_path='/home/hyc/fangtan/test/',target_path='./data/fangtan/test/',fps=30)

dataset = FFHQDataset('./data/ffhq',transform=transforms.Compose([transforms.ToTensor()]))
data=DataLoader(dataset, batch_size=2, shuffle=True)
data = [next(iter(data))]
for batch_num, input_dict in enumerate(data):
  frame1,frame2 = input_dict
  print(len(input_dict))
  # print(frame1.shape, frame2.shape)
  # print(frame1)

# dataset = FangtanDataset('./data/fangtan/train',transform=transforms.Compose([transforms.ToTensor()]))
# data=DataLoader(dataset, batch_size=2, shuffle=True)
# data = [next(iter(data))]
# for batch_num, (frame1, frame2,interval) in enumerate(data):
#   print(frame1.shape, frame2.shape)
#   print(frame1)