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
import shutil


def extract_frames(src_path,target_path,fps=1):
  for video_name in (os.listdir(src_path)):
      video_path = src_path + video_name
      print(video_path)
      image_path = target_path+video_name.split('.mp4')[0]+'/'
      print(image_path)
      if not os.path.exists(image_path):
          os.makedirs(image_path)
      dest = image_path + video_name.split('.mp4')[0]+'-%08d.jpg'
      call(["ffmpeg", "-i", video_path, "-r", str(fps), dest])



def read_video(name, frame_shape):
  if os.path.isdir(name):
      frames = sorted(os.listdir(name))
      num_frames = len(frames)
      video_array = np.array(
          [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
  elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
      image = io.imread(name)

      if len(image.shape) == 2 or image.shape[2] == 1:
          image = gray2rgb(image)

      if image.shape[2] == 4:
          image = image[..., :3]

      image = img_as_float32(image)

      video_array = np.moveaxis(image, 1, 0)

      video_array = video_array.reshape((-1,) + frame_shape)
      video_array = np.moveaxis(video_array, 1, 2)
  elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
      video = np.array(mimread(name))
      if len(video.shape) == 3:
          video = np.array([gray2rgb(frame) for frame in video])
      if video.shape[-1] == 4:
          video = video[..., :3]
      video_array = img_as_float32(video)
  else:
      raise Exception("Unknown file extensions  %s" % name)

  return video_array

class TwoFramesDataset(Dataset):
  def __init__(self, root_dir, frame_shape=(256, 256, 3),random_seed=0, is_train=True, max_interval=60):
      self.image_lists=[]
      self.max_interval=max_interval
      for root,dirs,files in os.walk(root_dir):
          for dir in dirs:
              self.image_lists.append(read_video(os.path.join(root,dir), frame_shape))
      train_videos, test_videos = train_test_split(self.image_lists, random_state=random_seed, test_size=0.2)
      if is_train:
          self.image_lists=train_videos
      else:
          self.image_lists=test_videos

  def __len__(self):
      return len(self.image_lists)

  def __getitem__(self, idx):
      image_list=self.image_lists[idx]
      frame1=random.randint(0,len(image_list)-2)
      frame2=0
      if frame1+self.max_interval>len(image_list)-1:
          frame2=random.randint(frame1+1, len(image_list)-1)
      return image_list[frame1], image_list[frame2], frame2-frame1


if __name__ == '__main__':
  extract_frames(src_path='./test_data/',target_path='./test_data/',fps=30)
  dataset=TwoFramesDataset('./test_data/')
  data=DataLoader(dataset, batch_size=2, shuffle=True)
  for batch_num, (frame1, frame2, interval) in enumerate(data):
      print(frame1.shape, frame2.shape, interval)
    

class FFHQDataset(Dataset):
  def __init__(self) -> None:
      super().__init__()
  def __len__(self):
      
  def __getitem__(self,idx):


class FangtanDataset(Dataset):
  def __init__(self) -> None:
      super().__init__()
  def __len__(self):
  
  def __getitem__(self,idx):