import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transf = transform
        self.img = os.listdir(image_dir)
    def __len__(self):
        return len(self.img)
    def __getitem__(self,index):
        im
