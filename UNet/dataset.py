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
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,self.image[index].replace(".jpg","_mask.gif"))
        image = np.array(Image.open())
        mask = np.asarray(image)
        mask[mask == 255.0] = 1.0
        if self.transf is not None:
            augmentation = self.transf(image=image,mask = mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        return image, mask
