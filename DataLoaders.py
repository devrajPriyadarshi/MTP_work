
from PIL import Image
import json
import numpy as np
from random import sample

import torch
from torch.utils.data import Dataset

from utils.shapenet_taxonomy import shapenet_category_to_id
from Preprocess import fixPointcloudOrientation, cv2ToTensor

views = [str(i).zfill(2)+".png" for i in range(24)]

class ShapeNetDataset(Dataset):
    def __init__(self, root = "data/ShapeNet/", split = "train", classes = [], num_views = 24, transforms = None):
        
        assert classes != []

        self.root = root
        self.split = split
        self.num_views = num_views
        self.transforms = transforms

        self.classID = [ shapenet_category_to_id[idx] for idx in classes]
        self.files = []

        with open(root+"splits/"+split+"_models.json") as f:
            self.train_split = json.load(f)
            for classes in self.classID:
                self.files += self.train_split[classes]
        

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        gt_pc = fixPointcloudOrientation(torch.from_numpy(np.load(self.root+"ShapeNet_pointclouds/"+self.files[index]+"/pointcloud_1024.npy")))
        img = np.array([ Image.open(self.root+"ShapeNetRendering/"+self.files[index]+"/rendering/"+ view).convert('RGB') for view in sample(views, self.num_views)])
        
        if self.transforms != None:
            img = torch.from_numpy(np.array([self.transforms(im) for im in img]))
            
        return img, gt_pc

# data = ShapeNetDataset(classes=["chair"])
# pc, im = data[0]
# print(pc.shape)
# print(im.shape)

