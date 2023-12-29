
from PIL import Image
import json
import numpy as np
from random import sample

import torch
import torchvision.transforms as tf
from torch.utils.data import Dataset

from shapenet_taxonomy import shapenet_category_to_id
from Preprocess import fixPointcloudOrientation

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
        img = [Image.open(self.root+"ShapeNetRendering/"+self.files[index]+"/rendering/"+ view).convert('RGB') for view in sample(views, self.num_views)]
        if self.transforms != None:
            img = [self.transforms(im) for im in img]
            img_tensor = torch.stack(img, dim = 0)
        else:
            toTen = tf.ToTensor()
            img = [toTen(im) for im in img]
            img_tensor = torch.stack(img, dim = 0)
            
        return img_tensor, gt_pc

# TRANSFORMS = tf.Compose([   tf.ToTensor(),
#                             tf.Resize((224,224), antialias=True),
#                             tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                         ])

# data = ShapeNetDataset(classes=["chair"], transforms= TRANSFORMS)
# im, pc = data[0]
# print(pc)
# print(im)
# print(pc.shape)
# print(im.shape)

