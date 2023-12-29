import json
from random import choice, randint

# import cv2
import numpy as np
# import numpy.typing as npt
import matplotlib.pyplot as plt

from DataLoaders import ShapeNetDataset
import torch
# from pytorch3d.structures import Pointclouds
# from pytorch3d.transforms import euler_angles_to_matrix
# from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointsRasterizationSettings, PointsRenderer, PointsRasterizer, AlphaCompositor

from shapenet_taxonomy import shapenet_category_to_id as ID
from MitsubaRendering import ImageFromNumpyArr

pc_folder = "data/ShapeNet/ShapeNet_pointclouds/"
rend_folder = "data/ShapeNet/ShapeNetRendering/"

json_file = open("data/ShapeNet/splits/train_models.json")
train_split = json.load(json_file)
json_file.close()

classes = [ ID['car'], ID['chair'] ]

def ImageFromTensor(pc: torch.Tensor):
    pcl = pc.detach().cpu().numpy()
    img = ImageFromNumpyArr(pcl)
    return img.clip(min=0.0, max=1.0)

def ImageFromNumpy(np_arr):
    img = ImageFromNumpyArr(np_arr)
    return img

if __name__ == "__main__":

    data = ShapeNetDataset(classes = ["chair"])
    img, pc = data[0]
    img = ImageFromTensor(pc)
    print(pc.shape)
    plt.figure(figsize=(5,5))
    plt.imshow(img.clip(min=0, max=1))
    plt.axis("off")
    plt.tight_layout()
    plt.show()