# Need function to rotate the Point Cloud
# Maybe to add noise to them -> DONT
# Maybe to enchance the image

# import cv2
import numpy as np
from numpy import ndarray

import torch
import torchvision.transforms as TF

from Vizualization import euler_angles_to_matrix

def fixPointcloudOrientation(pointcloud: torch.Tensor) -> torch.Tensor:
    pc_numpy = pointcloud.cpu().numpy()
    rot = euler_angles_to_matrix(torch.Tensor([0,0,np.pi/2]), "XYZ")
    center = np.array([ (np.min(pc_numpy[:,0]) + np.max(pc_numpy[:,0])) / 2 ,
                        (np.min(pc_numpy[:,1]) + np.max(pc_numpy[:,1])) / 2 ,
                        (np.min(pc_numpy[:,2]) + np.max(pc_numpy[:,2])) / 2 ])
    verts = torch.Tensor(pc_numpy - center)
    return verts@rot

# def cv2ToTensor(img: ndarray) -> torch.Tensor:
#     return TF.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))