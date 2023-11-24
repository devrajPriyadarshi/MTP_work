import numpy as np
from random import randint, sample
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from DataLoaders import ShapeNetDataset

import matplotlib.pyplot as plt

from Vizualization import ImageFromTensor
from Metric import ProjectionLoss

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    # device = torch.device("cpu")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == "__main__":

    CLASSES = ['chair', 'table']
    # ShapeNetTrainData = ShapeNetDataset(classes=CLASSES, split="train80", transforms=None, num_views=1)
    # ShapeNetValData = ShapeNetDataset(classes=CLASSES, split="val10", transforms=None, num_views=1)
    # ShapeNetTestData = ShapeNetDataset(classes=CLASSES, split="test10", transforms=None, num_views=1)

    ChairData = ShapeNetDataset(classes=["chair"], split="test10", transforms=None, num_views=1)
    TableData = ShapeNetDataset(classes=["table"], split="test10", transforms=None, num_views=1)


    print("\n--------------- Getting Samples ---------------")

    # print("\nTrain Data Size : \t" + str(len(ShapeNetTrainData)))
    # print("\nTest Data Size : \t" + str(len(ShapeNetTestData)))
    # print("\nVal Data Size : \t" + str(len(ShapeNetValData)))
    print("\nChair Test Data Size : \t" + str(len(ChairData)))
    print("\nTable Test Data Size : \t" + str(len(TableData)))
    print("")
    print("Classes: ", CLASSES)
    print("")

    projLoss = ProjectionLoss(rotations= [ [np.pi/2, 0, 0], [0, np.pi/2, 0], [0, 0, np.pi/2]],
                                     batch_reduction="mean",
                                     view_reduction="sum",
                                     batch_size = 1)

    for idx in sample(range(0, len(ChairData)), k = 10):
        img, pc = ChairData[idx]
        img = img.squeeze()
        pc_img = 255*ImageFromTensor(pc)
        proj1, proj2, proj3 = projLoss.getProjection(pc.unsqueeze(0).to(device))
        proj_arr1 = 255*np.resize(proj1.detach().squeeze().cpu().numpy(), (64, 64))
        proj_arr2 = 255*np.resize(proj2.detach().squeeze().cpu().numpy(), (64, 64))
        proj_arr3 = 255*np.resize(proj3.detach().squeeze().cpu().numpy(), (64, 64))
        
        cv2.imwrite("dataset_samples/chair_sample/chair_"+str(idx)+"_pc.png", pc_img)
        cv2.imwrite("dataset_samples/chair_sample/chair_"+str(idx)+"_img.png", img)
        cv2.imwrite("dataset_samples/chair_sample/chair_"+str(idx)+"_pc1.png", proj_arr1)
        cv2.imwrite("dataset_samples/chair_sample/chair_"+str(idx)+"_pc2.png", proj_arr2)
        cv2.imwrite("dataset_samples/chair_sample/chair_"+str(idx)+"_pc3.png", proj_arr3)

    for idx in sample(range(0, len(TableData)), k = 10):
        img, pc = TableData[idx]
        img = img.squeeze()
        pc_img = 255*ImageFromTensor(pc)
        proj1, proj2, proj3 = projLoss.getProjection(pc.unsqueeze(0).to(device))
        proj_arr1 = 255*np.resize(proj1.detach().squeeze().cpu().numpy(), (64, 64))
        proj_arr2 = 255*np.resize(proj2.detach().squeeze().cpu().numpy(), (64, 64))
        proj_arr3 = 255*np.resize(proj3.detach().squeeze().cpu().numpy(), (64, 64))
        
        cv2.imwrite("dataset_samples/table_sample/table_"+str(idx)+"_pc.png", pc_img)
        cv2.imwrite("dataset_samples/table_sample/table_"+str(idx)+"_img.png", img)
        cv2.imwrite("dataset_samples/table_sample/table_"+str(idx)+"_pc1.png", proj_arr1)
        cv2.imwrite("dataset_samples/table_sample/table_"+str(idx)+"_pc2.png", proj_arr2)
        cv2.imwrite("dataset_samples/table_sample/table_"+str(idx)+"_pc3.png", proj_arr3)