import numpy as np
from random import randint, sample
import cv2
from PIL import Image
import torch
import torchvision.transforms as tf
from torch.utils.data import DataLoader

from Network import Network
from DataLoaders import ShapeNetDataset

import matplotlib.pyplot as plt

from Preprocess import fixPointcloudOrientation, cv2ToTensor
from Vizualization import ComparePointClouds, ImageFromTensor, ImageFromNumpy

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

views = [str(i).zfill(2)+".png" for i in range(24)]

TRANSFORMS = tf.Compose([   tf.ToTensor(),
                            tf.Resize((224,224), antialias=True),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

if __name__ == "__main__":

    model_folder = "10_24_03_49"

    ModelData = torch.load("Models/"+model_folder+"/bestScore.pth")
    trainLoss = np.load("Models/"+model_folder+"/TrainingScoreArr.npy")
    valLoss = np.load("Models/"+model_folder+"/ValidationScoreArr.npy")
    assert len(trainLoss) == len(valLoss)

    classes = ModelData['classes']
    print("\n--------------- Intertpolating ---------------")

    NUM_VIEWS = ModelData["num_views"]
    NUM_ENCODER_LAYERS = ModelData["num_layers"]
    NUM_ENCODER_HEADS = ModelData["num_heads"]

    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS)
    net.load_state_dict(ModelData["model_state_dict"])

    net = net.to(device)
    net.eval()

    chair_pc_Tensor = fixPointcloudOrientation(torch.Tensor(np.load("tests/chair_sample/pointcloud_1024.npy")))
    chair_img_Tensor = np.array([Image.open("tests/chair_sample/rendering/"+ view).convert('RGB') for view in sample(views, NUM_VIEWS)])
    chair_img_Tensor = torch.from_numpy(np.array([TRANSFORMS(im) for im in chair_img_Tensor]))
    
    table_pc_Tensor = fixPointcloudOrientation(torch.Tensor(np.load("tests/table_sample/pointcloud_1024.npy")))
    table_img_Tensor = np.array([Image.open("tests/table_sample/rendering/"+ view).convert('RGB') for view in sample(views, NUM_VIEWS)])
    table_img_Tensor = torch.from_numpy(np.array([TRANSFORMS(im) for im in table_img_Tensor]))
    
    chair_img_features = net.encoder(chair_img_Tensor.unsqueeze(0).to(device))
    chair_predicted_pc = torch.transpose(net.decoder(chair_img_features), 1, 2)

    table_img_features = net.encoder(table_img_Tensor.unsqueeze(0).to(device))
    table_predicted_pc = torch.transpose(net.decoder(table_img_features), 1, 2)

    # img_chair_predicted_pc = ImageFromTensor(chair_predicted_pc.squeeze(0))
    # img_table_predicted_pc = ImageFromTensor(table_predicted_pc.squeeze(0))
    # img_chair_gt_pc = ImageFromTensor(chair_pc_Tensor)
    # img_table_gt_pc = ImageFromTensor(table_pc_Tensor)

    # print("\nDone!")

    # plt.figure(figsize=(5,5))
    # plt.subplot(221)
    # plt.imshow(img_chair_gt_pc)
    # plt.axis("off")
    # plt.subplot(222)
    # plt.imshow(img_table_gt_pc)
    # plt.axis("off")
    # plt.subplot(223)
    # plt.imshow(img_chair_predicted_pc)
    # plt.axis("off")
    # plt.subplot(224)
    # plt.imshow(img_table_predicted_pc)
    # plt.axis("off")
    # plt.show()

    weights = list(range(0,11))
    interpolate_tensor = torch.zeros((len(weights), 1000))
    for x in weights:
        interpolate_tensor[x] = torch.lerp(chair_img_features, table_img_features, x/10).squeeze(0)
    
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)

    interpolated_imgs = []
    for i in weights:
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)

    print("\nDone!")

    fig, axs = plt.subplots(nrows=1, ncols=11, figsize=(11*5, 5))
    for i in weights:
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/interpolation_result.png")
    # plt.show()