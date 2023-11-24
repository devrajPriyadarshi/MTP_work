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

from Vizualization import ImageFromTensor

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    # device = torch.device("cpu")
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
    print("\n--------------- Multi VIew Test ---------------")

    NUM_VIEWS = ModelData["num_views"]
    NUM_ENCODER_LAYERS = ModelData["num_layers"]
    NUM_ENCODER_HEADS = ModelData["num_heads"]

    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS)
    net.load_state_dict(ModelData["model_state_dict"])

    net = net.to(device)
    net.eval()

    chair_img_arr = []
    table_img_arr = []
    chair_tensor_arr = []
    table_tensor_arr = []
    chair_predicted_arr = []
    table_predicted_arr = []
    chair_predicted_imgs = []
    table_predicted_imgs = []

    for v in [1, 3, 6, 12, 24]:
        chair_img_Tensor = np.array([Image.open("tests/chair_sample/rendering/"+ view).convert('RGB') for view in sample(views, v)])
        chair_img_arr.append(chair_img_Tensor)
        chair_img_Tensor = torch.from_numpy(np.array([TRANSFORMS(im) for im in chair_img_Tensor]))
        chair_tensor_arr.append(chair_img_Tensor)
    
    for v in [1, 3, 6, 12, 24]:
        table_img_Tensor = np.array([Image.open("tests/table_sample/rendering/"+ view).convert('RGB') for view in sample(views, v)])
        table_img_arr.append(table_img_Tensor)
        table_img_Tensor = torch.from_numpy(np.array([TRANSFORMS(im) for im in table_img_Tensor]))
        table_tensor_arr.append(table_img_Tensor)
    
    for v in range(len([1, 3, 6, 12, 24])):
        chair_img_features = net.encoder(chair_tensor_arr[v].unsqueeze(0).to(device))
        chair_predicted_pc = torch.transpose(net.decoder(chair_img_features), 1, 2)
        chair_predicted_arr.append(chair_predicted_pc)

    for v in range(len([1, 3, 6, 12, 24])):
        table_img_features = net.encoder(table_tensor_arr[v].unsqueeze(0).to(device))
        table_predicted_pc = torch.transpose(net.decoder(table_img_features), 1, 2)
        table_predicted_arr.append(table_predicted_pc)

    for v in range(len([1, 3, 6, 12, 24])):
        img = ImageFromTensor(chair_predicted_arr[v])
        chair_predicted_imgs.append(img)

    for v in range(len([1, 3, 6, 12, 24])):
        img = ImageFromTensor(table_predicted_arr[v])
        table_predicted_imgs.append(img)

    print("\nDone!")
    print("Saving Figures...")

    vi = -1
    for v in [1,3,6,12,24]:
        vi+=1
        left = []
        if v == 1:
            y = np.array([['left1']])
            y_name = ['left1']
        else:
            y = np.array(['left1','left2','left3','left4','left5','left6','left7','left8','left9','left10','left11','left12',
                 'left13','left14','left15','left16','left17','left18','left19','left20','left21','left22','left23','left24'])
            y_name = y[:v].tolist()
            y = np.reshape(y[:v], (v//3, 3))
        y = y.tolist()
        m = [[y, 'right']]
        fig, axs = plt.subplot_mosaic(m)
        for i, ax in enumerate(y_name, 0):
            axs[ax].imshow(chair_img_arr[vi][i])
            axs[ax].axis("off")

        axs['right'].imshow(chair_predicted_imgs[vi])
        axs['right'].axis("off")
        plt.tight_layout()
        plt.savefig("Results/" + model_folder + "/chair_"+str(v)+"_view_result.png")
        # plt.show()
    
    vi = -1
    for v in [1,3,6,12,24]:
        vi+=1
        left = []
        if v == 1:
            y = np.array([['left1']])
            y_name = ['left1']
        else:
            y = np.array(['left1','left2','left3','left4','left5','left6','left7','left8','left9','left10','left11','left12',
                 'left13','left14','left15','left16','left17','left18','left19','left20','left21','left22','left23','left24'])
            y_name = y[:v].tolist()
            y = np.reshape(y[:v], (v//3, 3))
        y = y.tolist()
        m = [[y, 'right']]
        fig, axs = plt.subplot_mosaic(m)
        for i, ax in enumerate(y_name, 0):
            axs[ax].imshow(table_img_arr[vi][i])
            axs[ax].axis("off")

        axs['right'].imshow(table_predicted_imgs[vi])
        axs['right'].axis("off")
        plt.tight_layout()
        plt.savefig("Results/" + model_folder + "/table_"+str(v)+"_view_result.png")
