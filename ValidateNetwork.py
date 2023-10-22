import numpy as np
from random import randint, sample
import torch
import torchvision.transforms as tf
from torch.utils.data import DataLoader

from Metric import ChamferDistance, ProjectionLoss
from Network import Network
from DataLoaders import ShapeNetDataset

import matplotlib.pyplot as plt

from Vizualization import ComparePointClouds, ImageFromTensor, ImageFromNumpy

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

BATCH_SIZE = 16
CLASSES = ["chair", "table"]
NUM_VIEWS = 24
NUM_ENCODER_LAYERS = 2
NUM_ENCODER_HEADS = 1

SHUFFLE = True
WORKERS = 6

END_EPOCH = 1
LR = 0.0001

TRANSFORMS = tf.Compose([   tf.ToTensor(),
                            tf.Resize((224,224), antialias=True),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

def validator(net):
    print("\nRunning Validator...")
    net.to(device)
    net.eval()
    TotalChamferLoss = 0.0
    ValidateCriterion = ChamferDistance(point_reduction="mean", batch_reduction="sum")

    for i , data in enumerate(TestLoader, 0):
        imgs, pcs = data
        imgs = imgs.to(device)
        pcs = pcs.to(device)
        res = net(imgs)
        loss = ValidateCriterion(pcs, res)
        TotalChamferLoss += loss.item()
        # print(f'[Test: {i + 1:5d}] loss: {loss.item():.7f}')

    return TotalChamferLoss

if __name__ == "__main__":

    model_folder = "Overnight_21/"

    ModelData = torch.load("Models/"+model_folder+"bestScore.pth")
    trainLoss = np.load("Models/"+model_folder+"TrainingScoreArr.npy")
    valLoss = np.load("Models/"+model_folder+"ValidationScoreArr.npy")
    assert len(trainLoss) == len(valLoss)

    classes = ModelData['classes']
    epochLable = np.arange(start = 1, stop=ModelData['epoch']+2)

    ShapeNetTrainData = ShapeNetDataset(classes=classes, split="train", transforms=TRANSFORMS)
    ShapeNetTestData = ShapeNetDataset(classes=classes, split="val", transforms=TRANSFORMS)
    TestLoader = DataLoader(ShapeNetTestData, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

    print("\n--------------- Validating ---------------")

    print("\nTest Data Size : \t" + str(len(ShapeNetTestData)))
    print("Batches / Epochs: \t" + str(len(TestLoader)))


    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS)
    net.load_state_dict(ModelData["model_state_dict"])

    net = net.to(device)
    net.eval()
    # LOSS = validator(net)

    # print("Model Saved Loss: " + str(ModelData["score"]))
    # print("Calculated Total Loss: " + str(LOSS))

    # CD_Loss = ChamferDistance(point_reduction="mean", batch_reduction="sum")
    # PJ_Loss = ProjectionLoss(rotations=[ [np.pi/2, 0, 0], [0, np.pi/2, 0], [0, 0, np.pi/2]], batch_reduction="mean")

    # img, pc = ShapeNetTestData[randint(0, len(TestLoader)-1)]
    # img = img.unsqueeze(0).to(device)
    # pc = pc.to(device)
    # res = net(img)
    # print("CD loss: " + str(CD_Loss(pc.unsqueeze(0), res).item()))
    # print("PJ loss: " + str(PJ_Loss(pc.unsqueeze(0), res).item()))
    # plot = ComparePointClouds(pc, res.squeeze(0))
    # plot.show()

    res_folder = "trial1/"

    print("Saving Figures")

    plt.figure()
    plt.plot(epochLable, trainLoss/len(ShapeNetTrainData), label = "traning loss")
    plt.plot(epochLable, valLoss/len(ShapeNetTestData), label = "validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Chamfer Distance Error")
    plt.legend()
    plt.savefig("Results/"+res_folder+"history.png")
    for x in sample(range(0, len(TestLoader)), k = 10):

        img, pc = ShapeNetTestData[x]
        img = img.unsqueeze(0).to(device)
        pc = pc.to(device)
        res = net(img) 

        img1 = ImageFromTensor(pc)
        img2 = ImageFromTensor(res.squeeze(0))

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.imshow(img1.clip(min=0, max=1))
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(img2.clip(min=0, max=1))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("Results/"+res_folder+"sample_%d.png" % x)