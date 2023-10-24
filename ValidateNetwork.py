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

TRANSFORMS = tf.Compose([   tf.ToTensor(),
                            tf.Resize((224,224), antialias=True),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

def validator(net):
    print("\nRunning Validator...")
    net.to(device)
    net.eval()
    TotalChamferLoss = 0.0
    ValidateCriterion = ChamferDistance(point_reduction="sum", batch_reduction="mean")

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

    model_folder = "10_24_03_49"

    ModelData = torch.load("Models/"+model_folder+"/bestScore.pth")
    trainLoss = np.load("Models/"+model_folder+"/TrainingScoreArr.npy")
    valLoss = np.load("Models/"+model_folder+"/ValidationScoreArr.npy")
    assert len(trainLoss) == len(valLoss)

    BATCH_SIZE = ModelData["batch_size"]
    CLASSES = ModelData["classes"]
    NUM_VIEWS = ModelData["num_views"]
    NUM_ENCODER_LAYERS = ModelData["num_layers"]
    NUM_ENCODER_HEADS = ModelData["num_heads"]

    SHUFFLE = True
    WORKERS = 6

    END_EPOCH = ModelData['epoch']+1
    LR = ModelData["lr"]
    epochLable = np.arange(1,END_EPOCH+1)

    ShapeNetTrainData = ShapeNetDataset(classes=CLASSES, split="train", transforms=TRANSFORMS)
    ShapeNetTestData = ShapeNetDataset(classes=CLASSES, split="val", transforms=TRANSFORMS)
    TestLoader = DataLoader(ShapeNetTestData, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

    ShapeNetTestChairData = ShapeNetDataset(classes=["chair"], split="val", transforms=TRANSFORMS)
    ShapeNetTestTableData = ShapeNetDataset(classes=["table"], split="val", transforms=TRANSFORMS)

    print("\n--------------- Validating ---------------")

    print("\nTest Data Size : \t" + str(len(ShapeNetTestData)))
    print("Batches / Epochs: \t" + str(len(TestLoader)))
    print("")
    print("Classes: ", CLASSES)
    print("Batch Size: ", BATCH_SIZE)
    print("Learning Rate: ", LR)
    print("End Epoch: ", END_EPOCH)
    print("")
    print("Encoder Heads: ", NUM_ENCODER_HEADS)
    print("Encoder Layers: ", NUM_ENCODER_LAYERS)
    print("Number of Views trained on: ", NUM_VIEWS)

    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS)
    net.load_state_dict(ModelData["model_state_dict"])

    net = net.to(device)
    net.eval()
    # LOSS = validator(net)

    # print("Model Saved Loss: " + str(ModelData["score"]))
    # print("Calculated Total Loss: " + str(LOSS))

    # CD_Loss = ChamferDistance(point_reduction="sum", batch_reduction="mean")
    # PJ_Loss = ProjectionLoss(rotations=[ [np.pi/2, 0, 0], [0, np.pi/2, 0], [0, 0, np.pi/2]], batch_reduction="mean")
    # print("CD loss: " + str(CD_Loss(pc.unsqueeze(0), res).item()))
    # print("PJ loss: " + str(PJ_Loss(pc.unsqueeze(0), res).item()))

    print("\nSaving Figures..")

    plt.figure()
    plt.plot(epochLable, trainLoss/(len(ShapeNetTrainData)*1024), label = "traning loss")
    plt.plot(epochLable, valLoss/(len(ShapeNetTestData)*1024), label = "validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Chamfer Distance Error")
    plt.legend()
    plt.savefig("Results/"+model_folder+"/history.png")

    for x in sample(range(0, len(ShapeNetTestChairData)), k = 5):
        img, pc = ShapeNetTestChairData[x]
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
        plt.savefig("Results/"+model_folder+"/chair_sample_%d.png" % x)

    for x in sample(range(0, len(ShapeNetTestTableData)), k = 5):
        img, pc = ShapeNetTestTableData[x]
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
        plt.savefig("Results/"+model_folder+"/table_sample_%d.png" % x)