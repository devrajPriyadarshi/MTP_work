import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as tf
from torch.utils.data import DataLoader

from Metric import ChamferDistance, ProjectionLoss
from Network import Network
from DataLoaders import ShapeNetDataset

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

END_EPOCH = 100
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


    for _ , data in enumerate(TestLoader, 0):
        imgs, pcs = data
        imgs = imgs.to(device)
        pcs = pcs.to(device)
        res = net(imgs)
        loss = ValidateCriterion(pcs, res)
        TotalChamferLoss += loss.item()
        # print(f'[Test: {i + 1:5d}] loss: {loss.item():.7f}')

    return TotalChamferLoss

def training(net):
    
    net.to(device)
    net.train()
    bestScore = sys.maxsize
    bestEpoch = 0
    Loss = ChamferDistance(point_reduction="mean", batch_reduction="sum")

    optimizer = optim.Adam(net.parameters(), lr=LR)

    ValidationScoreArray = []
    TrainingScoreArray = []
    print("\nStart training..")
    for epoch in range(0, END_EPOCH):  
        
        net.train()
        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)
            loss = Loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.7f}')
                running_loss = 0.0
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%200 + 1):.7f}')

        currentScore = validator(net=net)
        print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
        print("")
        ValidationScoreArray.append(currentScore)
        TrainingScoreArray.append(TLoss)
        if currentScore < bestScore:
            bestScore = currentScore
            bestEpoch = epoch+1
            print("Saving Model...")
            torch.save(
                {   'epoch':epoch, 
                    'classes':CLASSES,
                    "lr":LR,
                    "batch_size":BATCH_SIZE,
                    "score":currentScore,
                    "num_views":NUM_VIEWS,
                    "num_heads":NUM_ENCODER_HEADS,
                    "num_layers":NUM_ENCODER_LAYERS,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                },
                'bestScore.pth')
            np.save( "ValidationScoreArr.npy", np.array(ValidationScoreArray))
            np.save( "TrainingScoreArr.npy", np.array(TrainingScoreArray))
    print("End training..")
    print("best Epoch = " + str(bestEpoch))
    print("best Score = " + str(bestScore))
        

def finetune(net: torch.nn.Module , modeldata_path: str):
    net.to(device)
    net.train()
    modeldata = torch.load(modeldata_path)

    bestScore = modeldata["score"]
    bestEpoch = modeldata["epoch"]
    net.load_state_dict(modeldata["model_state_dict"])
    
    Chamfer_loss = ChamferDistance(point_reduction="mean", batch_reduction="sum")
    Projection_loss = ProjectionLoss(rotations= [ [np.pi/2, 0, 0], [0, np.pi/2, 0], [0, 0, np.pi/2]], batch_reduction="mean")
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    optimizer.load_state_dict(modeldata["optimizer_state_dict"])

    ValidationScoreArray = np.array([])
    TrainingScoreArray = np.array([])
    ValidationScoreArray = np.concatenate((np.load("ValidationScoreArr.npy"), ValidationScoreArray))
    TrainingScoreArray = np.concatenate((np.load("TrainingScoreArr.npy"), TrainingScoreArray))
    

    print("\nfinetuning..")
    for epoch in range(bestEpoch, END_EPOCH):  
        
        net.train()
        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)

            loss = Chamfer_loss(pcs, res) + 0.01*Projection_loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.7f}')
                running_loss = 0.0
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%200 + 1):.7f}')

        currentScore = validator(net=net)
        print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
        print("")
        ValidationScoreArray.append(currentScore)
        TrainingScoreArray.append(TLoss)
        if currentScore < bestScore:
            bestScore = currentScore
            bestEpoch = epoch+1
            print("Saving Model...")
            torch.save(
                {   'epoch':epoch, 
                    "before_tune_epoch":bestEpoch,
                    "before_tune_score":modeldata["score"],
                    'classes':CLASSES,
                    "lr":LR,
                    "batch_size":BATCH_SIZE,
                    "score":currentScore,
                    "num_views":NUM_VIEWS,
                    "num_heads":NUM_ENCODER_HEADS,
                    "num_layers":NUM_ENCODER_LAYERS,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                },
                'bestScore_finetuned.pth')
            np.save( "ValidationScoreArr_finetuned.npy", np.array(ValidationScoreArray))
            np.save( "TrainingScoreArr_finetuned.npy", np.array(TrainingScoreArray))
    
    print("End training..")
    print("best Epoch = " + str(bestEpoch))
    print("best Score = " + str(bestScore))

def demo(net):
    
    net.to(device)
    net.train()
    bestScore = sys.maxsize
    bestEpoch = 0
    Loss = ChamferDistance(point_reduction="mean", batch_reduction="sum")

    optimizer = optim.Adam(net.parameters(), lr=LR)

    ValidationScoreArray = []
    TrainingScoreArray = []
    print("\nStart training..")
    for epoch in range(0, END_EPOCH):  
        
        net.train()
        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)
            loss = Loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')

        # currentScore = validator(net=net)
        # print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
        print("")
        # ValidationScoreArray.append(currentScore)
        # TrainingScoreArray.append(TLoss)
    
    print("End training..")
    # print("")
    # print("Training Score Array:")
    # print(TrainingScoreArray)
    # print("Validation Score Array:")
    # print(ValidationScoreArray)

if __name__ == "__main__":

    ShapeNetTrainData = ShapeNetDataset(classes=CLASSES, split="train", transforms=TRANSFORMS)
    TrainLoader = DataLoader(ShapeNetTrainData, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

    ShapeNetTestData = ShapeNetDataset(classes=CLASSES, split="val", transforms=TRANSFORMS)
    TestLoader = DataLoader(ShapeNetTestData, batch_size=BATCH_SIZE, num_workers=WORKERS)

    print("\n--------------- Training ---------------")

    print("\n")
    print("Classes to Train on: \t" + str(CLASSES))
    print("")
    print("Device: " + str(device))
  
    print("\nTrain Data Size: \t" + str(len(ShapeNetTrainData)))
    print("Batche Size: \t\t" + str(BATCH_SIZE))
    print("Batches / Epochs: \t" + str(len(TrainLoader)))
    print("Total Epochs: \t\t" + str(END_EPOCH))
    print("Total Iterations: \t" + str(END_EPOCH*len(TrainLoader)))

    print("\nTest Data Size : \t" + str(len(ShapeNetTestData)))
    print("Batches / Epochs: \t" + str(len(TestLoader)))


    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS)

    # training(net)
    finetune(net, "bestScore.pth")
    # validator(net)

    # time1 = time.time()
    # demo(net)
    # time2 = time.time()

    # print("Total Time: " + str(time2 - time1))