
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as tf
from torch.utils.data import DataLoader

from Metric0 import ChamferDistance, ProjectionLoss
from Network import Network
from DataLoaders import ShapeNetDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

dt = datetime.now()
FOLDER_NAME =  dt.strftime("%m_%d_%H_%M")

BATCH_SIZE = 16
CLASSES = ["chair", "table"]
NUM_VIEWS = 24
NUM_ENCODER_LAYERS = 2
NUM_ENCODER_HEADS = 4

SHUFFLE = True
WORKERS = 6

END_EPOCH = 30
LR = 0.0001

TRANSFORMS = tf.Compose([   tf.ToTensor(),
                            tf.Resize((224,224), antialias=True),
                            # tf.Resize((224,224)),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])


Chamfer_loss = ChamferDistance(point_reduction="sum", batch_reduction="mean")

def validator():
    print("\nRunning Validator...")

    net.eval()
    TotalChamferLoss = 0.0
    # ValidateCriterion = ChamferDistance(point_reduction="sum", batch_reduction="mean")
    # running_loss = 0.0
    with torch.no_grad():
        for _i , data in enumerate(TestLoader, 0):
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)
            loss = Chamfer_loss(pcs, res)
            TotalChamferLoss += loss.item()

    return TotalChamferLoss

def training():

    net.train()
    bestScore = sys.maxsize
    bestEpoch = 0
    # Loss = ChamferDistance(point_reduction="sum", batch_reduction="mean")

    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    ValidationScoreArray = []
    TrainingScoreArray = []
    print("\nStart training..")
    for epoch in range(0, END_EPOCH):
        print("Epoch "+str(epoch+1)+" Running..")
        
        f = open("Models/"+FOLDER_NAME+"/log.txt", 'a')
        towrite = "Epoch "+str(epoch+1)+" Running.."
        f.write(towrite+'\n')
        f.close()

        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)
            loss = Chamfer_loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}')
                
                f = open("Models/"+FOLDER_NAME+"/log.txt", 'a')
                towrite = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}'
                f.write(towrite+'\n')
                f.close()

                running_loss = 0.0
        scheduler.step()
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}')

        f = open("Models/"+FOLDER_NAME+"/log.txt", 'a')
        towrite = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}'
        f.write(towrite+'\n')
        f.close()
        
        currentScore = validator()
        print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
        print("")

        f = open("Models/"+FOLDER_NAME+"/log.txt", 'a')
        towrite = "At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore)
        f.write(towrite+'\n\n')
        f.close()

        ValidationScoreArray.append(currentScore)
        TrainingScoreArray.append(TLoss)
        if currentScore < bestScore:
            bestScore = currentScore
            # bestEpoch = epoch+1
            print("Saving Model at Epoch "+str(epoch+1)+"...")

            f = open("Models/"+FOLDER_NAME+"/log.txt", 'a')
            towrite = "Saving Model at Epoch "+str(epoch+1)+"..."
            f.write(towrite+'\n')
            f.close()

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
                "Models/"+FOLDER_NAME+'/bestScore.pth')
            np.save( "Models/"+FOLDER_NAME+"/ValidationScoreArr.npy", np.array(ValidationScoreArray))
            np.save( "Models/"+FOLDER_NAME+"/TrainingScoreArr.npy", np.array(TrainingScoreArray))
    
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
            "Models/"+FOLDER_NAME+'/endScore.pth')
    
    print("End training..")
    print("best Epoch = " + str(bestEpoch))
    print("best Score = " + str(bestScore))
    f = open("Models/"+FOLDER_NAME+"/log.txt", 'a')
    towrite = "End training..\n"+"best Epoch = " + str(bestEpoch)+"\n"+"best Score = " + str(bestScore)
    f.write(towrite+'\n')
    f.close()
        

def finetune(model_folder: str):

    modeldata = torch.load("Models/"+model_folder+"/bestScore.pth", map_location=device)

    bestScore = modeldata["score"]
    # bestScore = sys.maxsize
    bestEpoch = modeldata["epoch"]

    bs = bestScore

    print("")
    print("Best score: ", bestScore)
    print("Last Epoch: ", bestEpoch+1)
    net.load_state_dict(modeldata["model_state_dict"])

    _lr = 1e-4
    _alpha = 1
    _beta = 1
    _end_epoch = modeldata["epoch"] + 50

    # Chamfer_loss = ChamferDistance(point_reduction="sum", batch_reduction="mean")
    Projection_loss = ProjectionLoss(rotations= [[np.pi/2, 0, 0], [0, np.pi/2, 0], [0, 0, np.pi/2]],
                                     batch_reduction="mean",
                                     view_reduction="sum",
                                     batch_size = BATCH_SIZE)
    
    optimizer = optim.Adam(net.parameters(), lr=_lr)
    # optimizer.load_state_dict(modeldata["optimizer_state_dict"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    print("\n finetuning network..")
    print("\nTraining on LR: ", _lr)

    f = open("Models/"+model_folder+"/log.txt", 'a')
    towrite = "\n finetuning network.." + "\nTraining on LR: " + str(_lr)
    f.write(towrite+'\n')
    f.close()

    ValidationScoreArray = np.array([])
    TrainingScoreArray = np.array([])
    ValidationScoreArray = np.concatenate((np.load("Models/"+model_folder+"/ValidationScoreArr.npy"), ValidationScoreArray))
    TrainingScoreArray = np.concatenate((np.load("Models/"+model_folder+"/TrainingScoreArr.npy"), TrainingScoreArray))
    
    st_e = modeldata["epoch"]
    # print("\nfinetuning..")
    del modeldata
    for epoch in range(st_e, _end_epoch):  
        print("Epoch "+str(epoch+1)+" Running..")

        f = open("Models/"+model_folder+"/log.txt", 'a')
        towrite = "Epoch "+str(epoch+1)+" Running.."
        f.write(towrite+'\n')
        f.close()

        net.train()
        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)

            loss = _alpha*Chamfer_loss(pcs, res) + _beta*Projection_loss(pcs, res)
            # loss = Chamfer_loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()

            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}')
                
                f = open("Models/"+model_folder+"/log.txt", 'a')
                towrite = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}'
                f.write(towrite+'\n')
                f.close()

                running_loss = 0.0
        scheduler.step()
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}')
        
        f = open("Models/"+model_folder+"/log.txt", 'a')
        towrite = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}'
        f.write(towrite+'\n')
        f.close()
        
        currentScore = validator()
        print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
        print("")

        f = open("Models/"+model_folder+"/log.txt", 'a')
        towrite = "At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore)
        f.write(towrite+'\n\n')
        f.close()

        ValidationScoreArray = np.append(ValidationScoreArray, currentScore)
        TrainingScoreArray = np.append(TrainingScoreArray, TLoss)
        # print(TrainingScoreArray)
        if currentScore < bestScore:
            bestScore = currentScore
            bestEpoch = epoch+1
            print("Saving Model at Epoch "+str(epoch+1)+"...")
            
            f = open("Models/"+model_folder+"/log.txt", 'a')
            towrite = "Saving Model at Epoch "+str(epoch+1)+"..."
            f.write(towrite+'\n')
            f.close()

            torch.save(
                {   'epoch':epoch, 
                    "before_tune_epoch":st_e,
                    "before_tune_score":bs,
                    'classes':CLASSES,
                    "lr":_lr,
                    "batch_size":BATCH_SIZE,
                    "score":currentScore,
                    "num_views":NUM_VIEWS,
                    "num_heads":NUM_ENCODER_HEADS,
                    "num_layers":NUM_ENCODER_LAYERS,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                },
                "Models/"+model_folder+'/bestScore_finetuned.pth')
            np.save( "Models/"+model_folder+"/ValidationScoreArr_finetuned.npy", np.array(ValidationScoreArray))
            np.save( "Models/"+model_folder+"/TrainingScoreArr_finetuned.npy", np.array(TrainingScoreArray))
    
    print("End training..")
    print("best Epoch = " + str(bestEpoch))
    print("best Score = " + str(bestScore))
    f = open("Models/"+model_folder+"/log.txt", 'a')
    towrite = "End training..\n"+"best Epoch = " + str(bestEpoch)+"\n"+"best Score = " + str(bestScore)
    f.write(towrite+'\n')
    f.close()

def continue_training(model_folder: str):

    net.train()
    modeldata = torch.load("Models/"+model_folder+"/bestScore.pth")

    bestScore = modeldata["score"]
    # bestScore = sys.maxsize
    bestEpoch = modeldata["epoch"]

    print("")
    print("Best score: ", bestScore)
    print("Last Epoch: ", bestEpoch+1)
    net.load_state_dict(modeldata["model_state_dict"])
    
    # Chamfer_loss = ChamferDistance(point_reduction="sum", batch_reduction="mean")    
    _lr = 1e-7
    _end_epoch = modeldata["epoch"] + 100
    optimizer = optim.Adam(net.parameters(), lr=_lr)
    # optimizer.load_state_dict(modeldata["optimizer_state_dict"])

    print("\n continued training..")
    print("\nTraining on LR: ", _lr)

    f = open("Models/"+model_folder+"/log.txt", 'a')
    towrite = "\n continuing training.." + "\nTraining on LR: " + str(_lr)
    f.write(towrite+'\n')
    f.close()

    ValidationScoreArray = np.array([])
    TrainingScoreArray = np.array([])
    ValidationScoreArray = np.concatenate((np.load("Models/"+model_folder+"/ValidationScoreArr.npy"), ValidationScoreArray))
    TrainingScoreArray = np.concatenate((np.load("Models/"+model_folder+"/TrainingScoreArr.npy"), TrainingScoreArray))

    for epoch in range(modeldata["epoch"], _end_epoch):  
        print("Epoch "+str(epoch+1)+" Running..")

        f = open("Models/"+model_folder+"/log.txt", 'a')
        towrite = "Epoch "+str(epoch+1)+" Running.."
        f.write(towrite+'\n')
        f.close()
        
        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)

            loss = Chamfer_loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()

            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}')
                
                f = open("Models/"+model_folder+"/log.txt", 'a')
                towrite = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}'
                f.write(towrite+'\n')
                f.close()

                running_loss = 0.0
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}')
        
        f = open("Models/"+model_folder+"/log.txt", 'a')
        towrite = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}'
        f.write(towrite+'\n')
        f.close()
        
        currentScore = validator()
        print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
        print("")

        f = open("Models/"+model_folder+"/log.txt", 'a')
        towrite = "At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore)
        f.write(towrite+'\n\n')
        f.close()

        ValidationScoreArray = np.append(ValidationScoreArray, currentScore)
        TrainingScoreArray = np.append(TrainingScoreArray, TLoss)
        
        if currentScore < bestScore:
            bestScore = currentScore
            bestEpoch = epoch+1
            print("Saving Model at Epoch "+str(epoch+1)+"...")

            f = open("Models/"+model_folder+"/log.txt", 'a')
            towrite = "Saving Model at Epoch "+str(epoch+1)+"..."
            f.write(towrite+'\n')
            f.close()

            torch.save(
                {   'epoch':epoch, 
                    "last_train_epoch":bestEpoch,
                    "last_train_score":modeldata["score"],
                    'classes':CLASSES,
                    "lr":_lr,
                    "batch_size":BATCH_SIZE,
                    "score":currentScore,
                    "num_views":NUM_VIEWS,
                    "num_heads":NUM_ENCODER_HEADS,
                    "num_layers":NUM_ENCODER_LAYERS,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                },
                "Models/"+model_folder+'/bestScore.pth')
            np.save( "Models/"+model_folder+"/ValidationScoreArr_continued.npy", np.array(ValidationScoreArray))
            np.save( "Models/"+model_folder+"/TrainingScoreArr_continued.npy", np.array(TrainingScoreArray))
    
    print("End training..")
    print("best Epoch = " + str(bestEpoch))
    print("best Score = " + str(bestScore))
    f = open("Models/"+model_folder+"/log.txt", 'a')
    towrite = "End training..\n"+"best Epoch = " + str(bestEpoch)+"\n"+"best Score = " + str(bestScore)
    f.write(towrite+'\n')
    f.close()

def train_one():

    # os.mkdir("Models/"+FOLDER_NAME)

    net.train()
    bestScore = sys.maxsize
    bestEpoch = 0
    # Loss = ChamferDistance(point_reduction="sum", batch_reduction="mean")

    optimizer = optim.Adam(net.parameters(), lr=LR)

    ValidationScoreArray = []
    TrainingScoreArray = []
    print("\nStart training..")
    for epoch in range(0, 1):
        print("Epoch "+str(epoch+1)+" Running..")
        running_loss = 0.0
        TLoss = 0
        for i, data in enumerate(TrainLoader, 0):
            optimizer.zero_grad()
            imgs, pcs = data
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            res = net(imgs)
            loss = Chamfer_loss(pcs, res)
            loss.backward()
            optimizer.step()

            TLoss += loss.item()
            running_loss += loss.item()
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.7f}')
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.7f}')
                running_loss = 0.0
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i%100 + 1):.7f}')

        currentScore = validator()
        print("At Epoch \"" + str(epoch+1) + "\" Overall Score: " + str(currentScore))
    #     print("")
    #     ValidationScoreArray.append(currentScore)
    #     TrainingScoreArray.append(TLoss)
    #     if currentScore < bestScore:
    #         bestScore = currentScore
    #         # bestEpoch = epoch+1
    #         print("Saving Model at Epoch "+str(epoch+1)+"...")
    #         torch.save(
    #             {   'epoch':epoch, 
    #                 'classes':CLASSES,
    #                 "lr":LR,
    #                 "batch_size":BATCH_SIZE,
    #                 "score":currentScore,
    #                 "num_views":NUM_VIEWS,
    #                 "num_heads":NUM_ENCODER_HEADS,
    #                 "num_layers":NUM_ENCODER_LAYERS,
    #                 'model_state_dict': net.state_dict(), 
    #                 'optimizer_state_dict': optimizer.state_dict()
    #             },
    #             "Models/"+FOLDER_NAME+'/bestScore.pth')
    #         np.save( "Models/"+FOLDER_NAME+"/ValidationScoreArr.npy", np.array(ValidationScoreArray))
    #         np.save( "Models/"+FOLDER_NAME+"/TrainingScoreArr.npy", np.array(TrainingScoreArray))
    # print("End training..")
    # print("best Epoch = " + str(bestEpoch))
    # print("best Score = " + str(bestScore))
        

if __name__ == "__main__":

    # os.mkdir("Models/"+FOLDER_NAME)

    ShapeNetTrainData = ShapeNetDataset(classes=CLASSES, split="train80", transforms=TRANSFORMS)
    TrainLoader = DataLoader(ShapeNetTrainData, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

    ShapeNetTestData = ShapeNetDataset(classes=CLASSES, split="val10", transforms=TRANSFORMS)
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

    towrite = []

    towrite.append("\n--------------- Training ---------------")

    towrite.append("\n")
    towrite.append("Classes to Train on: \t" + str(CLASSES))
    towrite.append("")
    towrite.append("Device: " + str(device))
  
    towrite.append("\nTrain Data Size: \t" + str(len(ShapeNetTrainData)))
    towrite.append("Batche Size: \t\t" + str(BATCH_SIZE))
    towrite.append("Batches / Epochs: \t" + str(len(TrainLoader)))
    towrite.append("Total Epochs: \t\t" + str(END_EPOCH))
    towrite.append("Total Iterations: \t" + str(END_EPOCH*len(TrainLoader)))

    towrite.append("\nTest Data Size : \t" + str(len(ShapeNetTestData)))
    towrite.append("Batches / Epochs: \t" + str(len(TestLoader)))
    _fold = "12_03_02_23"
    # _fold = FOLDER_NAME

    with open("Models/"+_fold+"/log.txt", 'a') as f:
        for txt in towrite:
            f.write(txt+"\n")


    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS).to(device)

    # train_one()

    # training()
    finetune(_fold)

    # continue_training(_fold)
    # finetune(_fold)
