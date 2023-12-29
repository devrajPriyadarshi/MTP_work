import os
import json
from random import sample
from operator import getitem
from collections import OrderedDict

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as tf
from torch.utils.data import DataLoader

from Network import Network
from Metric0 import ChamferDistance
from DataLoaders import ShapeNetDataset
from Vizualization import ImageFromTensor

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

TRANSFORMS = tf.Compose([   tf.ToTensor(),
                            tf.Resize((224,224), antialias=True),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

invTrans = tf.Compose([ tf.Normalize(mean = [ 0., 0., 0. ],
                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                        tf.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                    std = [ 1., 1., 1. ]),
                               ])

def validator(dataset: ShapeNetDataset):
    print("\nRunning Validator...")
    net.to(device)
    net.eval()
    TotalChamferLoss = 0.0
    ValidateCriterion = ChamferDistance(point_reduction="mean", batch_reduction="sum")

    for i in tqdm(range(len(dataset))):
        imgs, pcs = dataset[i]
        imgs = imgs.unsqueeze(0)
        pcs = pcs.unsqueeze(0)
        imgs = imgs.to(device)
        pcs = pcs.to(device)
        res = net(imgs)
        loss = ValidateCriterion(pcs, res)
        TotalChamferLoss += loss.item()

    return TotalChamferLoss

def nViewRes(net: Network, imgTensor):
    views = [1, 3, 6, 12, 24]
    res_pc = []
    for v in range(len(views)):        
        res_pc.append(net(imgTensor[v]))

    return res_pc

def nViewChamferDistance(gt_vec, pred_vec):
    cd_loss = ChamferDistance(point_reduction="mean", batch_reduction="mean")

    res_cd = []
    views = [1, 3, 6, 12, 24]

    for v in range(len(views)):
        res_cd.append(cd_loss(gt_vec[v], pred_vec[v]).item())

    return res_cd

def nViewPCimg(pc_vec):
    res_img = []

    for v in tqdm(range(len(pc_vec))):
        res_img.append(ImageFromTensor(pc_vec[v].squeeze()))

    return res_img

if __name__ == "__main__":

    model_folder = "12_03_02_23"
    # model_folder = "11_23_18_25"

    ModelData = torch.load("Models/"+model_folder+"/bestScore.pth", map_location=device)
    trainLoss = np.load("Models/"+model_folder+"/TrainingScoreArr.npy")
    valLoss = np.load("Models/"+model_folder+"/ValidationScoreArr.npy")
    # assert len(trainLoss) == len(valLoss)

    # ModelData= torch.load("Models/"+model_folder+"/bestScore_finetuned.pth", map_location=device)
    # trainLoss= np.load("Models/"+model_folder+"/TrainingScoreArr_finetuned.npy")
    # valLoss= np.load("Models/"+model_folder+"/ValidationScoreArr_finetuned.npy")

    # assert len(trainLoss) == len(valLoss)
    # assert len(trainLoss_finetune) == len(valLoss_finetune)

    # beforeTuneEpoch = ModelData["before_tune_epoch"]

    BATCH_SIZE = ModelData["batch_size"]
    CLASSES = ModelData["classes"]
    NUM_VIEWS = ModelData["num_views"]
    NUM_ENCODER_LAYERS = ModelData["num_layers"]
    NUM_ENCODER_HEADS = ModelData["num_heads"]
    SHUFFLE = True
    WORKERS = 6
    
    epochLable = np.arange(1,len(trainLoss)+1)

    ShapeNetTrainData = ShapeNetDataset(classes=CLASSES, split="train80", transforms=TRANSFORMS)
    ShapeNetTestData = ShapeNetDataset(classes=CLASSES, split="test10", transforms=TRANSFORMS)
    TestLoader = DataLoader(ShapeNetTestData, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

    ShapeNetTestChairData = ShapeNetDataset(classes=["chair"], split="test10", transforms=TRANSFORMS)
    ShapeNetTestTableData = ShapeNetDataset(classes=["table"], split="test10", transforms=TRANSFORMS)

    net = Network(num_views=NUM_VIEWS, num_heads=NUM_ENCODER_HEADS, num_layer=NUM_ENCODER_LAYERS)
    net.load_state_dict(ModelData["model_state_dict"])

    del ModelData

    net = net.to(device)
    net.eval()

    chair = np.array([])
    table = np.array([])

    print("\nSaving Figures..")

    # plt.figure(figsize=(10,5))
    # plt.plot(epochLable[:beforeTuneEpoch+1], trainLoss[:beforeTuneEpoch+1]/(len(ShapeNetTrainData)), label = "Training Loss", linewidth = 1)
    # plt.plot(epochLable[:beforeTuneEpoch+1], valLoss[:beforeTuneEpoch+1]/(2*(len(ShapeNetTestData))), label = "Validation Loss", color="orange", linewidth = 1)
    # plt.plot(epochLable[beforeTuneEpoch+1:], valLoss[beforeTuneEpoch+1:]/len(ShapeNetTestData), label = "Validation Loss", color="orange", linewidth = 1)
    # plt.legend(['Training Loss', 'Validation Loss'], loc = 'upper right')
    # plt.xlabel("Epochs")
    # plt.ylabel("Chamfer Distance Error")
    # plt.axvline(x = beforeTuneEpoch+1.5, color = 'black', linestyle="dashed", linewidth = 0.7)
    # plt.twinx()
    # plt.plot(epochLable[beforeTuneEpoch+1:], trainLoss[beforeTuneEpoch+1:]/(len(ShapeNetTrainData)), label = "Finetuning Loss", color="green", linewidth = 1)
    # plt.ylabel("Chamfer Distance Error + Projection Loss")
    # plt.legend(['Finetuning Loss'], loc = 'upper right', bbox_to_anchor=(1,0.85))
    # plt.savefig("Results/"+model_folder+"/bestScore_finetuned/history.png")
    # plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochLable, trainLoss/(len(ShapeNetTrainData)), label = "Training Loss", linewidth = 1)
    plt.plot(epochLable, valLoss/((len(ShapeNetTestData))), label = "Validation Loss", color="orange", linewidth = 1)
    plt.legend(['Training Loss', 'Validation Loss'], loc = 'upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Chamfer Distance Error")
    plt.savefig("Results/"+model_folder+"/bestScore/history.png")
    plt.close()

    # k_chair = sample(range(0, len(ShapeNetTestChairData)), 10)
    # k_table = sample(range(0, len(ShapeNetTestTableData)), 10)
    
    # k_chair = [374]
    k_chair = [192, 98, 374] + sample(range(0, len(ShapeNetTestChairData)), 7)
    k_table = [358, 537, 797] + sample(range(0, len(ShapeNetTestTableData)), 7)

    print("\nChoosing Chair Samples: ", k_chair)
    # print("Choosing Table Samples: ", k_table)

    ShapeNetTestChairData1 = ShapeNetDataset(classes=["chair"], split="val10", transforms=TRANSFORMS, num_views=1)
    ShapeNetTestChairData3 = ShapeNetDataset(classes=["chair"], split="val10", transforms=TRANSFORMS, num_views=3)
    ShapeNetTestChairData6 = ShapeNetDataset(classes=["chair"], split="val10", transforms=TRANSFORMS, num_views=6)
    ShapeNetTestChairData12 = ShapeNetDataset(classes=["chair"], split="val10", transforms=TRANSFORMS, num_views=12)
    ShapeNetTestChairData24 = ShapeNetDataset(classes=["chair"], split="val10", transforms=TRANSFORMS, num_views=24)

    ShapeNetTestTableData1 = ShapeNetDataset(classes=["table"], split="val10", transforms=TRANSFORMS, num_views=1)
    ShapeNetTestTableData3 = ShapeNetDataset(classes=["table"], split="val10", transforms=TRANSFORMS, num_views=3)
    ShapeNetTestTableData6 = ShapeNetDataset(classes=["table"], split="val10", transforms=TRANSFORMS, num_views=6)
    ShapeNetTestTableData12 = ShapeNetDataset(classes=["table"], split="val10", transforms=TRANSFORMS, num_views=12)
    ShapeNetTestTableData24 = ShapeNetDataset(classes=["table"], split="val10", transforms=TRANSFORMS, num_views=24)

    chair_cf_dist = {}
    table_cf_dist = {}

    try:
        # with open("Results/"+model_folder+"/bestScore_finetuned/chair_sample_cd.json", "r") as outfile: 
        with open("Results/"+model_folder+"/bestScore/chair_sample_cd.json", "r") as outfile: 
            chair_cf_dist = json.load(outfile)
            print("Found Json")
    except:
        pass
    try:
        # with open("Results/"+model_folder+"/bestScore_finetuned/table_sample_cd.json", "r") as outfile: 
        with open("Results/"+model_folder+"/bestScore/table_sample_cd.json", "r") as outfile: 
            table_cf_dist = json.load(outfile)
            print("Found Json")
    except:
        pass

    for k in k_chair:
        print("\nChair Sample %d:"%k)
        try: 
            # os.mkdir("Results/"+model_folder+"/bestScore_finetuned/"+"chair_sample_%d"%k) 
            os.mkdir("Results/"+model_folder+"/bestScore/"+"chair_sample_%d"%k) 
        except OSError as error:
            print("Chair Sample %d Already There..."%k)
            continue

        img1, pc1 = ShapeNetTestChairData1[k]
        img3, pc3 = ShapeNetTestChairData3[k]
        img6, pc6 = ShapeNetTestChairData6[k]
        img12, pc12 = ShapeNetTestChairData12[k]
        img24, pc24 = ShapeNetTestChairData24[k]

        img_vec = [img1.unsqueeze(0).to(device), img3.unsqueeze(0).to(device), img6.unsqueeze(0).to(device), img12.unsqueeze(0).to(device), img24.unsqueeze(0).to(device)]
        gt_vec = [pc1.to(device).unsqueeze(0), pc3.to(device).unsqueeze(0), pc6.to(device).unsqueeze(0), pc12.to(device).unsqueeze(0), pc24.to(device).unsqueeze(0)]

        pred_vec = nViewRes(net, img_vec)
        cd_loss = nViewChamferDistance(gt_vec, pred_vec)

        pred_img = nViewPCimg(pred_vec)
        gt_img = nViewPCimg([gt_vec[0]])
        views = [1,3,6,12,24]

        t = 0
        for img in pred_img:
            plt.figure(figsize=(5,5))
            plt.imshow(img.clip(min=0, max=1))
            plt.axis("off")
            plt.tight_layout()
            # plt.savefig("Results/"+model_folder+"/bestScore_finetuned/chair_sample_%d/predPC_%dView.png" % (k, views[t]) )
            plt.savefig("Results/"+model_folder+"/bestScore/chair_sample_%d/predPC_%dView.png" % (k, views[t]) )
            plt.close()
            t+=1

        plt.figure(figsize=(5,5))
        plt.imshow(gt_img[0].clip(min=0, max=1))
        plt.axis("off")
        plt.tight_layout()
        # plt.savefig("Results/"+model_folder+"/bestScore_finetuned/chair_sample_%d/gtPC.png" % k )
        plt.savefig("Results/"+model_folder+"/bestScore/chair_sample_%d/gtPC.png" % k )
        plt.close()
        
        chamfer_dist = {
            "1_view": cd_loss[0],
            "3_view": cd_loss[1],
            "6_view": cd_loss[2],
            "12_view": cd_loss[3],
            "24_view": cd_loss[4]
        }

        chair_cf_dist["%d"%k] = chamfer_dist

        img = invTrans(img_vec[0].squeeze()).cpu().numpy()
        img = np.transpose(img, [1,2,0])
        img = (255*img).astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        # img.save("Results/"+model_folder+"/bestScore_finetuned/chair_sample_%d/single_view.png" % k)
        img.save("Results/"+model_folder+"/bestScore/chair_sample_%d/single_view.png" % k)

    # with open("Results/"+model_folder+"/bestScore_finetuned/chair_sample_cd.json", "w") as outfile:
    with open("Results/"+model_folder+"/bestScore/chair_sample_cd.json", "w") as outfile:
        chair_cf_dist = dict( OrderedDict( sorted( chair_cf_dist.items(), key = lambda x:getitem(x[1], '24_view') ) ) )
        json.dump(chair_cf_dist, outfile, indent="\t")

    print("\n\nDone with Chairs!")

    for k in k_table:
        print("\nTable Sample %d:"%k)
        try: 
            # os.mkdir("Results/"+model_folder+"/bestScore_finetuned/"+"table_sample_%d"%k) 
            os.mkdir("Results/"+model_folder+"/bestScore/"+"table_sample_%d"%k) 
        except OSError as error:
            print("Table Sample %d Already There..."%k)
            continue

        img1, pc1 = ShapeNetTestTableData1[k]
        img3, pc3 = ShapeNetTestTableData3[k]
        img6, pc6 = ShapeNetTestTableData6[k]
        img12, pc12 = ShapeNetTestTableData12[k]
        img24, pc24 = ShapeNetTestTableData24[k]

        img_vec = [img1.unsqueeze(0).to(device), img3.unsqueeze(0).to(device), img6.unsqueeze(0).to(device), img12.unsqueeze(0).to(device), img24.unsqueeze(0).to(device)]
        gt_vec = [pc1.to(device).unsqueeze(0), pc3.to(device).unsqueeze(0), pc6.to(device).unsqueeze(0), pc12.to(device).unsqueeze(0), pc24.to(device).unsqueeze(0)]

        pred_vec = nViewRes(net, img_vec)
        cd_loss = nViewChamferDistance(gt_vec, pred_vec)

        pred_img = nViewPCimg(pred_vec)
        gt_img = nViewPCimg([gt_vec[0]])
        views = [1,3,6,12,24]

        t = 0
        for img in pred_img:
            plt.figure(figsize=(5,5))
            plt.imshow(img.clip(min=0, max=1))
            plt.axis("off")
            plt.tight_layout()
            # plt.savefig("Results/"+model_folder+"/bestScore_finetuned/table_sample_%d/predPC_%dView.png" % (k, views[t]) )
            plt.savefig("Results/"+model_folder+"/bestScore/table_sample_%d/predPC_%dView.png" % (k, views[t]) )
            plt.close()
            t+=1

        plt.figure(figsize=(5,5))
        plt.imshow(gt_img[0].clip(min=0, max=1))
        plt.axis("off")
        plt.tight_layout()
        # plt.savefig("Results/"+model_folder+"/bestScore_finetuned/table_sample_%d/gtPC.png" % k )
        plt.savefig("Results/"+model_folder+"/bestScore/table_sample_%d/gtPC.png" % k )
        plt.close()

        chamfer_dist = {
            "1_view": cd_loss[0],
            "3_view": cd_loss[1],
            "6_view": cd_loss[2],
            "12_view": cd_loss[3],
            "24_view": cd_loss[4]
        }

        table_cf_dist["%d"%k] = chamfer_dist

        img = invTrans(img_vec[0].squeeze()).cpu().numpy()
        img = np.transpose(img, [1,2,0])
        img = (255*img).astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        # img.save("Results/"+model_folder+"/bestScore_finetuned/table_sample_%d/single_view.png" % k)
        img.save("Results/"+model_folder+"/bestScore/table_sample_%d/single_view.png" % k)

    # with open("Results/"+model_folder+"/bestScore_finetuned/table_sample_cd.json", "w") as outfile: 
    with open("Results/"+model_folder+"/bestScore/table_sample_cd.json", "w") as outfile: 
        table_cf_dist = dict( OrderedDict( sorted( table_cf_dist.items(), key = lambda x:getitem(x[1], '24_view') ) ) )
        json.dump(table_cf_dist, outfile, indent="\t")

    print("\n\nDone with Tables!")

# ##############################################

    try: 
        # with open("Results/"+model_folder+"/bestScore_finetuned/classes_cd.json", "r") as outfile:
        with open("Results/"+model_folder+"/bestScore/classes_cd.json", "r") as outfile:
            print("File Already Exists!")
            print(json.load(outfile))
    except:
        print("\n\nCalculating Classwise Error...")
        
        chair_dataset = [ShapeNetTestChairData1, ShapeNetTestChairData3, ShapeNetTestChairData6, ShapeNetTestChairData12, ShapeNetTestChairData24]
        table_dataset = [ShapeNetTestTableData1, ShapeNetTestTableData3, ShapeNetTestTableData6, ShapeNetTestTableData12, ShapeNetTestTableData24]
        class_chf_dist = {}
        views = ["1_view", "3_view", "6_view", "12_view", "24_view"]
        for i in range(len(chair_dataset)):
            print("\n "+views[i]+": ")
            chair_cd = (validator(chair_dataset[i]) /len(ShapeNetTestChairData))
            table_cd = (validator(table_dataset[i]) /len(ShapeNetTestTableData))

            class_chf_dist[views[i]] = {"chair":chair_cd, "table":table_cd}

        # with open("Results/"+model_folder+"/bestScore_finetuned/classes_cd.json", "w") as outfile: 
        with open("Results/"+model_folder+"/bestScore/classes_cd.json", "w") as outfile: 
            json.dump(class_chf_dist, outfile, indent="\t")