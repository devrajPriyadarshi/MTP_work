import numpy as np
from random import sample
from tqdm import tqdm
import torch
import torchvision.transforms as tf

from Network import Network

import matplotlib.pyplot as plt

from Preprocess import fixPointcloudOrientation
from Vizualization import ImageFromTensor
from DataLoaders import ShapeNetDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
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
    # model_folder = "11_23_18_25"

    # ModelData = torch.load("Models/"+model_folder+"/bestScore.pth", map_location=device)
    
    ModelData = torch.load("Models/"+model_folder+"/bestScore_finetuned.pth", map_location=device)
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

    chair_data = ShapeNetDataset(classes=['chair'], split="val10", transforms=TRANSFORMS)
    table_data = ShapeNetDataset(classes=['table'], split="val10", transforms=TRANSFORMS)
    
    table1_img , table1_pc = table_data[339] # normal
    table2_img , table2_pc = table_data[778] # circle
    table3_img , table3_pc = table_data[797] # short

    chair1_img , chair1_pc = chair_data[374] # down-low
    chair2_img , chair2_pc = chair_data[358] # big
    chair3_img , chair3_pc = chair_data[192] # normal

    chair1_img = chair1_img.unsqueeze(0).to(device)
    chair2_img = chair2_img.unsqueeze(0).to(device)
    chair3_img = chair3_img.unsqueeze(0).to(device)
    # chair1_pc = chair1_pc.unsqueeze(0).to(device)
    # chair2_pc = chair2_pc.unsqueeze(0).to(device)
    # chair3_pc = chair3_pc.unsqueeze(0).to(device)

    table1_img = table1_img.unsqueeze(0).to(device)
    table2_img = table2_img.unsqueeze(0).to(device)
    table3_img = table3_img.unsqueeze(0).to(device)
    # table1_pc = table1_pc.unsqueeze(0).to(device)
    # table2_pc = table2_pc.unsqueeze(0).to(device)
    # table3_pc = table3_pc.unsqueeze(0).to(device)


    # table -> chair 1
    pred_chair1 = net.encoder(chair1_img)
    pred_chair2 = net.encoder(chair2_img)
    pred_chair3 = net.encoder(chair3_img)
    
    pred_table1 = net.encoder(table1_img)
    pred_table2 = net.encoder(table2_img)
    pred_table3 = net.encoder(table3_img)
    
    weights = [2/6, 2.334/6, 2.667/6, 3/6, 3.334/6, 3.667/6, 4/6]

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_chair3, pred_table2, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/ChairToTable1_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/ChairToTable1.png")
    plt.close()

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_chair1, pred_table3, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/ChairToTable2_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/ChairToTable2.png")
    plt.close()
    
    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_chair2, pred_table1, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/ChairToTable3_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/ChairToTable3.png")
    plt.close()
    
    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_chair1, pred_chair2, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/ChairToChair1_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/ChairToChair1.png")
    plt.close()

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_chair2, pred_chair3, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/ChairToChair2_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/ChairToChair2.png")
    plt.close()

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_chair1, pred_chair3, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/ChairToChair3_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/ChairToChair3.png")
    plt.close()

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_table1, pred_table2, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/TableToTable1_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/TableToTable1.png")
    plt.close()

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_table2, pred_table3, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/TableToTable2_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/TableToTable2.png")
    plt.close()

    interpolate_tensor = torch.zeros((len(weights), 1000))
    for i in range(len(weights)):
        interpolate_tensor[i] = torch.lerp(pred_table1, pred_table3, weights[i])
    interpolated_pc = torch.transpose(net.decoder(interpolate_tensor.to(device)), 1, 2)
    interpolated_imgs = []
    for i in tqdm(range(len(weights))):
        img = ImageFromTensor(interpolated_pc[i])
        interpolated_imgs.append(img)
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(6*5, 5))
    for i in range(len(weights)):
        axs[i].imshow(interpolated_imgs[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig("Results/" + model_folder + "/bestScore_finetuned/TableToTable3_dettailed.png")
    # plt.savefig("Results/" + model_folder + "/bestScore/TableToTable3.png")
    plt.close()
    