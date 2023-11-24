from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torchvision.transforms as tf

from Network import Network
from DataLoaders import ShapeNetDataset

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    # device = torch.device("cpu")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

TRANSFORMS = tf.Compose([   tf.ToTensor(),
                            tf.Resize((224,224), antialias=True),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

if __name__ == "__main__":

    model_folder = "10_24_03_49"

    ModelData = torch.load("Models/"+model_folder+"/bestScore.pth")
    # trainLoss = np.load("Models/"+model_folder+"/TrainingScoreArr.npy")
    # valLoss = np.load("Models/"+model_folder+"/ValidationScoreArr.npy")
    # assert len(trainLoss) == len(valLoss)

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

    ShapeNetTestChairData = ShapeNetDataset(classes=["chair"], split="val", transforms=TRANSFORMS)
    ShapeNetTestTableData = ShapeNetDataset(classes=["table"], split="val", transforms=TRANSFORMS)

    print("\n--------------- Codeword Clustering ---------------")

    print("\nDevice : " , device)
    print("\nChair Test Data Size : \t" + str(len(ShapeNetTestChairData)))
    print("Table Test Data Size : \t" + str(len(ShapeNetTestTableData)))
    # print("Batches / Epochs: \t" + str(len(TestLoader)))
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


    # codewords = []
    # for x in tqdm(range(0, len(ShapeNetTestChairData))):
    #     img, pc = ShapeNetTestChairData[x]
    #     img = img.unsqueeze(0).to(device)
    #     # pc = pc.to(device)
    #     res = net.encoder(img)
    #     codewords.append(res.detach().squeeze().cpu().numpy())

    # for x in tqdm(range(0, len(ShapeNetTestTableData))):
    #     img, pc = ShapeNetTestTableData[x]
    #     img = img.unsqueeze(0).to(device)
    #     # pc = pc.to(device)
    #     res = net.encoder(img)
    #     codewords.append(res.detach().squeeze().cpu().numpy())

    # codewords = np.array(codewords)

    # np.save("Models/"+model_folder+"/codewords.npy", codewords)
    
    codewords = np.load("Models/"+model_folder+"/codewords.npy")

    # # PCA
    # pca = PCA(n_components=2)
    # pca_code = pca.fit_transform(codewords)

    # plt.figure(figsize=(5,5))

    # chair_x = pca_code[ : len(ShapeNetTestChairData), 0]
    # chair_y = pca_code[ : len(ShapeNetTestChairData), 1]

    # table_x = pca_code[ len(ShapeNetTestChairData): , 0]
    # table_y = pca_code[ len(ShapeNetTestChairData): , 1]

    # plt.scatter(chair_x, chair_y, c="b")
    # plt.scatter(table_x, table_y, c="r")
    # plt.legend(["chair","table"])
    # plt.show()

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1500)
    tsne_code = tsne.fit_transform(codewords)

    plt.figure(figsize=(10,10))

    chair_x = tsne_code[ : len(ShapeNetTestChairData), 0]
    chair_y = tsne_code[ : len(ShapeNetTestChairData), 1]

    table_x = tsne_code[ len(ShapeNetTestChairData): , 0]
    table_y = tsne_code[ len(ShapeNetTestChairData): , 1]

    plt.scatter(chair_x, chair_y, s=15, c="b", alpha=0.3)
    plt.scatter(table_x, table_y, s=15, c="r", alpha=0.3)
    plt.legend(["chair","table"])
    # plt.show()
    plt.savefig("Results/" + model_folder + "/codeword_clustering.png")