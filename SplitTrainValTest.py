
# from PIL import Image
import json
# import numpy as np
from random import sample

# import torch
# from torch.utils.data import Dataset

from shapenet_taxonomy import shapenet_category_to_id
# from Preprocess import fixPointcloudOrientation, cv2ToTensor

views = [str(i).zfill(2)+".png" for i in range(24)]


def get_f(root = "data/ShapeNet/", split = "train", classes = [], num_views = 24, transforms = None):
    
    assert classes != []

    root = root
    split = split
    num_views = num_views
    transforms = transforms

    classID = [ shapenet_category_to_id[idx] for idx in classes]
    files = []

    with open(root+"splits/"+split+"_models.json") as f:
        train_split = json.load(f)
        for classes in classID:
            files += train_split[classes]

    return len(files)

train_split = get_f(classes=['chair', 'table'])
val_split = get_f(split="val", classes=['chair', 'table'])

print(train_split)
print(val_split)

total = train_split+val_split

print(train_split/total)
print(val_split/total)
print(total*0.05)
print(764/val_split)

def split_models(root = "data/ShapeNet/", split = "train", classes = [], num_views = 24, transforms = None):
    
    assert classes != []

    root = root
    split = split
    num_views = num_views
    transforms = transforms

    classID = [ shapenet_category_to_id[idx] for idx in classes]
    train_files = []
    val_files = []
    test_files = []

    train_dict = {}
    val_dict = {}
    test_dict = {}

    with open(root+"splits/"+"train"+"_models.json") as f:
        train_split = json.load(f)
        for classes in classID:
            train_files = train_split[classes]
            train_dict[classes] = train_files

    with open(root+"splits/"+"val"+"_models.json") as f:
        val_split = json.load(f)
        for classes in classID:
            val_files, test_files = val_split[classes][:len(val_split[classes])//2] , val_split[classes][len(val_split[classes])//2:]
            
            val_dict[classes] = val_files
            test_dict[classes] = test_files
            print(len(val_split[classes]))
            print(len(val_files))
            print(len(test_files))
    return train_dict, test_dict, val_dict
    

train_dict, test_dict, val_dict = split_models(split="val", classes=['chair', 'table'])

with open("data/ShapeNet/"+"splits/"+"train80_models.json", "w") as outfile:
    json.dump(train_dict, outfile)

with open("data/ShapeNet/"+"splits/"+"test10_models.json", "w") as outfile:
    json.dump(test_dict, outfile)

with open("data/ShapeNet/"+"splits/"+"val10_models.json", "w") as outfile:
    json.dump(val_dict, outfile)