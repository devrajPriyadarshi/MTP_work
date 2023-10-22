import json
from random import choice, randint

import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointsRasterizationSettings, PointsRenderer, PointsRasterizer, AlphaCompositor

from shapenet_taxonomy import shapenet_category_to_id as ID
from MitsubaRendering import ImageFromNumpyArr

pc_folder = "data/ShapeNet/ShapeNet_pointclouds/"
rend_folder = "data/ShapeNet/ShapeNetRendering/"

json_file = open("data/ShapeNet/splits/train_models.json")
train_split = json.load(json_file)
json_file.close()

classes = [ ID['car'], ID['chair'] ]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class Vizualize_1:
    
    def __call__(self, rgbImg:npt.ArrayLike , pointcloud: npt.ArrayLike, img_metadata: list) -> plt:
        
        azimuthal, elevation, inPlaneRotation, distance, fov = img_metadata

        rot = euler_angles_to_matrix(torch.Tensor([0,0,np.pi/2]).to(device), "XYZ")
        center = np.array([ (np.min(pointcloud[:,0]) + np.max(pointcloud[:,0])) / 2 ,
                            (np.min(pointcloud[:,1]) + np.max(pointcloud[:,1])) / 2 ,
                            (np.min(pointcloud[:,2]) + np.max(pointcloud[:,2])) / 2 ])

        verts = torch.Tensor(np.array([pointcloud - center])).to(device)
        feat = torch.Tensor(np.ones(verts.shape)*255).to(device)
        point_cloud = Pointclouds(points=verts@rot, features=feat)

        R, T = look_at_view_transform(dist=7, elev=elevation, azim=-azimuthal, degrees=True)
        cameras = FoVPerspectiveCameras(znear=0.01,fov=10, R=R, T=T, device=device)

        raster_settings = PointsRasterizationSettings(
            image_size=512, 
            radius = 0.005,
            points_per_pixel = 100
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        pc_img = renderer(point_cloud)[0, ..., :3].cpu().numpy()
        
        alpha = np.sum(rgbImg, axis=-1) > 0
        alpha = np.uint8(alpha * 255)
        res = np.dstack((rgbImg, alpha))

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow(rgbImg)
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(pc_img/np.max(pc_img))
        plt.axis("off")
        plt.tight_layout()

        return plt

def Vizualize_all(rgbImgs:torch.Tensor , pointcloud: torch.Tensor) -> plt:
    """
    Input:
        rgbImgs: a torch tensor of Nx3xHxW shape. N multiview images of object.
        pointcloud: a torch tensor of Mx3 shape.
    Output:
        plt: matplotlib.pyplot obj

    Minimum N = 2.
    """

    # Pointcloud Processing
    azimuthal, elevation = 30, 30
    verts = torch.Tensor(np.array([pointcloud.cpu().numpy()])).to(device)
    feat = torch.Tensor(np.ones(verts.shape)*255).to(device)
    point_cloud = Pointclouds(points=verts, features=feat)

    R, T = look_at_view_transform(dist=7, elev=elevation, azim=-azimuthal, degrees=True)
    cameras = FoVPerspectiveCameras(znear=0.01,fov=10, R=R, T=T, device=device)

    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = 0.005,
        points_per_pixel = 100
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    pc_img = renderer(point_cloud)[0, ..., :3].cpu().numpy()

    # Images Processing
    imgs_N = rgbImgs.shape[0]

    plt.figure(figsize=(10, 5))
    for _ in range(1,imgs_N+1):
        plt.subplot(100+(imgs_N+1)*10+_)
        plt.imshow(rgbImgs[_-1].permute(1,2,0))
        plt.axis("off")
    plt.subplot(100+(imgs_N+1)*10+(imgs_N+1))
    plt.imshow(pc_img/np.max(pc_img))
    plt.axis("off")
    plt.tight_layout()

    return plt

def ComparePointClouds(pc1: torch.Tensor, pc2: torch.Tensor) -> plt:
    """
    Input:
        pc1: a torch tensor of Nx3 shape. N Points.
        pc2: a torch tensor of Nx3 shape. N Points.
    Output:
        plt: matplotlib.pyplot obj
    """

    # Pointcloud Processing
    azimuthal, elevation = 30, 30
    verts = torch.Tensor(np.array([pc1.cpu().detach().numpy()])).to(device)
    feat = torch.Tensor(np.ones(verts.shape)*255).to(device)
    point_cloud = Pointclouds(points=verts, features=feat)

    R, T = look_at_view_transform(dist=7, elev=elevation, azim=-azimuthal, degrees=True)
    cameras = FoVPerspectiveCameras(znear=0.01,fov=10, R=R, T=T, device=device)

    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = 0.005,
        points_per_pixel = 100
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    pc_img = renderer(point_cloud)[0, ..., :3].cpu().numpy()

    verts = torch.Tensor(np.array([pc2.cpu().detach().numpy()])).to(device)
    feat = torch.Tensor(np.ones(verts.shape)*255).to(device)
    point_cloud = Pointclouds(points=verts, features=feat)

    R, T = look_at_view_transform(dist=7, elev=elevation, azim=-azimuthal, degrees=True)
    cameras = FoVPerspectiveCameras(znear=0.01,fov=10, R=R, T=T, device=device)

    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = 0.005,
        points_per_pixel = 100
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    pc_img2 = renderer(point_cloud)[0, ..., :3].cpu().numpy()


    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(pc_img/np.max(pc_img))
    plt.axis("off")
    
    plt.subplot(122)
    plt.imshow(pc_img2/np.max(pc_img2))
    plt.axis("off")
    
    plt.tight_layout()

    return plt

def ImageFromTensor(pc: torch.Tensor):
    pcl = pc.detach().cpu().numpy()
    img = ImageFromNumpyArr(pcl)
    return img.clip(min=0.0, max=1.0)

def ImageFromNumpy(np_arr):
    img = ImageFromNumpyArr(np_arr)
    return img

if __name__ == "__main__":

    file = choice(train_split[choice(classes)])

    rendering_metadata = open(rend_folder + file + "/rendering/rendering_metadata.txt", "r")
    metadatas_ = rendering_metadata.read()
    metadatas = metadatas_.split('\n')[:-1]
    rendering_metadata.close()

    renderings = open(rend_folder + file + "/rendering/renderings.txt", "r")
    renders_ = renderings.read()
    renders = renders_.split('\n')[:-1]
    renderings.close()

    version     = randint(0,len(renders)-1)

    pointcloud  = np.load(pc_folder + file +"/pointcloud_2048.npy")
    metadata    = list(map(float, metadatas[version].split()))
    image       = cv2.imread(rend_folder + file + "/rendering/" + renders[version])

    visualizer = Vizualize_1()

    plot = visualizer(rgbImg=image, pointcloud=pointcloud, img_metadata=metadata)
    plot.show()