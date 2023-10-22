import numpy as np
from numpy import pi
from random import uniform
import math
from time import time

import torch
from torchvision.transforms import GaussianBlur
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import euler_angles_to_matrix

from matplotlib import pyplot as plt
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class ChamferDistance():
    def __init__(self, point_reduction = "sum", batch_reduction = "mean"):
        self.br = batch_reduction
        self.pr = point_reduction
    def __call__(self,input, output):
        return chamfer_distance(input, output, point_reduction=self.pr, batch_reduction=self.br)[0]
    

class ProjectionLoss():

    def __init__(self, rotations: list, batch_reduction = "mean", k = 5, sigma = 1.2, fc = 120, u = 32, d = 2.5):
        self.k = k
        self.sigma = sigma
        self.fc = fc
        self.u = u
        self.d = d

        self.bce = torch.nn.BCELoss(reduction="sum").to(device)
        self.L1 = torch.nn.L1Loss(reduction="sum").to(device)
        self.bce = torch.nn.MSELoss(reduction="sum").to(device)

        self.blurFilter = self.gaussianBlock(sigma, k)

        self.rotations = rotations
        self.batch_reduction = batch_reduction

    def __call__(self, pc1_batch: torch.Tensor, pc2_batch: torch.Tensor) -> torch.Tensor:
        A = torch.zeros((pc1_batch.shape[0], 64*64)).to(device)
        B = torch.zeros((pc1_batch.shape[0], 64*64)).to(device)
        for i in range(pc1_batch.shape[0]):
            pc1 = pc1_batch[i]
            pc2 = pc2_batch[i]
            for rot in self.rotations:
                proj1 = self.projectImg(pc1, rot)
                proj2 = self.projectImg(pc2, rot)
                proj1 = proj1.flatten()
                proj2 = proj2.flatten()
                A[i] = proj1
                B[i] = proj2
        if self.batch_reduction == "mean":
            return self.L1(A,B)/pc1_batch.shape[0]
        else:
            return self.L1(A,B)
    
    def gaussianBlock(self, sigma: float, k: int):
        arr = np.zeros((k,k), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                arr[i][j] = math.exp(-((i-k//2)**2 + (j-k//2)**2)/(2*sigma*sigma))/(2*pi*sigma*sigma)
        return arr
    
    def projectImg(self, pc1: torch.Tensor, eul: list) -> torch.Tensor:

        fx = fy = self.fc
        u0 = v0 = self.u
        d = self.d

        pc = pc1.detach()
        ones = torch.ones((1024, 1)).to(device)
        pc = torch.concat((pc, ones), dim=1).to(device)

        R = euler_angles_to_matrix(torch.Tensor(eul).to(device), "XYZ").to(device)
        K = torch.Tensor(np.array([  [ fx, 0, u0], [ 0, fy, v0], [ 0, 0, 1]])).to(device)
        T = torch.Tensor(np.array([ [0], [0], [d]])).to(device)
        ext = torch.cat( (R, T), dim = 1)

        pc = K@(ext@torch.transpose(pc, 0, 1))
        pc = torch.transpose(pc, 0, 1)
        div = pc[:, 2]
        div = div.reshape((1024, 1))
        pc = torch.div(pc, div)
        pc = torch.round(pc)

        proj_arr = torch.zeros((64,64)).to(device)

        for points in pc.cpu().numpy():
            x = int(points[0])
            y = int(points[1])
            for i in range(self.k):
                for j in range(self.k):
                    if x-self.k//2+i < 64 and y-self.k//2+j < 64:
                        proj_arr[x-self.k//2+i, y-self.k//2+j] += self.blurFilter[i][j]

        proj_arr = proj_arr/torch.max(proj_arr)

        return proj_arr
    
if __name__ == "__main__":
    PL_Obj = ProjectionLoss(rotations=[[0,0,np.pi/2]])
    CD_Obj = ChamferDistance()

    pc = torch.Tensor(np.load("tests\chair_sample\pointcloud_1024.npy")).to(device)
    pc2 = torch.Tensor(np.load("tests\chair_sample\pointcloud_1024.npy")).to(device)
    pc2 = pc2 + (0.0001**0.5)*torch.rand(pc2.shape).to(device)

    t1 = time()    
    loss1 = PL_Obj(pc.unsqueeze(0), pc2.unsqueeze(0))
    t2 = time()
    loss2 = CD_Obj(pc.unsqueeze(0), pc2.unsqueeze(0))
    t3 = time()
    
    print(loss1)
    print(loss2)

    print(t2 - t1)
    print(t3 - t1)

    da = np.load("tests\chair_sample\pointcloud_1024.npy")
    pc3 = torch.Tensor(np.array([da, da,da, da,da, da,da, da])).to(device)
    pc4 = torch.Tensor(np.array([da, da,da, da,da, da,da, da])).to(device)
    pc4 = pc4 + (0.0001**0.5)*torch.rand(pc4.shape).to(device)
    
    t1 = time()
    loss1 = PL_Obj(pc3, pc4)
    t2 = time()
    loss2 = CD_Obj(pc3, pc4)
    t3 = time()
    
    print(loss1)
    print(loss2)

    print(t2 - t1)
    print(t3 - t1)

    # img = (obj.projectImg(pc, eul=[0,0,0])).cpu().numpy()
    # fig = plt.figure()
    # plt.imshow(img, cmap="gray")
    # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# xs = pc[:,0].detach().cpu().numpy()
# ys = pc[:,1].detach().cpu().numpy()
# zs = pc[:,2].detach().cpu().numpy()
# img = ax.scatter(xs, ys, zs, cmap=plt.hot())
# fig.colorbar(img)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()