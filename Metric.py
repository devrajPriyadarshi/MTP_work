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
    # device = torch.device("cpu")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class ChamferDistance():
    def __init__(self, point_reduction = "sum", batch_reduction = "mean"):
        self.br = batch_reduction
        self.pr = point_reduction
    def __call__(self,input, output):
        return chamfer_distance(input, output, point_reduction=self.pr, batch_reduction=self.br)[0]
    

class ProjectionLoss(torch.nn.Module):

    def __init__(self, rotations: list, batch_reduction = "mean", view_reduction = "sum", batch_size = 32, k = 5, sigma = 1.2, fc = 120, u = 32, d = 2.5):
        super(ProjectionLoss, self).__init__()
        self.k = k
        self.sigma = sigma
        self.fc = fc
        self.u = u
        self.d = d

        self.L1 = torch.nn.L1Loss(reduction="sum").to(device)
        self.MSE = torch.nn.MSELoss(reduction="sum").to(device)

        self.blurFilter = self.gaussianBlock(sigma, k)

        self.batch_size = batch_size
        self.rotations = rotations
        self.batch_reduction = batch_reduction
        self.view_reduction = view_reduction

        # might move these to forward call
        self.prod_ = torch.Tensor(np.array([64*np.ones(1024),np.ones(1024),np.zeros(1024)])).to(device)

        fx = fy = self.fc
        u0 = v0 = self.u
        d = self.d
        self.K = torch.Tensor(np.array([  [ fx, 0, u0], [ 0, fy, v0], [ 0, 0, 1]])).to(device)
        self.T = torch.Tensor(np.array([ [0], [0], [d]])).to(device)
        self.R1 = euler_angles_to_matrix(torch.Tensor(self.rotations[0]), "XYZ").to(device)
        self.R2 = euler_angles_to_matrix(torch.Tensor(self.rotations[1]), "XYZ").to(device)
        self.R3 = euler_angles_to_matrix(torch.Tensor(self.rotations[2]), "XYZ").to(device)

        self.gaussianPlate = torch.zeros((64*64,64,64)).to(device)

        for idx in range(0, 64):
            for idy in range(0, 64):
                 for i in range(self.k):
                    for j in range(self.k):
                        if (idx-self.k//2+i >= 0 and idx-self.k//2+i < 64) and (idy-self.k//2+j >= 0 and idy-self.k//2+j < 64):
                            self.gaussianPlate[ idx*64 + idy, idx-self.k//2+i, idy-self.k//2+j] += self.blurFilter[i][j]

    def forward(self, gt_batch: torch.Tensor, res_batch: torch.Tensor) -> torch.Tensor:

        self.ones = torch.ones((gt_batch.shape[0], 1024, 1)).to(device)
        self.prod = self.prod_.repeat(gt_batch.shape[0], 1, 1)

        proj1 = torch.matmul( self.K@torch.concat((self.R1,self.T), dim = 1), torch.transpose(torch.concat((gt_batch.detach(), self.ones), dim = 2), 1, 2))
        proj2 = torch.div(proj1, proj1[:,2,:].unsqueeze(1))
        proj3 = torch.sum(torch.mul(self.prod, torch.round(proj2)), dim = 1).int()
        gt_proj1 = torch.sum(self.gaussianPlate[proj3.tolist(),:,:], dim=1).flatten(start_dim=1)
        proj1 = torch.matmul( self.K@torch.concat((self.R1,self.T), dim = 1), torch.transpose(torch.concat((res_batch.detach(), self.ones), dim = 2), 1, 2))
        proj2 = torch.div(proj1, proj1[:,2,:].unsqueeze(1))
        proj3 = torch.sum(torch.mul(self.prod, torch.round(proj2)), dim = 1).int()
        pred_proj1 = torch.sum(self.gaussianPlate[proj3.tolist(),:,:], dim=1).flatten(start_dim=1)

        proj1 = torch.matmul( self.K@torch.concat((self.R2,self.T), dim = 1), torch.transpose(torch.concat((res_batch.detach(), self.ones), dim = 2), 1, 2))
        proj2 = torch.div(proj1, proj1[:,2,:].unsqueeze(1))
        proj3 = torch.sum(torch.mul(self.prod, torch.round(proj2)), dim = 1).int()
        gt_proj2 = torch.sum(self.gaussianPlate[proj3.tolist(),:,:], dim=1).flatten(start_dim=1)
        proj1 = torch.matmul( self.K@torch.concat((self.R2,self.T), dim = 1), torch.transpose(torch.concat((res_batch.detach(), self.ones), dim = 2), 1, 2))
        proj2 = torch.div(proj1, proj1[:,2,:].unsqueeze(1))
        proj3 = torch.sum(torch.mul(self.prod, torch.round(proj2)), dim = 1).int()
        pred_proj2 = torch.sum(self.gaussianPlate[proj3.tolist(),:,:], dim=1).flatten(start_dim=1)

        proj1 = torch.matmul( self.K@torch.concat((self.R3,self.T), dim = 1), torch.transpose(torch.concat((res_batch.detach(), self.ones), dim = 2), 1, 2))
        proj2 = torch.div(proj1, proj1[:,2,:].unsqueeze(1))
        proj3 = torch.sum(torch.mul(self.prod, torch.round(proj2)), dim = 1).int()
        gt_proj3 = torch.sum(self.gaussianPlate[proj3.tolist(),:,:], dim=1).flatten(start_dim=1)
        proj1 = torch.matmul( self.K@torch.concat((self.R3,self.T), dim = 1), torch.transpose(torch.concat((res_batch.detach(), self.ones), dim = 2), 1, 2))
        proj2 = torch.div(proj1, proj1[:,2,:].unsqueeze(1))
        proj3 = torch.sum(torch.mul(self.prod, torch.round(proj2)), dim = 1).int()
        pred_proj3 = torch.sum(self.gaussianPlate[proj3.tolist(),:,:], dim=1).flatten(start_dim=1)

        Loss = self.L1(gt_proj1, pred_proj1) + self.L1(gt_proj2, pred_proj2) + self.L1(gt_proj3, pred_proj3)
        
        if self.view_reduction == "mean": Loss /= len(self.rotations)
        if self.batch_reduction == "mean": Loss /= gt_batch.shape[0]

        return Loss
        
    def gaussianBlock(self, sigma: float, k: int):
        arr = np.zeros((k,k), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                arr[i][j] = math.exp(-((i-k//2)**2 + (j-k//2)**2)/(2*sigma*sigma))/(2*pi*sigma*sigma)
        return arr
    
    # def projectImg(self, pc_orig: torch.Tensor, eul: list) -> torch.Tensor:

    #     pc = torch.concat((pc_orig, self.ones), dim=1)
    #     R = euler_angles_to_matrix(torch.Tensor(eul), "XYZ").to(device)
    #     ext = torch.cat( (R, self.T), dim = 1)
    #     pc = self.K@(ext@torch.transpose(pc, 0, 1))
    #     div = pc[2,:]
    #     pc = torch.div(pc, div.unsqueeze(0))
    #     pc = torch.round(pc)

    #     idxs = torch.sum(torch.mul(pc, self.prod), dim=0).int().tolist()
    #     proj_arr = torch.sum(self.gaussianPlate[idxs,:,:], dim = 0)

    #     proj_arr = proj_arr/torch.max(proj_arr)
    #     return proj_arr
    
if __name__ == "__main__":
    PL_Obj = ProjectionLoss(rotations=[[0,0,np.pi/2], [0,0,np.pi/2], [0,0,np.pi/2]])
    CD_Obj = ChamferDistance()

    # pc = torch.Tensor(np.load("tests\chair_sample\pointcloud_1024.npy")).to(device)
    # pc2 = torch.Tensor(np.load("tests\chair_sample\pointcloud_1024.npy")).to(device)
    # pc2 = pc2 + (0.0001**0.5)*torch.rand(pc2.shape).to(device)

    # t1 = time()    
    # loss1 = PL_Obj(pc.unsqueeze(0), pc2.unsqueeze(0))
    # t2 = time()
    # loss2 = CD_Obj(pc.unsqueeze(0), pc2.unsqueeze(0))
    # t3 = time()
    
    # print(loss1)
    # print(loss2)

    # print(t2 - t1)
    # print(t3 - t1)

    da = np.load("tests\chair_sample\pointcloud_1024.npy")
    pc3 = torch.Tensor(np.array([da, da,da, da,da, da,da, da, da, da,da, da,da, da,da, da, da, da,da, da,da, da,da, da, da, da,da, da,da, da,da, da])).to(device)
    pc4 = torch.Tensor(np.array([da, da,da, da,da, da,da, da, da, da,da, da,da, da,da, da, da, da,da, da,da, da,da, da, da, da,da, da,da, da,da, da])).to(device)
    pc4 = pc4 + (0.0001**0.5)*torch.rand(pc4.shape).to(device)
    
    t1 = time()
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    loss1 = PL_Obj(pc3, pc4)
    t2 = time()
    # loss2 = CD_Obj(pc3, pc4)
    t3 = time()
    
    print(loss1)
    # print(loss2)

    print(t2 - t1)
    print(t3 - t1)

#     # img = (obj.projectImg(pc, eul=[0,0,0])).cpu().numpy()
#     # fig = plt.figure()
#     # plt.imshow(img, cmap="gray")
#     # plt.show()

# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # xs = pc[:,0].detach().cpu().numpy()
# # ys = pc[:,1].detach().cpu().numpy()
# # zs = pc[:,2].detach().cpu().numpy()
# # img = ax.scatter(xs, ys, zs, cmap=plt.hot())
# # fig.colorbar(img)

# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_zlabel('Z')

# # plt.show()