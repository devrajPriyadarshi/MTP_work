# Need function to rotate the Point Cloud
# Maybe to add noise to them -> DONT
# Maybe to enchance the image

# import cv2
import numpy as np
from numpy import ndarray

import torch
import torchvision.transforms as TF

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def fixPointcloudOrientation(pointcloud: torch.Tensor) -> torch.Tensor:
    pc_numpy = pointcloud.cpu().numpy()
    rot = euler_angles_to_matrix(torch.Tensor([0,0,np.pi/2]), "XYZ")
    center = np.array([ (np.min(pc_numpy[:,0]) + np.max(pc_numpy[:,0])) / 2 ,
                        (np.min(pc_numpy[:,1]) + np.max(pc_numpy[:,1])) / 2 ,
                        (np.min(pc_numpy[:,2]) + np.max(pc_numpy[:,2])) / 2 ])
    verts = torch.Tensor(pc_numpy - center)
    return verts@rot

# def cv2ToTensor(img: ndarray) -> torch.Tensor:
#     return TF.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))