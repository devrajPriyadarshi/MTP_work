import numpy as np
import torch
import torch.nn as nn

"""
The Original Implementation of FoldingNet
Shapes are changed to Accomodate our need of 1024 points instead of 2048
therefore m != 2025 (45, 45), m = 1024 (32, 32)
"""

class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_channel=512):
        super(Decoder, self).__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, 32, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, 32, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)   # (2, 32, 32)

        # reshape
        self.grid = torch.Tensor(np.array(self.grid)).view(2, -1)  # (2, 32, 32) -> (2, 32 * 32)
        
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(in_channel + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 32 * 32)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 32 * 32)
        
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, 32 * 32)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        
        return recon2