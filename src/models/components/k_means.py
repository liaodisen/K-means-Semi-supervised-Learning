import torch
from torch import nn

class KmeanModel(nn.Module):
    def __init__(self, num_centroids):
        super(KmeanModel, self).__init__()
        self.num_centroids = num_centroids
        #512 to match the last layer of resnet18
        self.centroids = nn.Parameter(torch.randn(num_centroids, 512))

    def forward(self, x):
        z = x[:, None, :]
        mu = self.centroids[None]
        dist = (z-mu).norm(2,dim=2)
        # returns the distance to each centroids.
        return dist # 128 x 10
