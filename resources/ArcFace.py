
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=128, margin=0.3, scale=16, easy_margin=False, device=None):
        super(ArcFaceLoss, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.m = margin
        self.s = scale
        self.easy_margin = easy_margin

        # Precompute constants
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        # Assign the device
        self.device = device

        # Move to the device if provided
        if self.device is not None:
            self.to(self.device)

    def forward(self, input, label):
        # Normalize input and weights
        x = F.normalize(input)
        W = F.normalize(self.weight)

        # Compute cosine similarity
        cosine = F.linear(x, W)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute phi (cosine after margin is applied)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-8))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding of labels on the correct device
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Combine phi and cosine for output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
