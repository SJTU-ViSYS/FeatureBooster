import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import NoGradientError


class FastAPLoss(nn.Module):
    def __init__(self, num_bins=32, max_distance=256.):
        super(FastAPLoss, self).__init__()
        self.num_bins = num_bins
        self.max_distance = max_distance

    def forward(self, des_distance, pos_labels, neg_labels, size_average=True):
        """
        des_distance: Tensor, [N, M]
        pos_labels: Tensor, [N, M], 0 or 1
        neg_labels: Tensor, [N, M], 0 or 1
        """
        Delta = torch.tensor(self.max_distance / self.num_bins).cuda()
        Z = torch.linspace(0., self.max_distance, steps=self.num_bins+1).cuda()
        Z = Z.view(-1, 1, 1)
        N_pos = torch.sum(pos_labels, dim=1)
        pulse = F.relu(1. - torch.abs(des_distance - Z) / Delta)
        h_pos = torch.sum(pulse * pos_labels, dim=2).t()
        h_neg = torch.sum(pulse * neg_labels, dim=2).t()
        H_pos = torch.cumsum(h_pos, dim=1)
        H = torch.cumsum(h_pos + h_neg, dim=1)

        h_product = h_pos * H_pos
        safe_idx = (h_product > 0) & (H > 0)
        FastAP = torch.zeros_like(h_pos)
        FastAP[safe_idx] = h_product[safe_idx] / H[safe_idx]
        FastAP = torch.sum(FastAP, dim=1) / N_pos
        FastAP = torch.clamp(FastAP, max=1.0)

        if size_average:
            loss = 1 - torch.mean(FastAP)
        else:
            loss = 1 - FastAP

        return loss
