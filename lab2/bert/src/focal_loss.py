from torch import nn
import torch
from torch.nn import functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target):
        # convert output to pseudo probability
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1 - probs, self.gamma)

        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            focal_loss = (focal_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss
