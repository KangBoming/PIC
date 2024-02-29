import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss_L2(nn.Module):
    """Focal Loss Function with Weight Decay"""
    def __init__(self, gamma=0, pos_weight=1, weight_decay=0, logits=False, reduction='sum', name='FocalLoss'):
        super(FocalLoss_L2, self).__init__()
        self.name = name
        self.gamma = gamma
        self.weight = pos_weight
        self.weight_decay = weight_decay
        self.logits = logits
        self.reduce = reduction
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_loss = BCE_loss * ((1 - pt) ** self.gamma)
        weight = self.weight * targets + 1 - targets
        focal_loss = weight * focal_loss
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        loss = focal_loss + self.weight_decay * l2_reg
        if self.reduce == 'mean':
            return torch.mean(loss)
        elif self.reduce == 'sum':
            return torch.sum(loss)
        else:
            return loss