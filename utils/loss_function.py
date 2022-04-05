import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self, eps=1e-8, reduction="none"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logit, target):
        
        target_diff = target - torch.mean(target)
        logit_diff = logit - torch.mean(logit)
        
        loss = 1 - torch.sum(target_diff * logit_diff) / (torch.sqrt(torch.sum(target_diff ** 2)) * torch.sqrt(torch.sum(logit_diff ** 2)))

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError


class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=1e-3, reduction="none"):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, logit, target):

        abs_target = torch.abs(target).clamp(1e-8)

        # where target is too small, use just divide by threshold to avoid divide by 0
        loss = torch.where(abs_target < self.threshold, torch.square(target - logit),
                           torch.square(target - logit) / abs_target)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError
