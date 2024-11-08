import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, model, weight=None, **kwargs):
        super().__init__()

    def forward(self, input, target):
        return F.mse_loss(input.view(-1), target.view(-1), reduction='mean')


class QLoss(nn.Module):
    """
    Regression loss that minimizes the q-error for each prediction
    """

    def __init__(self, model, weight=None, min_val=1e-3, penalty_negative=1e5,
                 **kwargs):
        self.min_val = min_val
        self.penalty_negative = penalty_negative
        super().__init__()

    def forward(self, input, target):
        input_zero_mask = input == 0
        target_zero_mask = target == 0

        # add small value to zero entries to avoid division by zero
        if input_zero_mask.any():
            input = input + input_zero_mask * torch.full(input.shape, 0.0000001, device=input.device)
        if target_zero_mask.any():
            target = target + target_zero_mask * torch.full(input.shape, 0.0000001, device=target.device)

        q_error = torch.zeros((len(target), 1), device=target.device)

        # create mask for entries which should be penalized for negative/too small estimates
        penalty_mask = input < self.min_val
        inverse_penalty_mask = input >= self.min_val
        q_error_penalties = torch.mul(1 - input, penalty_mask) * self.penalty_negative

        # influence on loss for a negative estimate is >= penalty_negative constant
        q_error = torch.add(q_error, q_error_penalties)

        # calculate normal q error for other instances
        input_masked = torch.mul(input, inverse_penalty_mask)
        target_masked = torch.mul(target.reshape((-1, 1)), inverse_penalty_mask)

        q_error = torch.add(q_error, torch.max(torch.div(input_masked, target.reshape((-1, 1))),
                                               torch.div(target_masked, input)))

        loss = torch.mean(q_error)

        return loss


class ProcentualLoss(nn.Module):
    def __init__(self, model, weight=None):
        super().__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(input - target) / target)
