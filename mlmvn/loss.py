# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_loss.ipynb.

# %% auto 0
__all__ = ['ComplexMSELoss']

# %% ../nbs/01_loss.ipynb 3
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

# %% ../nbs/01_loss.ipynb 4
class ComplexMSELoss(Function):
    
    @staticmethod
    def forward(ctx, y_pred, y, categories, periodicity):    
        y_tmp = y
        if y.size().__len__() == 1:
            y_tmp = torch.unsqueeze(y, 1)

        # Class to several angles due to periodicity using bisector
        target_angles = (
            (categories * torch.arange(periodicity) + y_tmp + 0.5)
            / (categories * periodicity)
            * 2
            * np.pi
        )
        
        predicted_angles = torch.remainder(y_pred.angle() + 2 * np.pi, 2 * np.pi)
        # errors = torch.exp(1.0j * target_angles) - torch.exp(1.0j * predicted_angles.unsqueeze(1))
        errors = torch.exp(1.0j * target_angles) - torch.exp(1.0j * predicted_angles)
        loss_angle = target_angles - predicted_angles


        if periodicity > 1:
            # select smallest error
            idx = torch.argmin(torch.abs(errors), dim=1, keepdim=True)
            errors = errors.gather(1, idx)

            idx = torch.argmin(torch.abs(loss_angle), dim=1, keepdim=True)
            loss_angle = loss_angle.gather(1, idx)

        

        ctx.save_for_backward(y_pred, y, errors)
        # return errors.mean()
        return torch.mean(
            torch.square(
                np.pi - torch.abs(torch.abs(loss_angle) - np.pi)
            )
        ) 
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y, errors = ctx.saved_tensors
        grad_input = errors
        if y_pred.shape != errors.shape: grad_input = errors.squeeze()
        return grad_input, None, None, None
