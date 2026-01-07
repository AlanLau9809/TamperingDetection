#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial

import torch
import torch.nn as nn

# Try to import from pytorchvideo, fall back to local implementation
try:
    from pytorchvideo.losses.soft_target_cross_entropy import SoftTargetCrossEntropyLoss
except ImportError:
    # Provide local implementation of SoftTargetCrossEntropyLoss
    class SoftTargetCrossEntropyLoss(nn.Module):
        """
        Soft target cross entropy loss with optional target normalization.
        """
        def __init__(self, normalize_targets=True):
            super(SoftTargetCrossEntropyLoss, self).__init__()
            self.normalize_targets = normalize_targets

        def forward(self, input, target):
            """
            Args:
                input (torch.Tensor): Logits from model [N, C]
                target (torch.Tensor): Soft targets [N, C] with values in [0, 1]
            """
            if self.normalize_targets:
                # Normalize targets to sum to 1
                target = torch.nn.functional.softmax(target, dim=1)

            # KL divergence loss between input (logits) and target (soft labels)
            log_probs = torch.nn.functional.log_softmax(input, dim=1)
            loss = torch.sum(-target * log_probs, dim=1).mean()

            return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(inputs, targets)
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(SoftTargetCrossEntropyLoss, normalize_targets=False),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
