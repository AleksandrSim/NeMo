# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import LogitsType, LossType, NeuralType, ChannelType
from nemo.utils import logging

__all__ = ['DiceLoss']


class DiceLoss(Loss, Typing):
    """
    Neural module which implements Dice Loss for segmentation models.
    This loss function is suitable for both binary and multi-class segmentation tasks.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
            inputs: Predicted probabilities/logits from the model
            targets: Ground truth segmentation masks
        """
        return {
            "inputs": NeuralType(('B', 'C', 'H', 'W'), LogitsType()),
            "targets": NeuralType(('B', 'C', 'H', 'W'), LogitsType()),
        }

    @property
    def output_types(self):
        """
        Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            smooth (float): Smoothing constant for numerical stability (default: 1.0)
            reduction: specifies the reduction to apply to the final loss, choose 'mean' or 'sum'
        """
        super().__init__()
        self.smooth = smooth

        if reduction not in ['mean', 'sum']:
            logging.warning(f'{reduction} reduction is not supported. Setting reduction to "mean".')
            reduction = 'mean'

        self.reduction = reduction

    def _dice_coefficient(self, inputs, targets):
        """
        Compute the Dice coefficient for each batch.
        Args:
            inputs (torch.Tensor): Predicted logits from the model (NxCxHxW)
            targets (torch.Tensor): Ground truth segmentation masks (NxCxHxW)
        Returns:
            torch.Tensor: Dice coefficient for each class.
        """
        intersection = torch.sum(inputs * targets, dim=(2, 3))  # Sum over height and width
        input_sum = torch.sum(inputs, dim=(2, 3))  # Sum over height and width for inputs
        target_sum = torch.sum(targets, dim=(2, 3))  # Sum over height and width for targets
        dice_score = (2 * intersection + self.smooth) / (input_sum + target_sum + self.smooth)
        return dice_score

    @typecheck()
    def forward(self, inputs, targets):
        """
        Forward pass for Dice loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities/logits from the model (NxCxHxW)
            targets (torch.Tensor): Ground truth segmentation masks (NxCxHxW)
        """
        # Apply softmax for multi-class segmentation
        inputs = torch.softmax(inputs, dim=1)

        # Compute the Dice coefficient
        dice_score = self._dice_coefficient(inputs, targets)

        # Dice loss is 1 - Dice coefficient
        dice_loss = 1 - dice_score

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

# Test the implementation
if __name__ == "__main__":
    inputs = torch.randn(2, 3, 256, 256)  # Example input batch (N=2, C=3, H=256, W=256)
    targets = torch.randn(2, 3, 256, 256)  # Example target batch

    dice_loss_fn = DiceLoss()
    loss = dice_loss_fn(inputs=inputs, targets=targets)
    print(f"Dice Loss: {loss.item()}")
