# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_linear_layer,
    get_params_for_weight_decay_optimization,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank, torch_dtype_from_precision
from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomSampler
from nemo.collections.vision.data.megatron.vit_dataset import build_train_valid_datasets
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

# Import custom loss function from dice_loss.py
from nemo.collections.vision.losses.dice_loss import DiceLoss

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches


class VitSegmentationModel(MegatronModule):
    """Vision Transformer Segmentation Model with dynamic upsampling."""

    def __init__(
        self, model_cfg, model_parallel_config, num_classes, input_size, target_size, finetune=False, pre_process=True, post_process=True
    ):
        super(VitSegmentationModel, self).__init__()

        # Initialize backbone and layers
        self.hidden_size = model_cfg.hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.target_size = target_size
        self.finetune = finetune
        self.pre_process = pre_process
        self.post_process = post_process

        # Vit backbone (encoder)
        self.backbone = VitBackbone(
            model_cfg,
            model_parallel_config,
            init_method=init_method_normal(model_cfg.init_method_std),
            pre_process=self.pre_process,
            post_process=False,  # We disable post_process in the backbone for segmentation
        )

        # Dynamic upsampling block
        self.upsample_blocks = self._make_upsample_blocks(self.hidden_size, self.target_size)

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(self.hidden_size, self.num_classes, kernel_size=1)

    def _make_upsample_blocks(self, input_dim, target_size):
        """
        Create dynamic upsampling blocks to upscale the output from transformer to the target size.
        """
        upsample_blocks = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, padding=1),
            nn.Upsample(size=(target_size[0] // 4, target_size[1] // 4), mode='bilinear', align_corners=True),
            nn.Conv2d(input_dim // 2, input_dim // 4, kernel_size=3, padding=1),
            nn.Upsample(size=(target_size[0] // 2, target_size[1] // 2), mode='bilinear', align_corners=True),
            nn.Conv2d(input_dim // 4, input_dim // 8, kernel_size=3, padding=1),
            nn.Upsample(size=target_size, mode='bilinear', align_corners=True),
        )
        return upsample_blocks

    def forward(self, x):
        # Pass through Vit backbone (encoder)
        hidden_states = self.backbone(x)

        # Upsampling to match target size
        upsampled = self.upsample_blocks(hidden_states)

        # Pass through segmentation head
        seg_map = self.segmentation_head(upsampled)

        return seg_map


class MegatronVitSegmentationModel(MegatronBaseModel):
    """Megatron Vision Transformer Model for Segmentation."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        super().__init__(cfg, trainer=trainer)

        self._validate_trainer()

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        # Build segmentation model
        self.model = self._build_segmentation_model(cfg)

        # AMP and autocasting settings
        self.autocast_dtype = torch_dtype_from_precision(self.trainer.precision)
        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        # Initialize the custom loss function (Dice Loss)
        self.loss_fn = DiceLoss()

    def _build_segmentation_model(self, cfg):
        """Build the segmentation model with dynamic upsampling."""
        return VitSegmentationModel(
            model_cfg=cfg,
            model_parallel_config=self.model_parallel_config,
            num_classes=cfg.num_classes,
            input_size=(cfg.img_h, cfg.img_w),
            target_size=(cfg.target_h, cfg.target_w),
            finetune=cfg.get("finetune", False),
            pre_process=True,
            post_process=True
        )

    def forward(self, images):
        # Forward pass through the segmentation model
        return self.model(images)

    def training_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        Batch should be a list of microbatches and those microbatches should be on CPU.
        Microbatches are then moved to GPU during the pipeline.
        The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        loss_mean, _ = self.fwd_bwd_step(dataloader_iter, forward_only=False)

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            self._optimizer._finish_bucket_grad_sync()

            # launch grad reductions
            if not parallel_state.is_pipeline_first_stage():
                self.reduce_overlap_gradients()
        elif self.megatron_amp_O2:
            # # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            self._optimizer.allreduce_main_grads()
        else:
            self.allreduce_gradients()

        # Log the loss and LR
        torch.distributed.broadcast(loss_mean, get_last_rank())
        self.log('train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        self.log('lr', self._optimizer.param_groups[0]['lr'], rank_zero_only=True)

        return loss_mean

    def validation_step(self, dataloader_iter):
        loss_mean, _ = self.fwd_bwd_step(dataloader_iter, forward_only=True)

        if parallel_state.is_pipeline_last_stage():
            torch.distributed.broadcast(loss_mean, get_last_rank())
            self.log('val_loss', loss_mean, prog_bar=True, rank_zero_only=True)

        return loss_mean

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        fwd_bwd_function = get_forward_backward_func()
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.cfg.encoder_seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )
        loss_tensor = torch.stack([loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]).mean()
        return loss_tensor, None

    def get_forward_output_and_loss_func(self):
        def loss_func(labels, output_tensor):
            logits = output_tensor.contiguous().float()
            loss = self.loss_fn(logits, labels)
            return loss

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            images, labels = batch[0].cuda(), batch[1].cuda()
            output_tensor = model(images)
            return output_tensor, partial(loss_func, labels)

        return fwd_output_and_loss_func

    def setup_optimizer_param_groups(self):
        if self.cfg.get('do_layer_norm_weight_decay', False):
            self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):
        return super().configure_optimizers()

    def setup(self, stage=None):
        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for ViT Segmentation...')
        self._train_ds, self._validation_ds = build_train_valid_datasets(
            model_cfg=self.cfg,
            data_path=self.cfg.data.data_path,
            image_size=(self.cfg.img_h, self.cfg.img_w),
        )
        self._test_ds = None
        logging.info(f'Finished building datasets for ViT Segmentation.')
        return self._train_ds, self._validation_ds, self._test_ds
    

