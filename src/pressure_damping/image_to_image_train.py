from dataclasses import dataclass
import os

import torch

from pathlib import Path
from typing import Dict, Literal, Optional, List, OrderedDict, Tuple

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.nn import MSELoss

import torchvision

import torch.multiprocessing as mp
import torch.distributed
import torch.backends.cudnn

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.utils.distributed import reduce_tensor

from loguru import logger

import torch.distributed as dist

from src.pressure_damping.image_to_image_dataset import UnityImageToImageDataset, Curve
from src.utils.matt_heatmap import UnityMakeHeatmaps

import dill

@dataclass
class ImageToImageTrainConfig:
    """
    Configuration for training an image-to-image model.
    """

    train_ds: UnityImageToImageDataset
    """
    The training dataset.
    """
    
    tune_ds: UnityImageToImageDataset | None
    """
    The tuning dataset.
    """

    heatmap_generator: Module
    """
    The heatmap generator to use.
    """

    model: Module
    """
    The model to train.
    """

    optimizer: str
    """
    The optimizer to use.
    """

    loss_fn: str
    """
    The loss function to use.
    """

    scheduler: str
    """
    The learning rate scheduler to use.
    """

    batch_size: int
    """
    The batch size to use.
    """

    num_epochs: int
    """
    The number of epochs to train for.
    """

    world_size: int = torch.cuda.device_count()
    """
    The number of GPUs to use.
    """

    random_seed: int = 42
    """
    The random seed to use.
    """
    

class ImageToImageTrainRunner:
    """
    Runs training for an image-to-image model.
    """

    config: ImageToImageTrainConfig

    def __init__(self, config: ImageToImageTrainConfig):
        self.config = config


    def run(self):
        """
        Runs training.
        """

        mp.set_start_method('spawn', force=True)

        logger.info(f"Starting training with {self.config.world_size} GPUs.")

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        torch.cuda.empty_cache()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "52388"

        logger.info(f'Training model for {self.config.num_epochs} epochs with batch size {self.config.batch_size},')
        logger.info(f'on {self.config.world_size} GPUs.')

        children: List[mp.Process] = []

        MAIN_QUEUE = mp.get_context('spawn').SimpleQueue()

        packed_config = dill.dumps(self.config, byref=False, recurse=True)

        for i in range(self.config.world_size):
            process = mp.Process(
                target=ImageToImageTrainRunner._threaded_main,
                args=(i, packed_config, MAIN_QUEUE)
            )

            children.append(process)
            process.start()

        for i in range(self.config.world_size):
            children[i].join()

        logger.success('Training complete.')

        if not MAIN_QUEUE.empty():
            try:
                out = MAIN_QUEUE.get()

                logger.trace(f'Got output from queue: {out}')
            except Exception as e:
                logger.warning(f'Could not get elements from queue: {e}')
        else:
            logger.warning('Queue is empty.')

        logger.success('Collecting complete.')


    @staticmethod
    def _threaded_main(rank: int, config_data: bytes, queue: mp.SimpleQueue):
        config: ImageToImageTrainConfig = dill.loads(config_data)

        torch.cuda.set_device(rank)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed(config.random_seed)

        # initialize the process group
        torch.distributed.init_process_group("nccl", rank=rank, world_size=config.world_size)

        logger.info(f'GPU #{rank} initialized.')

        train_sampler = DistributedSampler(config.train_ds, shuffle=True)
        train_dataloader = DataLoader(config.train_ds,
                                      batch_size=config.batch_size,
                                      num_workers=4,
                                      pin_memory=False,
                                      sampler=train_sampler,
                                      drop_last=False)

        if rank == 0:
            logger.info(f'Loaded train dataset with {len(config.train_ds)} entries in {len(train_dataloader)} batches.')                                    

        if config.tune_ds is not None:
            tune_sampler = DistributedSampler(config.tune_ds, shuffle=True)

            tune_dataloader = DataLoader(config.tune_ds,
                                      batch_size=config.batch_size,
                                      num_workers=4,
                                      pin_memory=False,
                                      sampler=tune_sampler,
                                      drop_last=False)

            if rank == 0:
                logger.info(f'Loaded tune dataset with {len(config.tune_ds)} entries in {len(tune_dataloader)} batches.')

        model = config.model.to(rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DDP(model, device_ids=[rank])

        heatmap_generator = config.heatmap_generator.to(rank)

        # TODO(guilherme): move to a function
        loss_fn = MSELoss().to(rank) if config.loss_fn == 'mse' else None
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if config.optimizer == 'adam' else None
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if config.scheduler == 'step' else None

        for epoch in range(config.num_epochs + 1):
            if rank == 0:
                logger.info(f'Epoch {epoch} starting.')

            train_loss = ImageToImageTrainRunner._threaded_train(
                rank=rank,
                model=model,
                heatmap_generator=heatmap_generator,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=train_dataloader
            )


    @staticmethod
    def _threaded_train(
        rank: int,
        model: DDP,
        heatmap_generator: UnityMakeHeatmaps,
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataloader: DataLoader
    ) -> float:
        """
        Trains the model for one epoch. Returns the average loss.
        """

        model.train()

        total_loss: float = 0.0

        for batch_idx, (images, transform, (height_shift, width_shift), labels) in enumerate(train_dataloader):
            # move data to GPU
            images = images.to(rank)
            transform = transform.to(rank)

            # generate heatmaps
            heatmaps = heatmap_generator.forward(
                label_data=labels,
                label_height_shift=height_shift,
                label_width_shift=width_shift,
                transform_matrix=transform
            )

            pass

        return total_loss / len(train_dataloader)

        