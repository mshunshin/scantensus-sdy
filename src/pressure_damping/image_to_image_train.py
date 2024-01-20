from dataclasses import dataclass
import os

import torch

from pathlib import Path
from typing import Dict, Literal, Optional, List, OrderedDict, Tuple

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

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

from src.pressure_damping.image_to_image_dataset import UnityImageToImageDataset

@dataclass
class ImageToImageTrainConfig:
    """
    Configuration for training an image-to-image model.
    """

    train_ds: UnityImageToImageDataset
    """
    The training dataset.
    """
    
    tune_ds: UnityImageToImageDataset
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

    optimizer: Optimizer
    """
    The optimizer to use.
    """

    loss_fn: torch.nn.Module
    """
    The loss function to use.
    """

    scheduler: LRScheduler
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

        logger.info(f'Training model for {self.config.epochs} epochs with batch size {self.config.batch_size},')
        logger.info(f'on {self.config.world_size} GPUs.')

        children: List[mp.Process] = []

        MAIN_QUEUE = mp.get_context('spawn').SimpleQueue()

        for i in range(self.config.world_size):
            process = mp.Process(
                target=ImageToImageTrainRunner._threaded_main,
                args=(i, self.config, MAIN_QUEUE)
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
    def _threaded_main(rank: int, config: ImageToImageTrainConfig, queue: mp.SimpleQueue):
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

        tune_sampler = DistributedSampler(config.tune_ds, shuffle=True)

        tune_dataloader = DataLoader(config.tune_ds,
                                      batch_size=config.batch_size,
                                      num_workers=2,
                                      pin_memory=False,
                                      sampler=tune_sampler,
                                      drop_last=False)

        if rank == 0:
            logger.info(f'Loaded tune dataset with {len(config.val_ds)} entries in {len(tune_dataloader)} batches.')


        

        