from pathlib import Path
from typing import Literal

from src.utils.matt_heatmap import UnityMakeHeatmaps
from src.pressure_damping.image_to_image_dataset import UnityImageToImageDataset
from src.utils.visualization import visualize_heatmap

import torch
import torchvision

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import segmentation_models_pytorch as smp

from kornia.geometry.conversions import normalize_homography

from loguru import logger

from tap import Tap
import dill

from src.pressure_damping.image_to_image_train import ImageToImageTrainConfig, ImageToImageTrainRunner

POSSIBLE_PROJECTS = [
    'imp-coro-flow-inv',
    'imp-coro-shunshin-sdy-flow-bad',
    'imp-coro-shunshin-sdy-flow-good',
    'imp-coro-shunshin-sdy-flow-diff',
    'imp-coro-seligman-expert-first-beat',
    'imp-coro-shunshin-sdy-validation-expert'
]

class Arguments(Tap):
    projects: list[str] = ['imp-coro-flow-inv', 'imp-coro-shunshin-sdy-flow-good', 'imp-coro-shunshin-sdy-flow-bad', 'imp-coro-shunshin-sdy-flow-diff']
    """
    The projects to be used for training.
    """

    crop_shape: tuple[int, int] | None = None
    """
    The shape to crop the images to. If none, no cropping is done.
    """

    output_shape: tuple[int, int] = (512, 512)
    """
    The shape to resize the images to.
    """

    firebase_path: Path = Path('.firebase.json')
    """
    The path to the firebase authentication file.
    """

    num_epochs: int = 100
    """
    The number of epochs to train for.
    """

    model: Literal['unet', 'deeplabv3'] = 'unet' 
    """
    The model to use.
    """

    optimizer: Literal['adam'] = 'adam'

    loss: Literal['mse'] = 'mse'

    scheduler: Literal['step'] = 'step'


def get_model(args: Arguments) -> torch.nn.Module:
    if args.model == 'unet':
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None, 
            in_channels=1,
            classes=1, 
            activation=None
        )
    elif args.model == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation=None
        )
    else:
        raise ValueError(f'Invalid model: {args.model}')
    return model


if __name__ == '__main__':
    args = Arguments().parse_args()

    logger.info(f'Using projects: "{args.projects}"')

    dataset = UnityImageToImageDataset(
        project_codes=args.projects,
        crop_shape=args.crop_shape,
        output_shape=args.output_shape,
        firebase_certificate=Path('.firebase.json'),
        debug_mode=False
    )

    heatmap_gen = UnityMakeHeatmaps(
        keypoint_names=['curve-flow'],
        image_crop_size=dataset.final_shape,
        image_out_size=dataset.output_shape,
        heatmap_scale_factors=[1]
    )

    epochs = args.num_epochs

    model = get_model(args)

    config = ImageToImageTrainConfig(
        train_ds=dataset,
        tune_ds=None,
        heatmap_generator=heatmap_gen,
        model=model,
        optimizer=args.optimizer,
        loss_fn=args.loss,
        scheduler=args.scheduler,
        batch_size=10,
        num_epochs=epochs,
    )

    runner = ImageToImageTrainRunner(config)

    runner.run()