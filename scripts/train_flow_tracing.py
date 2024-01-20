from pathlib import Path

from src.utils.matt_heatmap import UnityMakeHeatmaps
from src.pressure_damping.image_to_image_dataset import UnityImageToImageDataset
from src.utils.visualization import visualize_heatmap


import torch
import torchvision

from kornia.geometry.conversions import normalize_homography

from loguru import logger

POSSIBLE_PROJECTS = [
    'imp-coro-flow-inv',
    'imp-coro-shunshin-sdy-flow-bad',
    'imp-coro-shunshin-sdy-flow-good',
    'imp-coro-shunshin-sdy-flow-diff',
    'imp-coro-seligman-expert-first-beat',
    'imp-coro-shunshin-sdy-validation-expert'
]

if __name__ == '__main__':
    project = POSSIBLE_PROJECTS[2]

    logger.info(f'Using project "{project}"')

    dataset = UnityImageToImageDataset(
        project_codes=[project],
        crop_shape=(512, 512),
        output_shape=(512, 512),
        firebase_certificate=Path('.firebase.json'),
        debug_mode=True
    )