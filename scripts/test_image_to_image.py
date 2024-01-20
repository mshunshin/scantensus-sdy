from pathlib import Path

from src.pressure_damping.label_processer import FirebaseCurveFetcher
from src.utils.cluster_source import ClusterSource

from src.utils.matt_heatmap import UnityMakeHeatmaps
from src.pressure_damping.image_to_image_dataset import UnityImageToImageDataset
from src.pressure_damping.pretransformations import PretransformationsModule
from src.utils.visualization import visualize_heatmap

import numpy as np
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

    image, T, (height_shift, width_shift), labels = dataset[0]

    heatmap_gen = UnityMakeHeatmaps(
        keypoint_names=['curve-flow'],
        image_crop_size=dataset.crop_shape,
        image_out_size=dataset.output_shape,
        heatmap_scale_factors=[1, 2, 4]
    )

    heatmaps, weights = heatmap_gen.forward(
        label_data=[labels],
        transform_matrix=[normalize_homography(T.inverse(), dsize_src=dataset.output_shape, dsize_dst=dataset.output_shape)],
        label_height_shift=[height_shift],
        label_width_shift=[width_shift]
    )

    # save heatmaps[0] as heatmap.png
    torchvision.utils.save_image(heatmaps[0].to(torch.float32), 'out/heatmap.png')

    heatmap_viz = visualize_heatmap(image, heatmaps[0].squeeze())
    heatmap_viz.save('out/heatmap_viz.png')

    pass

    



    