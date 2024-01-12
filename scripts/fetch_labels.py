from pathlib import Path

from src.pressure_damping.label_processer import FirebaseCurveFetcher
from src.utils.cluster_source import ClusterSource

from src.utils.matt_heatmap import UnityMakeHeatmaps
from src.pressure_damping.curve_dataset import CurveDataset
from src.pressure_damping.pretransformations import PretransformationsModule

import numpy as np
import torch


if __name__ == '__main__':
    dataset = CurveDataset(
        projects=['imp-coro-flow-inv'],
        output_shape=(170, 422),
        firebase_certificate=Path('.firebase.json')
    )

    image, raw_labels, T = dataset[0]

    H = image.shape[1]
    W = image.shape[2]

    heatmap_gen = UnityMakeHeatmaps(
        keypoint_names=['curve-flow'],
        image_crop_size=(H, W),
        image_out_size=(H, W)
    )

    """
    heatmap_gen(
        label_data=[raw_labels],
        label_height_shift=[0],
        label_width_shift=[0],
        transform_matrix=[T]
    )
    """

    pretransforms = PretransformationsModule(
        keypoint_names=['curve-flow'],
        image_crop_size=(170, 422),
        image_out_size=(170, 422)
    )

    images, heatmaps, weights = pretransforms.forward(
        label_data=[raw_labels],
        images=image.unsqueeze(0),
        dataset_transform_matrix=T.unsqueeze(0),
    )

    pass

    



    