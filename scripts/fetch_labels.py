from pathlib import Path

from src.pressure_damping.label_processer import FirebaseCurveFetcher
from src.utils.cluster_source import ClusterSource

from src.utils.matt_heatmap import UnityMakeHeatmaps
from src.pressure_damping.unity_dataset import UnityDataset
from src.pressure_damping.curve_dataset import CurveDataset

import numpy as np
import torch


if __name__ == '__main__':
    dataset = CurveDataset(
        projects=['imp-coro-flow-inv'],
        firebase_certificate=Path('.firebase.json')
    )

    image, raw_labels = dataset[0]

    H = image.shape[1]
    W = image.shape[2]

    heatmap_gen = UnityMakeHeatmaps(
        keypoint_names=['curve-flow'],
        image_crop_size=(H, W),
        image_out_size=(H, W)
    )

    heatmap_gen(
        label_data=[raw_labels],
        label_height_shift=[0],
        label_width_shift=[0],
        transform_matrix=[torch.eye(3)]
    )

    pass

    



    