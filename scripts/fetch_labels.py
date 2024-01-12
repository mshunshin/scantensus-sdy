from pathlib import Path

from src.pressure_damping.label_processer import FirebaseCurveFetcher
from src.utils.cluster_source import ClusterSource

from src.utils.matt_heatmap import UnityMakeHeatmaps
from src.pressure_damping.unity_dataset import UnityDataset
from src.pressure_damping.curve_dataset import CurveDataset


if __name__ == '__main__':
    dataset = CurveDataset(
        projects=['imp-coro-flow-inv'],
        firebase_certificate=Path('.firebase.json')
    )

    heatmap_gen = UnityMakeHeatmaps(
        keypoint_names=['curve-flow'],
        image_crop_size=(512, 512),
        image_out_size=(512, 512)
    )

    image = dataset[0]

    pass

    



    