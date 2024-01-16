import torch
import torch.nn as nn

from src.utils.matt_heatmap import UnityMakeHeatmaps, transform_image

class PretransformationsModule(nn.Module):
    make_heatmaps: UnityMakeHeatmaps

    def __init__(self, 
                 keypoint_names: list[str], 
                 image_crop_size: tuple[int, int],
                 image_out_size: tuple[int, int]) -> None:
        super(PretransformationsModule, self).__init__()

        self.make_heatmaps = UnityMakeHeatmaps(keypoint_names=keypoint_names, 
                                               image_crop_size=image_crop_size, 
                                               image_out_size=image_out_size)

    
    def forward(self, 
                label_data: list[dict[str, any]],
                images: torch.Tensor,
                dataset_transform_matrices: list[torch.Tensor],
                label_height_shift: list[int] | None = None, 
                label_width_shift: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """

        if label_height_shift is None:
            label_height_shift = [0] * len(label_data)
        
        if label_width_shift is None:
            label_width_shift = [0] * len(label_data)

        # Make the heatmaps.
        heatmaps, weights = self.make_heatmaps(
            label_data=label_data,
            label_height_shift=label_height_shift,
            label_width_shift=label_width_shift,
            transform_matrix=dataset_transform_matrices
        )

        return images, heatmaps, weights




        

                                
