import torch

import json
import logging

import numpy as np
import scipy
import math

from typing import Sequence, Optional

import distutils

def curve_list_to_str(curve_list: list[float], round_digits=1):
    out = " ".join([str(round(value, round_digits)) for value in curve_list])
    return out


def curve_str_to_list(curve_str: str):
    out = [(float(value)) for value in curve_str.split()]
    return out


def curve_str_to_list_int(curve_str: str):
    out = [(int(value)) for value in curve_str.split()]
    return out


def bool_list_to_str(curve_list: list[bool]):
    out = " ".join([str(value) for value in curve_list])
    return out


def bool_str_to_list(curve_str: str):
    out = [(bool(distutils.util.strtobool(value))) for value in curve_str.split()]
    return out

class UnityMakeHeatmaps(torch.nn.Module):

    def __init__(self,
                 keypoint_names,
                 image_crop_size,
                 image_out_size,
                 heatmap_scale_factors=(2, 4),
                 dot_sd=4,
                 curve_sd=2,
                 dot_weight_sd=40,
                 curve_weight_sd=10,
                 dot_weight=40,
                 curve_weight=10,
                 sub_pixel=True,
                 single_weight=True,
                 device="cpu"):

        super().__init__()

        self.keypoint_names = keypoint_names
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.heatmap_scale_factors = heatmap_scale_factors
        self.dot_sd = dot_sd
        self.curve_sd = curve_sd
        self.dot_weight_sd = dot_weight_sd
        self.curve_weight_sd = curve_weight_sd
        self.dot_weight = dot_weight
        self.curve_weight = curve_weight
        self.sub_pixel = sub_pixel
        self.single_weight = single_weight
        self.device = device

    def forward(self,
                label_data,
                label_height_shift,
                label_width_shift,
                transform_matrix):

        batch_size = len(transform_matrix)

        out_heatmaps = []
        out_weights = []
        for scale_factor in self.heatmap_scale_factors:
            heatmaps_batch = []
            weights_batch = []
            for i in range(batch_size):
                heatmaps, weights = make_labels_and_masks(image_in_size=self.image_crop_size,
                                                          image_out_size=self.image_out_size,
                                                          label_data=label_data[i],
                                                          keypoint_names=self.keypoint_names,
                                                          label_height_shift=label_height_shift[i],
                                                          label_width_shift=label_width_shift[i],
                                                          heatmap_scale_factor=scale_factor,
                                                          transform_matrix=transform_matrix[i],
                                                          dot_sd=self.dot_sd,
                                                          curve_sd=self.curve_sd,
                                                          dot_weight_sd=self.dot_weight_sd,
                                                          curve_weight_sd=self.curve_weight_sd,
                                                          dot_weight=self.dot_weight,
                                                          curve_weight=self.curve_weight,
                                                          sub_pixel=self.sub_pixel,
                                                          single_weight=self.single_weight,
                                                          device=self.device)
                heatmaps_batch.append(heatmaps)
                weights_batch.append(weights)
            out_heatmaps.append(torch.stack(heatmaps_batch))
            out_weights.append(torch.stack(weights_batch))

        return out_heatmaps, out_weights


def make_labels_and_masks(image_in_size,
                          image_out_size,
                          keypoint_names,
                          label_data,
                          label_height_shift=0,
                          label_width_shift=0,
                          transform_matrix=None,
                          heatmap_scale_factor=1,
                          dot_sd=4,
                          curve_sd=2,
                          dot_weight_sd=40,
                          curve_weight_sd=10,
                          dot_weight=40,
                          curve_weight=10,
                          sub_pixel=True,
                          single_weight=True,
                          device="cpu"):

    # if you are using in a different thread, e.g. a dataloader and device must be cpu.

    device = torch.device(device)

    num_keypoints = len(keypoint_names)

    if transform_matrix is not None:
        transform_matrix = transform_matrix.to(device)
        target_out_height = image_out_size[0] // heatmap_scale_factor
        target_out_width = image_out_size[1] // heatmap_scale_factor
        image_out_size_t = torch.tensor(image_out_size, dtype=torch.float, device=device)
    else:
        target_out_height = image_in_size[0] // heatmap_scale_factor
        target_out_width = image_in_size[1] // heatmap_scale_factor
        image_out_size_t = None

    image_in_size_t = torch.tensor(image_in_size, dtype=torch.float, device=device)

    heatmaps = torch.zeros((num_keypoints, target_out_height, target_out_width), device=device, dtype=torch.uint8)

    if not single_weight:
        weights = torch.zeros((num_keypoints, target_out_height, target_out_width), device=device, dtype=torch.uint8)
    else:
        weights = torch.zeros((num_keypoints, 1, 1), device=device, dtype=torch.uint8)

    if type(label_data) is str:
        label_data = json.loads(label_data)

    if label_data is None:
        logging.error(f"No label data supplied")
        return heatmaps, weights

    # label_data: dict[keypoint_name: str, list[dict[key: str, value: any]]]

    for keypoint_idx, keypoint in enumerate(keypoint_names):
        try:
            keypoint_data = label_data.get(keypoint, None)
        except Exception:
            logging.exception(f"{keypoint} - {label_data}")

        if keypoint_data is None:
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        label_type = keypoint_data[0]['type']

        if label_type is None:
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        if label_type == "off":
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 1
            continue

        if label_type == "blurred":
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        heatmaps[keypoint_idx, ...] = 0
        weights[keypoint_idx, ...] = 0

        for instance_num, instance_data in enumerate(keypoint_data):

            y_data = instance_data["y"]
            x_data = instance_data["x"]
            straight_segment = instance_data.get("straight_segment", None)
            if straight_segment is not None:
                straight_segment = straight_segment

            if type(y_data) is str:
                ys = curve_str_to_list(y_data)
            else:
                ys = y_data

            if type(x_data) is str:
                xs = curve_str_to_list(x_data)
            else:
                xs = x_data

            if type(straight_segment) is str:
                straight_segments = bool_str_to_list(straight_segment)
            else:
                straight_segments = straight_segment

            if len(ys) != len(xs) or not all(np.isfinite(ys)) or not all(np.isfinite(xs)) or len(ys) == 0:
                print(f"problem with data {keypoint}, {ys}, {xs}")
                heatmaps[keypoint_idx, ...] = 0
                weights[keypoint_idx, ...] = 0
                continue

            coord = torch.tensor([ys, xs], device=device).transpose(0, 1)
            label_shift = torch.tensor([label_height_shift, label_width_shift], device=device)
            coord = coord + label_shift

            if transform_matrix is not None:
                coord = normalize_coord(coord=coord, image_size=image_in_size_t)
                coord = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=coord)
                coord = unnormalize_coord(coord=coord, image_size=image_out_size_t)
            else:
                coord = coord.unsqueeze(0)

            coord = coord / heatmap_scale_factor

            dot_sd_t = torch.tensor([dot_sd, dot_sd], dtype=torch.float, device=device)
            dot_weight_sd_t = torch.tensor([dot_weight_sd, dot_weight_sd], dtype=torch.float, device=device)

            if "flow" not in keypoint:
                curve_sd_t = torch.tensor([curve_sd, curve_sd], dtype=torch.float, device=device)
                curve_weight_sd_t = torch.tensor([curve_weight_sd, curve_weight_sd], dtype=torch.float, device=device)
            else:
                curve_sd_t = torch.tensor([curve_sd*2, 0.5], dtype=torch.float, device=device)
                curve_weight_sd_t = torch.tensor([curve_weight_sd*2, 0.5], dtype=torch.float, device=device)

            if len(ys) == 1:
                out_heatmap = render_gaussian_dot_u(point=coord[0, 0, :],
                                                    std=dot_sd_t,
                                                    size=(target_out_height, target_out_width),
                                                    mul=255)

                if not single_weight:
                    out_weight = render_gaussian_dot_u(point=coord[0, 0, :],
                                                       std=dot_weight_sd_t,
                                                       size=(target_out_height, target_out_width),
                                                       mul=(dot_weight-1)).add(1)
                else:
                    out_weight = torch.tensor([dot_weight], device=device, dtype=torch.uint8)

            elif len(ys) >= 2:
                points_np = coord[0, :, :].cpu().numpy()
                ys = points_np[:, 0].tolist()
                xs = points_np[:, 1].tolist()

                curve_points_len = line_len(points_np)
                curve_points_len = int(curve_points_len)
                curve_points_len = max(curve_points_len, len(ys))
                out_curve_y, out_curve_x = interpolate_curveline(ys=ys, xs=xs, straight_segments=straight_segments, total_points_out=curve_points_len * 2)

                curve_points = torch.tensor([out_curve_y, out_curve_x],
                                            dtype=torch.float,
                                            device=device).T

                if sub_pixel:
                    out_heatmap = render_gaussian_curve_u(points=curve_points,
                                                          std=curve_sd_t,
                                                          size=(target_out_height, target_out_width),
                                                          mul=255).to(device)

                    if not single_weight:
                        out_weight = render_gaussian_curve_u(points=curve_points,
                                                             std=curve_weight_sd_t,
                                                             size=(target_out_height, target_out_width),
                                                             mul=(curve_weight-1)).add(1).to(device)
                    else:
                        out_weight = torch.tensor([curve_weight], device=device, dtype=torch.uint8)

                else:
                    curve_kernel_size = 2 * ((math.ceil(curve_sd) * 5) // 2) + 1
                    curve_weight_kernel_size = 2 * ((math.ceil(curve_weight_sd) * 5) // 2) + 1

                    out_heatmap = make_curve_labels(points=curve_points,
                                                    image_size=(target_out_height, target_out_width),
                                                    kernel_sd=curve_sd,
                                                    kernel_size=curve_kernel_size)

                    out_heatmap = torch.tensor(out_heatmap, device=device)
                    out_heatmap = out_heatmap.mul(255).to(torch.uint8)

                    if not single_weight:
                        out_weight = make_curve_labels(points=curve_points,
                                                       image_size=(target_out_height, target_out_width),
                                                       kernel_sd=curve_weight_sd,
                                                       kernel_size=curve_weight_kernel_size)

                        out_weight = torch.tensor(out_weight, device=device)
                        out_weight = out_weight.mul(curve_weight-1).add(1).to(torch.uint8)
                    else:
                        out_weight = torch.tensor([curve_weight], device=device, dtype=torch.uint8)

            else:
                print(f"Error - no idea what problem with data was: {ys}, {xs}")
                continue

            heatmaps[keypoint_idx, ...] = torch.max(out_heatmap, heatmaps[keypoint_idx, ...])
            weights[keypoint_idx, ...] = torch.max(out_weight, weights[keypoint_idx, ...])

    return heatmaps, weights



def normalize_coord(coord: torch.Tensor, image_size: torch.Tensor):

    coord = (coord * 2 / image_size) - 1

    return coord


def unnormalize_coord(coord: torch.Tensor, image_size: torch.tensor):

    coord = (coord + 1) * image_size / 2

    return coord


def apply_matrix_to_coords(transform_matrix: torch.Tensor, coord: torch.Tensor):

    if coord.dim() == 2:
        coord = coord.unsqueeze(0)

    batch_size = coord.shape[0]

    if transform_matrix.dim() == 2:
        transform_matrix = transform_matrix.unsqueeze(0)

    if transform_matrix.size()[1:] == (3, 3):
        transform_matrix = transform_matrix[:, :2, :]

    A_batch = transform_matrix[:, :, :2]
    if A_batch.size(0) != batch_size:
        A_batch = A_batch.repeat(batch_size, 1, 1)

    B_batch = transform_matrix[:, :, 2].unsqueeze(1)

    coord = coord.bmm(A_batch.transpose(1, 2)) + B_batch.expand(coord.shape)

    return coord


def transform_image(image: torch.Tensor, transform_matrix: torch.Tensor, out_image_size=(512,512)):

    device = image.device

    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]

    out_image_h = out_image_size[0]
    out_image_w = out_image_size[1]

    identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=device)
    intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

    grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w], align_corners=False)
    grid = grid.reshape(intermediate_grid_shape)

    # For some reason it gives you w, h at the output of affine_grid. So switch here.
    grid = grid[..., [1, 0]]
    grid = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=grid)
    grid = grid[..., [1, 0]]

    grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

    # There is no constant selection for padding mode - so border will have to do to weights.
    image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0)

    return image


from typing import Tuple, Optional, List, Union

import torch
import torch.nn.functional as F


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True,
        device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    ys: Optional[torch.Tensor] = None
    xs: Optional[torch.Tensor] = None

    if normalized_coordinates:
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
    else:
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([ys, xs]))  # 2xHxW

    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def render_gaussian_dot(
        mean: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int]) -> torch.Tensor:
    r"""Renders the PDF of a 2D Gaussian distribution.

    mean is y, x
    std is y,x
    size is height,width

    Shape:
        - `mean`: :math:`(*, 2)`
        - `std`: :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        - Output: :math:`(*, H, W)`
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    mean = mean.unsqueeze(-2).unsqueeze(-2)
    std = std.unsqueeze(-2).unsqueeze(-2)

    grid: torch.Tensor = create_meshgrid(height=height,
                                         width=width,
                                         normalized_coordinates=False,
                                         device=mean.device)
    grid = grid.to(mean.dtype)

    delta = (grid - mean)
    k = -0.5 * (delta / std) ** 2
    gauss = torch.exp(torch.sum(k, dim=-1))

    return gauss

@torch.jit.script
def render_gaussian_curve(
        mean: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        chunk_size: int = 128) -> torch.Tensor:
    r"""Renders the PDF of a 2D Gaussian distribution.

    mean is y, x
    std is y, x
    size is height, width

    Shape:
        - `mean`: :math:`(*, 2)`
        - `std`: :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        - Output: :math:`(*, H, W)`
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    num_points = mean.shape[0]

    if std.dim() == 1:
        std = std.repeat((num_points, 1))

    mean = mean.unsqueeze(-2).unsqueeze(-2)
    std = std.unsqueeze(-2).unsqueeze(-2)

    grid: torch.Tensor = create_meshgrid(height=height,
                                         width=width,
                                         normalized_coordinates=False,
                                         device=mean.device)
    grid = grid.to(mean.dtype)
    out = torch.ones(size=(chunk_size+1, height, width), dtype=torch.float32, device=mean.device) * float("Inf")

    i = 0
    while i < num_points:
        j = min(i + chunk_size, num_points)
        mini_chunk_size = j - i

        delta = (grid - mean[i:j, ...])
        out[:mini_chunk_size, ...] = ((delta / std[i:j, ...]) ** 2).sum(dim=-1)
        out[chunk_size, ...] = out.min(dim=-3)[0]

        i = j

    out = torch.exp(-0.5 * out[chunk_size, ...])
    return out


#@torch.jit.script
def render_gaussian_dot_u(
        point: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 255.0
) -> torch.Tensor:

    gauss = render_gaussian_dot(mean=point, std=std, size=size)
    return gauss.mul(mul).to(torch.uint8)

#@torch.jit.script
def render_gaussian_dot_f(
        point: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 1.0
) -> torch.Tensor:

    gauss = render_gaussian_dot(mean=point, std=std, size=size)
    return gauss.mul(mul)

#@torch.jit.script
def render_gaussian_curve_u(
        points: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 255.0
) -> torch.Tensor:

    gauss = render_gaussian_curve(mean=points, std=std, size=size)
    return gauss.mul(mul).to(torch.uint8)

#@torch.jit.script
def render_gaussian_curve_f(
        points: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 1.0
) -> torch.Tensor:
    gauss = render_gaussian_curve(mean=points, std=std, size=size)
    return gauss.mul(mul)





def gaussian(window_size: int, sigma: torch.tensor):
    sigma = sigma.unsqueeze(-1)

    x = torch.arange(window_size, device=sigma.device).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum(dim=-1, keepdim=True)


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: torch.Tensor,
        force_even: bool = False) -> torch.Tensor:

    ksize_y, ksize_x = kernel_size
    sigma_y = sigma[..., 0]
    sigma_x = sigma[..., 1]

    kernel_x: torch.Tensor = gaussian(ksize_x, sigma_x)
    kernel_y: torch.Tensor = gaussian(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-2)
    )
    return kernel_2d

def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]

def gaussian_blur2d(x: torch.tensor,
                    kernel_size: Tuple[int, int],
                    sigma: torch.Tensor,
                    ):
    if x.ndim != 4:
        raise Exception

    b, c, h, w = x.shape

    filter = get_gaussian_kernel2d(kernel_size, sigma)
    filter = filter.unsqueeze(1)

    filter_height, filter_width = filter.shape[-2:]
    padding_shape: List[int] = compute_padding((filter_height, filter_width))
    input_pad: torch.Tensor = F.pad(x, padding_shape, mode='constant')

    out = F.conv2d(input_pad, filter, groups=c, padding=0, stride=1)

    return out

def gaussian_blur2d_norm(y_pred: torch.Tensor,
                         kernel_size: Tuple[int, int],
                         sigma: torch.Tensor
                         ):

    max_y_pred = torch.max(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), dim=2, keepdim=True)[0].unsqueeze(3)

    y_pred = gaussian_blur2d(x=y_pred, kernel_size=kernel_size, sigma=sigma)

    max_y__pred = torch.max(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), dim=2, keepdim=True)[0].unsqueeze(3)
    min_y__pred = torch.min(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), dim=2, keepdim=True)[0].unsqueeze(3)

    y_pred = ((y_pred - min_y__pred) / (max_y__pred - min_y__pred)) * max_y_pred

    return y_pred



def dedupe_points(points: np.array):
    last = None
    temp = []
    for point in points:
        if not (point == last).all():
            temp.append(point)
        last = point

    points = np.array(temp)

    return points


def line_len(points: np.array):
    line_len = np.sum(np.sqrt(np.sum((points[1:, 0:2] - points[:-1, 0:2]) ** 2, axis=-1)), axis=-1)
    return line_len


def line_len_t(points: torch.tensor):
    line_len = torch.sum(torch.sqrt(torch.sum((points[1:, 0:2] - points[:-1, 0:2]) ** 2, dim=-1)), dim=-1)
    return line_len


def interpolate_line(points: np.array, num_points: int, even_spacing=True):

    ##REMEMBER AXIS = 0 (by default is -1)

    if even_spacing:
        n = points.shape[0]
        distances = np.arange(n)
        curve_distance_max = n - 1
    else:
        # Uneven spacing not implemented
        raise Exception

    line = scipy.interpolate.interp1d(x=distances, y=points, axis=0, kind='linear')

    new_distances = np.linspace(0, curve_distance_max, num_points)

    return line(new_distances)


def interpolate_curve(points: np.array, num_points: int, even_spacing=True):

    num_points = int(num_points)

    if even_spacing:
        n = points.shape[0]
        distances = np.arange(n)
        curve_distance_max = n - 1
    else:
        raise Exception
        #distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        #num_curve_points = distance[-1]
        #distance = np.insert(distance, 0, 0)
        #distance = distance / distance[-1]


    cs = scipy.interpolate.CubicSpline(distances, points, bc_type='natural')

    new_distances = np.linspace(0, curve_distance_max, num_points)

    return cs(new_distances)


def interpolate_curveline(ys: Sequence[float], xs: Sequence[float], straight_segments: Optional[Sequence[int]] = None, total_points_out: int = 200):
    #straight_segments - straight is true, curve is false

    num_points_in = len(ys)

    if total_points_out < num_points_in:
        raise Exception(f"xs: {xs}, ys: {ys}, ss: {straight_segments}, npi: {num_points_in}")

    if straight_segments is None:
        straight_segments = [0] * num_points_in

    num_segments_in = num_points_in - 1
    points_per_segment_out = total_points_out // num_segments_in
    extra_points_last_segment = total_points_out - (points_per_segment_out * num_segments_in)

    out_curve_y = []
    out_curve_x = []

    temp_curve_y = []
    temp_curve_x = []

    last_straight_marker = False

    # This works like a state machine.
    # Iterate through the points
    # Add them to a temporary list
    # Every time you come across a straight segment
    # Interpolate the points that you have so far.
    # Add that point again to the list and keep on.
    # Trigger on next loop or if next point is last.

    for i in range(num_points_in):
        temp_curve_y.append(ys[i])
        temp_curve_x.append(xs[i])

        straight_segment = straight_segments[i]
        is_last_segment = (i == (num_points_in - 1))

        if straight_segment or last_straight_marker or is_last_segment:

            partial_curve_points = len(temp_curve_x)
            partial_curve_segments = partial_curve_points - 1

            if partial_curve_points == 0:
                raise Exception  # this shouldn't happen
            elif partial_curve_points == 1:
                pass  # The first point is a straight segment
            elif partial_curve_points == 2:
                if is_last_segment:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(partial_curve_segments*points_per_segment_out)+extra_points_last_segment)
                    out_curve_y.extend(temp[:, 0].tolist())
                    out_curve_x.extend(temp[:, 1].tolist())
                else:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(partial_curve_segments*points_per_segment_out)+1)
                    out_curve_y.extend(temp[:-1, 0].tolist())
                    out_curve_x.extend(temp[:-1, 1].tolist())
            elif partial_curve_points > 2:
                if is_last_segment:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(points_per_segment_out*partial_curve_segments)+extra_points_last_segment)
                    out_curve_y.extend(temp[:, 0].tolist())
                    out_curve_x.extend(temp[:, 1].tolist())
                else:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(points_per_segment_out*partial_curve_segments)+1)
                    out_curve_y.extend(temp[:-1, 0].tolist())
                    out_curve_x.extend(temp[:-1, 1].tolist())

            temp_curve_x = []
            temp_curve_y = []

            temp_curve_y.append(ys[i])
            temp_curve_x.append(xs[i])

            if straight_segment:
                last_straight_marker = True
            else:
                last_straight_marker = False

        else:
            continue

    return out_curve_y, out_curve_x

def make_curve_labels(points: np.ndarray, image_size, kernel_sd: int, kernel_size: int):

    image_height, image_width = image_size
    kernel_size_half = kernel_size // 2

    blur_kernel = get_blur_kernel(kernel_sd=kernel_sd, kernel_size=kernel_size)

    out = np.zeros((image_height + kernel_size - 1, image_width + kernel_size - 1), dtype=np.float32)

    points = points.astype(np.int)
    curve_ys = points[:, 0]
    curve_xs = points[:, 1]

    for curve_y, curve_x in zip(curve_ys, curve_xs):
        if curve_y + kernel_size_half + 1 > image_height:
            continue
        elif curve_x + kernel_size_half + 1 > image_height:
            continue
        elif curve_y - kernel_size_half < 0:
            continue
        elif curve_x - kernel_size_half < 0:
            continue
        else:
            out[curve_y:curve_y + kernel_size, curve_x:curve_x + kernel_size] = np.maximum(out[curve_y:curve_y + kernel_size, curve_x:curve_x + kernel_size], blur_kernel)

    out = out[kernel_size_half:image_height + kernel_size_half,
          kernel_size_half:image_width + kernel_size_half]

    return out


def get_blur_kernel(kernel_sd, kernel_size):

    if kernel_size % 2 != 1:
        raise Exception

    blur_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    blur_kernel[kernel_size // 2, kernel_size // 2] = 1
    blur_kernel = scipy.ndimage.filters.gaussian_filter(blur_kernel, (kernel_sd, kernel_sd))
    blur_kernel = blur_kernel / np.max(blur_kernel)

    return blur_kernel