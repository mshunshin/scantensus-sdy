from dataclasses import dataclass, asdict
import datetime
from pathlib import Path
from typing import Optional
import requests_cache
import torch

from torch.utils.data import Dataset
from src.pressure_damping.label_processer import FetchResult, FirebaseCurveFetcher, Curve

from loguru import logger
from src.pressure_damping.unity_dataset import center_crop_or_pad_t, transform_image
from src.utils.cluster_source import ClusterSource
from src.utils.securion_source import SecurionSource


# avoid scaling stuff
# 1. pad the image and keep the shifts (x and y)
# 2. work out a random transform (rot, trans, scale, shear) for the image
# 3. image_t = apply_transform(image, transform)
# 4. make_heatmap([T], shifts)

# (optional 1. move stuff to kornia)


import torchvision

CLUSTER_STORE_URL = "https://storage.googleapis.com/scantensus/fiducial"
SECURION_STORE_URL = "http://cardiac5.ts.zomirion.com:50601/scantensus-database-png-flat"

@dataclass
class CurveDatasetResult:
    """
    The result of fetching curves.
    """

    image: torch.Tensor

    transform_matrix: torch.Tensor

    labels: dict[str, any]

    height_shift: float

    width_shift: float

class CurveDataset(Dataset):
    """
    A `Dataset` that allows for loading image and curve data from multiple sources.
    """

     # MARK: Inputs

    project_codes: list[str]
    """
    The list of project codes associated with this dataset.
    """

    output_shape: tuple[int, int]
    """
    The output shape of the dataset images and labels in the form (height, width).
    """

    crop_shape: tuple[int, int]
    """
    The shape to crop the images to.
    """

    # MARK: Generated Properties

    fetcher: FirebaseCurveFetcher
    """
    The fetcher to use to fetch the curves.
    """

    fetch_results: list[FetchResult] = []
    """
    The fetch results.
    """

    unrolled_curves: list[Curve] = []
    """
    The unrolled curves.
    """

    unrolled_matt_data: list[dict[str, any]] = []
    """
    The unrolled Matt-formated data.
    """

    raw_labels: dict[str, dict[str, any]] = {}
    """
    The raw `labels/` database for each of the projects.
    """

    def __init__(self, 
                projects: list[str], 
                output_shape: tuple[int, int] = (512, 512),
                crop_shape: tuple[int, int] = (608, 608),
                firebase_certificate: Path = Path('.firebase.json')) -> None:
        super().__init__()

        self.project_codes = projects
        self.output_shape = output_shape
        self.crop_shape = crop_shape
        self.fetcher = FirebaseCurveFetcher(firebase_certificate)

        self._fetch_curves()

        logger.info(f'Loaded {len(self.fetch_results)} results containing {sum([len(res.curves) for res in self.fetch_results])} curves.')

    
    def _fetch_curves(self):
        """
        Fetches the curves for this dataset.
        """

        for project_code in self.project_codes:
            result = self.fetcher.fetch(project_code)

            self.fetch_results.append(result)
            
            self.unrolled_curves.extend(result.curves)
            self.unrolled_matt_data.extend(result.matt_data)

            self.raw_labels[project_code] = self.fetcher.fetch_raw(project_code)



    def _fetch_image(self, code: str) -> Optional[torch.Tensor]:
        """
        Fetches the image for the given code.

        Arguments:
            code (str): The code to fetch the image for. Currently only supports cluster codes.
        Returns:
            Optional[torch.Tensor]: The image as a torch.Tensor, or None if the code is not supported.
        """

        source: ClusterSource | SecurionSource = None

        if 'clusters' in code:
            source = ClusterSource(unity_code=code,
                                    png_cache_dir=None,
                                    server_url=CLUSTER_STORE_URL)
        else: 

            source = SecurionSource(unity_code=code,
                                    png_cache_dir=None,
                                    server_url=SECURION_STORE_URL)


        png_session = requests_cache.CachedSession(cache_name='png_cache',
                                                   use_cache_dir=True,
                                                   cache_control=False,
                                                   expire_after=datetime.timedelta(days=300),
                                                   backend='sqlite',
                                                   stale_if_error=True,
                                                   wal=True,
                                                   timeout=30)


        image_path = source.get_frame_url()

        return self.read_image_into_t(image_path, png_session)


    def read_image_into_t(self,
                          image_path: str,
                          png_session: requests_cache.CachedSession = None,
                          device="cpu"):
        """
        Reads an image from the given image_path and returns it as a torch.Tensor.

        Args:
            image_path (str): The path to the image file.
            png_session (requests_cache.CachedSession, optional): The session to use for downloading PNG images. Defaults to None.
            device (str, optional): The device to load the image onto. Defaults to "cpu".

        Returns:
            torch.Tensor: The image as a torch.Tensor.

        Raises:
            Exception: If the image fails to load.
        """

        if image_path.startswith("http://") or image_path.startswith("https://"):
            r = png_session.get(image_path)
            if r.status_code == 200:
                img_bytes = torch.frombuffer(r.content, dtype=torch.uint8)
                image = torchvision.io.decode_png(img_bytes)

                logger.debug(f"{image_path}: Successfully loaded")
            else:
                raise Exception(f"Failed to load {image_path}")
        else:
            image = torchvision.io.read_image(image_path)

        # ensure dim = 3
        if image.ndim == 2:
            image = torch.unsqueeze(image, 0)

        # remove alpha layer
        if image.shape[0] == 4:
            image = image[:3, ...]

        return image


    def transform_image(self, image: torch.Tensor, transform_matrix: torch.Tensor, out_image_size=(512,512)):
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
        grid = self.apply_matrix_to_coords(transform_matrix=transform_matrix, coord=grid)
        grid = grid[..., [1, 0]]

        grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

        # There is no constant selection for padding mode - so border will have to do to weights.
        image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0)

        return image


    def apply_matrix_to_coords(self, transform_matrix: torch.Tensor, coord: torch.Tensor):
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


    def _process_labels(self, curve: Curve) -> dict[str, any]:
        """
        Processes the labels for the given curve to match the requirements of UnityMakeHeatmap.

        This does not yet handle multiple curves on the same image (such as when training for multiple tasks at once).

        Arguments:
            curve (Curve): The curve to process.
        
        Returns:
            dict[str, any]: The processed labels.
        """

        keypoint_name = curve.label

        result: dict[str, any] = {}

        result[keypoint_name] = [{
            'x': curve.xs, 
            'y': curve.ys, 
            'straight_segment': curve.straight_flag,
            'type': curve.type
            }]

        return result


    def _pad_or_crop(self, image: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """
        Pads or crops the given image to match `self.crop_shape`.

        Internally calls Matt's `center_crop_or_pad_t` function.
        """

        return center_crop_or_pad_t(image, self.crop_shape, device='cpu')


    def _generate_scale_matrix(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generates scale-only transform matrix for the given image to match `self.output_shape`.
        """

        H, W = image.shape[1], image.shape[2]
        h, w = self.output_shape

        return torch.tensor([
            [h / H, 0, 0],
            [0, w / W, 0],
            [0, 0, 1]
        ])


    def _generate_transform_matrix(self, image: torch.Tensor) -> torch.Tensor:
        # TODO(guilherme): https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.get_affine_matrix2d
        pass 


    def __len__(self) -> int:
        return len(self.unrolled_curves)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, any], torch.Tensor]:
        curve = self.unrolled_curves[index]
        image = self._fetch_image(curve.file)

        image, height_shift, width_shift = self._pad_or_crop(image)
        transform_matrix = self._generate_scale_matrix(image)
        
        image = image.to(torch.float32).div(255.0)

        image = transform_image(
            image=image,
            # why the inverse tho?
            transform_matrix=transform_matrix.inverse(),
            out_image_size=self.output_shape
        )

        image = image.mul(255.0).to(torch.uint8)

        return asdict(CurveDatasetResult(
            image=image,
            transform_matrix=transform_matrix,
            labels=self._process_labels(curve),
            height_shift=height_shift,
            width_shift=width_shift
        ))



    