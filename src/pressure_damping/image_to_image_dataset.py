from dataclasses import dataclass, asdict
import datetime
from pathlib import Path
from typing import Optional
import requests_cache
import torch
import torchvision
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from src.pressure_damping.label_processer import FetchResult, FirebaseCurveFetcher

from loguru import logger
from src.pressure_damping.unity_dataset import center_crop_or_pad_t, transform_image
from src.utils.cluster_source import ClusterSource
from src.utils.securion_source import SecurionSource

from kornia.geometry.transform import get_affine_matrix2d, warp_perspective

CLUSTER_STORE_URL = "https://storage.googleapis.com/scantensus/fiducial"
SECURION_STORE_URL = "http://cardiac5.ts.zomirion.com:50601/scantensus-database-png-flat"

class UnityImageToImageDataset(Dataset):
    """
    A PyTorch dataset to work with Image-to-Image tasks using our data sources and pre-processors.
    """

    # MARK: Init Properties

    project_codes: list[str]
    """
    The internal code for the Unity projects included in this dataset.
    """

    output_shape: tuple[int, int]
    """
    The output shape of the dataset images and labels in the form (height, width).
    """

    crop_shape: tuple[int, int]
    """
    The shape to crop the images to in the form (height, width).
    """

    firebase_auth: Path

    debug_mode: bool = False
    """
    Wether to enable debug mode.
    """
    

    # MARK: Generated Properties

    curves: list['Curve'] = []
    """
    A single collection of all the curves.
    """

    def __init__(self, project_codes: list[str], output_shape: tuple[int, int], crop_shape: tuple[int, int], firebase_certificate: Path, debug_mode: bool = False) -> None:
        self.project_codes = project_codes
        self.output_shape = output_shape
        self.crop_shape = crop_shape
        self.debug_mode = debug_mode
        self.firebase_auth = firebase_certificate

        self.curves = self._fetch_curves()


    def _fetch_curves(self) -> list['Curve']:
        """
        Fetches the curves for this dataset.
        """

        fetcher = FirebaseCurveFetcher(self.firebase_auth)

        curves = []

        for project_code in self.project_codes:
            result = fetcher.fetch(project_code)

            curves.extend(result.curves)

        return curves

            

    def _process_labels(self, curve: 'Curve') -> dict[str, any]:
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

        return self._read_image_into_t(image_path, png_session)


    def _read_image_into_t(self,
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


    def _pad_or_crop(self, image: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """
        Pads or crops the given image to match `self.crop_shape`.

        Internally calls Matt's `center_crop_or_pad_t` function.
        """

        return center_crop_or_pad_t(image, self.crop_shape, device='cpu')


    def _random_transform_with_fixed_scaling(
        self,
        image: torch.Tensor,
        translation_range: tuple[float] = (-0.1, 0.1),
        rotation_range: tuple[float] = (-5, 5),
        scale: tuple[float] = (1.0, 1.0)
        ) -> torch.Tensor:
        """
        Generates a random affine transformation matrix for a given image with a fixed scaling factor.

        :param image: The image to generate the transformation matrix for.
        :param translation_range: The range of the translation in pixels.
        :param rotation_range: The range of the rotation in degrees.
        :param scale: The tuple containing the (y, x) scale of the image.
        :return: The transformation matrix.
        """
        if image.dim() == 3:
            (B, H, W) = image.shape
        elif image.dim() == 4:
            (B, _, H, W) = image.shape
        else:
            raise ValueError(f'Image has invalid shape: {image.shape}')


        translations = torch.distributions.Uniform(*translation_range).sample((B, 2))
        center = torch.tensor([H / 2, W / 2]).repeat(B, 1)
        angle = torch.distributions.Uniform(*rotation_range).sample((B,))
        scale = torch.tensor(scale).repeat(B, 1)

        return get_affine_matrix2d(
            translations=translations,
            center=center,
            scale=scale,
            angle=angle
        )


    # MARK: `torch.Dataset` implementation

    def __len__(self) -> int:
        return len(self.curves)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float], dict[str, any]]:
        curve = self.curves[index]
        image = self._fetch_image(curve.file)

        if self.debug_mode:
            torchvision.utils.save_image(image.to(torch.float32).div(255.0), 'out/raw.png')

        # make image grayscale if necessary
        if image.shape[0] == 3:
            image = F.rgb_to_grayscale(image)

        image, height_shift, width_shift = center_crop_or_pad_t(image, self.crop_shape, device='cpu')

        if self.debug_mode:
            torchvision.utils.save_image(image.to(torch.float32).div(255.0), 'out/cropped.png')

        scale_H = self.output_shape[0] / image.shape[1]
        scale_W = self.output_shape[1] / image.shape[2]

        T = self._random_transform_with_fixed_scaling(image, scale = (scale_H, scale_W))

        image = image.to(torch.float32).div(255.0).unsqueeze(1)
        image = warp_perspective(image, T, dsize=self.output_shape, mode='bilinear', align_corners=True)

        if self.debug_mode:
            torchvision.utils.save_image(image, 'out/transformed.png')

        image = image.mul(255.0).to(torch.uint8)

        return image.squeeze(1), T, (height_shift, width_shift), self._process_labels(curve)

        





@dataclass 
class Curve:
    project: str
    file: str
    user: str
    time: str
    label: str
    xs: list[float]
    ys: list[float]
    straight_flag: list[int]
    type: str