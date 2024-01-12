import datetime
from pathlib import Path
from typing import Optional
import requests_cache
import torch

from torch.utils.data import Dataset
from src.pressure_damping.label_processer import FetchResult, FirebaseCurveFetcher, Curve

from loguru import logger
from src.utils.cluster_source import ClusterSource

import torchvision

CLUSTER_STORE_URL = "https://storage.googleapis.com/scantensus/fiducial"

class CurveDataset(Dataset):
    """
    A `Dataset` that allows for loading image and curve data from multiple sources.
    """

    project_codes: list[str]
    """
    The list of project codes associated with this dataset.
    """

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

    def __init__(self, projects: list[str], firebase_certificate: Path = Path('.firebase.json')) -> None:
        super().__init__()

        self.project_codes = projects
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



    def _fetch_image(self, code: str) -> Optional[torch.Tensor]:
        """
        Fetches the image for the given code.
        """

        source: ClusterSource = None

        if 'clusters' in code:
            source = ClusterSource(unity_code=code,
                                    png_cache_dir=None,
                                    server_url=CLUSTER_STORE_URL)
        else: 
            logger.error(f'Code of unknown source: {code}')

            return None 


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


    def __len__(self) -> int:
        return len(self.unrolled_curves)


    def __getitem__(self, index: int) -> any:
        matt_data = self.unrolled_matt_data[index]
        curve = self.unrolled_curves[index]

        image = self._fetch_image(curve.file)

        return image


    