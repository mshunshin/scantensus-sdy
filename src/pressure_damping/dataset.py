import torch

from torch import Tensor
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class FirebaseKeypointDataset(Dataset):
    """
    A torch dataset for the Firebase images and curves as labels.
    """

    firebase_label_store: str = "https://console.firebase.google.com/u/0/project/scantensus/database/scantensus/data/fiducial/"

    def __init__(self, credentials_path: Path) -> None:
        super().__init__()

        self.credentials_path = credentials_path

