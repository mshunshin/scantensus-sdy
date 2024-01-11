from src.sdy_file import SDYFile

import torch

class SDYInference():
    def run(file: SDYFile):
        pixel_array = file.spectrum
        pixel_array = torch.from_numpy(pixel_array.astype("float32"))

        pixel_array = torch.sqrt(pixel_array * 2) * 1.5
        pixel_array = torch.clip(pixel_array, 0, 255)
        pixel_array = torch.flip(pixel_array, [0])
        pixel_array = pixel_array.squeeze(0).squeeze(0).repeat([1,3,1,1])
