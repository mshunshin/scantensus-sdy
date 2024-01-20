import PIL
import PIL.Image
from matplotlib import pyplot as plt

import torch
import torchvision

import numpy as np

def visualize_heatmap(image: torch.Tensor, heatmap: torch.Tensor) -> PIL.Image.Image:
    """
    Visualizes the given heatmap on top of the given image.

    :param image: The image to visualize the heatmap on.
    :param heatmap: The heatmap to visualize.
    :return: The image with the heatmap on top.
    """
    # convert to PIL image
    image = torchvision.transforms.functional.to_pil_image(image)
    heatmap = torchvision.transforms.functional.to_pil_image(heatmap)

    # resize heatmap to match image size
    heatmap = heatmap.resize(image.size)

    # convert to numpy arrays
    heatmap = np.array(heatmap)

    # normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # convert heatmap to jet colormap
    heatmap = np.uint8(255 * plt.cm.jet(heatmap))

    # convert to PIL image
    heatmap = PIL.Image.fromarray(heatmap)

    # create overlay
    overlay = PIL.Image.blend(image.convert('RGBA'), heatmap.convert('RGBA'), alpha=0.5)

    return overlay