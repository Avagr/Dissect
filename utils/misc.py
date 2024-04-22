import os
import random
from datetime import datetime

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom


def colorize_text(tokens, scores):
    cmap = matplotlib.colormaps['coolwarm']
    token_colors = (matplotlib.colors.Normalize(vmin=-1, vmax=1)(scores))
    template = '<span class="attention-word"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''.join(
        [template.format(matplotlib.colors.rgb2hex(cmap(color)[:3]), word) for word, color in
         zip(tokens, token_colors)])
    return colored_string


def clean_tokens(words):
    replacements = {'▁': ' ', '&': '&amp;', '<': '&lt;', '>': '&gt;'}
    return [''.join(replacements.get(ch, ch) for ch in word) for word in words]


def plot_image_with_heatmap(image: Image, heatmap: np.array, ax: plt.Axes = None):
    image_np = np.array(image)
    zoom_y = image_np.shape[0] / heatmap.shape[0]
    zoom_x = image_np.shape[1] / heatmap.shape[1]
    heatmap_resized = zoom(heatmap, (zoom_y, zoom_x))
    if ax is None:
        plt.imshow(image_np, cmap='gray')
        plt.imshow(heatmap_resized, cmap='coolwarm', alpha=0.5)
        plt.show()
    else:
        ax.imshow(image_np, cmap='gray')
        ax.imshow(heatmap_resized, cmap='coolwarm', alpha=0.5)


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp():
    return str(datetime.now())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
