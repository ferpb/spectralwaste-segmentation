import torch
import matplotlib.pyplot as plt
import numpy as np


def get_color_labels(labels, palette):
    palette_array = np.array(palette, dtype=np.uint8)
    color_labels = palette_array[labels]
    return color_labels

def plot_labels(image, labels, palette, ax=None):

    if not ax:
        ax = plt.gca()
    color_labels_alpha = np.dstack([get_color_labels(labels, palette), np.zeros_like(labels)])
    color_labels_alpha[labels > 0, 3] = 180
    ax.imshow(image)
    ax.imshow(color_labels_alpha, interpolation='none')
    ax.axis('off')

def normalize_percentile(image, q_min=5, q_max=95, per_channel=True, clip=True):
    if per_channel:
        min_v = np.percentile(image, q_min, axis=(0,1), keepdims=True)
        max_v = np.percentile(image, q_max, axis=(0,1), keepdims=True)
    else:
        min_v = np.percentile(image, q_min)
        max_v = np.percentile(image, q_max)

    image = (image - min_v) / (max_v - min_v)
    if clip: image = np.clip(image, 0, 1)
    return image

def false_color(hyper):
    if hyper.shape[-1] <= 3:
        return hyper
    bands = hyper[:, :, [134, 170, 200]]
    bands_norm = normalize_percentile(bands, per_channel=True)
    return (bands_norm * 255).astype(np.uint8)