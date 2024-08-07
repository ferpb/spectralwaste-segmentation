import numpy as np
from pathlib import Path

from typing import Union

from collections import namedtuple

import torch

import torchvision
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

import imageio.v3 as imageio


class SemanticSegmentationTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.RandomRotation(30),
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
            T.ToPureTensor()
        ])

    def forward(self, *inputs):
        return self.transform(*inputs)

class SemanticSegmentationTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.ToPureTensor()
        ])

    def forward(self, *inputs):
        return self.transform(*inputs)


class SpectralWasteSegmentation(torchvision.datasets.VisionDataset):
    SpectralWasteClass = namedtuple(
        'SpectralWasteClass',
        ['name', 'id', 'color', 'ignore_in_eval']
    )

    classes = [
        SpectralWasteClass('background', 0, (0, 0, 0), True),
        SpectralWasteClass('film', 0, (218, 247, 6), False),
        SpectralWasteClass('basket', 0, (51, 221, 255), False),
        SpectralWasteClass('cardboard', 0, (52, 50, 221), False),
        SpectralWasteClass('video_tape', 0, (202, 152, 195), False),
        SpectralWasteClass('filament', 0, (0, 128, 0), False),
        SpectralWasteClass('bag', 0, (255, 165, 0), False)
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        input_mode: Union[str, list[str]] = ['rgb', 'hyper'],
        target_mode: str = 'labels_rgb',
        target_type: str = 'semantic',
        transform=None,
        target_transform=None,
        transforms=None
    ):
        super().__init__(root, transforms, transform, target_transform)

        assert target_type in ['semantic', 'instance', '']

        self.input_mode = input_mode
        self.target_mode = target_mode
        self.target_type = target_type

        if not isinstance(input_mode, list):
            self.input_mode = [input_mode]

        self.classes_names = [c.name for c in self.classes]
        self.palette = [c.color for c in self.classes]
        self.num_classes = len(self.classes_names)

        self.input_dirs = [Path(root, mode, split) for mode in self.input_mode]
        self.target_dir = Path(root, self.target_mode, split)

        self.input_paths = [list(sorted(dir.iterdir())) for dir in self.input_dirs]
        self.target_paths = list(sorted(self.target_dir.glob(f'*{target_type}.png')))

        sample = self[0]
        if isinstance(sample[0], list):
            self.num_channels = [input.shape[0] for input in sample[0]]
        else:
            self.num_channels = sample[0].shape[0]

    def __getitem__(self, idx) -> tuple[Union[torch.FloatTensor, list[torch.FloatTensor]], torch.LongTensor]:
        # load inputs
        inputs = []
        for i, m in enumerate(self.input_mode):
            path = self.input_paths[i][idx]

            if path.suffix == '.npy':
                input = np.load(path).astype(np.float32)
            elif path.suffix == '.png':
                input = imageio.imread(path)
            elif path.suffix == '.tiff':
                input = imageio.imread(path)
                input = input.transpose(1, 2, 0)
            else:
                raise ValueError

            # convert image to float if it is integer
            if issubclass(input.dtype.type, np.integer):
                input = input.astype(np.float32) / np.iinfo(input.dtype).max

            # convert to tensor
            input = tv_tensors.Image(input)
            input = input.permute(2, 0, 1)
            inputs.append(input)

        # load target
        target = imageio.imread(self.target_paths[idx])
        target = torch.tensor(target.astype(np.int64))

        if self.target_type == 'instance':
            masks = []
            labels = []
            for id in target.unique():
                if id == 0:
                    continue
                masks.append(target == id)
                labels.append(id // 1024)
            target = dict(masks=tv_tensors.Mask(torch.stack(masks)), categories=torch.stack(labels))
        elif self.target_type == 'semantic' or self.target_type == '':
            target = tv_tensors.Mask(target)
        else:
            raise ValueError

        # apply transformations
        if self.transforms:
            target, *inputs = self.transforms(target, *inputs)

        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs, target

    def __len__(self):
        return len(self.input_paths[0])