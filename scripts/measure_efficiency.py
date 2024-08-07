import pickle
import matplotlib.pyplot as plt
import pandas

import torch
from torch.utils import benchmark

from spectralwaste_segmentation import models


batch_size = 1
device = 'cuda'

pca_model = pickle.load(open('data/spectralwaste_segmentation/hyper_pca3/reduction_model.pkl', 'rb'))
components = torch.tensor(pca_model.components_)[..., None, None].to(dtype=torch.float32).to(device)


# Functions

def unimodal(model, input):
    with torch.inference_mode():
        model(input)

def unimodal_reduction(model, input):
    with torch.inference_mode():
        reduced_input = torch.nn.functional.conv2d(input, components)
        model(reduced_input)

def multimodal(model, input):
    with torch.inference_mode():
        model(input)

def multimodal_reduction(model, input):
    with torch.inference_mode():
        reduced_input = torch.nn.functional.conv2d(input[1], components)
        model([input[0], reduced_input])


configs = {
    'mininet_rgb': (3, models.create_model('mininet', 3, 7), unimodal),
    'mininet_hyper': (224, models.create_model('mininet', 224, 7), unimodal),
    'mininet_hyper3': (224, models.create_model('mininet', 3, 7), unimodal_reduction),
    'mininet_rgb_hyper': ([3, 224], models.create_model('mininet_multimodal', [3, 224], 7), multimodal),
    'mininet_rgb_hyper3': ([3, 224], models.create_model('mininet_multimodal', [3, 3], 7), multimodal_reduction),

    'segformer_rgb': (3, models.create_model('segformer_b0', 3, 7), unimodal),
    'segformer_hyper': (224, models.create_model('segformer_b0', 224, 7), unimodal),
    'segformer_hyper3': (224, models.create_model('segformer_b0', 3, 7), unimodal_reduction),
    'segformer_rgb_hyper': ([3, 224], models.create_model('segformer_b0_multimodal', [3, 224], 7), multimodal),
    'segformer_rgb_hyper3': ([3, 224], models.create_model('segformer_b0_multimodal', [3, 3], 7), multimodal_reduction),

    'cmx_rgb_hyper': ([3, 224], models.create_model('cmx_b0', [3, 224], 7), multimodal),
    'cmx_rgb_hyper3': ([3, 224], models.create_model('cmx_b0', [3, 3], 7), multimodal_reduction),
}

results = dict()

print('components', components.numel())

for name, (input_channels, model, inference_fn) in configs.items():
    print(name)

    result = dict()

    model.to(device)
    model.eval()
    components = components.to(device)

    if isinstance(input_channels, list):
        input = [torch.randn(batch_size, num, 256, 256, dtype=torch.float32, device=device) for num in input_channels]
    else:
        input = torch.randn(batch_size, input_channels, 256, 256, dtype=torch.float32, device=device)

    result['parameters'] = sum([p.numel() for p in model.parameters()])

    with torch.inference_mode():

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_flops=True) as p:
            inference_fn(model, input)

        timer = benchmark.Timer(
            stmt='inference_fn(model, input)',
            num_threads=32,
            globals={'model': model, 'input': input, 'inference_fn': inference_fn}
        )

    result['flops'] = sum([event.flops for event in p.events()])
    result['time'] = timer.timeit(100).median
    results[name] = result

print(pandas.DataFrame(results).T)