import numpy as np
import pickle
import argparse
import imageio.v3 as imageio

from pathlib import Path

from sklearn.decomposition import PCA, FactorAnalysis, FastICA, NMF


# Hacer PCA con las imágenes normalizadas

def load_image(path, normalize=False):
    input = imageio.imread(path)

    # ensure input has channels last
    if path.suffix == '.tiff':
        input = input.transpose(1, 2, 0)

    # convert image to float
    if normalize:
        input = input.astype(np.float32) / np.iinfo(input.dtype).max
    else:
        input = input.astype(np.float32)

    return input


def main(args):
    data_root = Path(args.data_path)

    # Load training images
    train_paths = list(Path(data_root, 'train').iterdir())[:100]
    inputs = []
    for path in train_paths:
        inputs.append(load_image(path, args.normalize))
    inputs = np.array(inputs)

    # Create reduction model
    if args.reduction_method == 'pca':
        reduction_model = PCA(n_components=args.num_components)
    elif args.reduction_method == 'fa':
        reduction_model = FactorAnalysis(n_components=args.num_components)
    elif args.reduction_method == 'ica':
        reduction_model = FastICA(n_components=args.num_components)
    elif args.reduction_method == 'nmf':
        reduction_model = NMF(n_components=args.num_components)

    # Fit model
    reduction_model.fit(inputs.reshape(-1, inputs.shape[-1]))

    output_root = Path(f'{data_root}_{args.reduction_method}{args.num_components}')

    # Apply model to all files
    for path in data_root.glob('*/*'):
        split = path.parts[-2]
        image_id = path.stem
        new_path = Path(output_root, split, f'{image_id}.npy')

        input = load_image(path, args.normalize)
        reduced = reduction_model.transform(input.reshape(-1, input.shape[-1]))
        reduced = reduced.reshape(*input.shape[:-1], -1)

        new_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(new_path, reduced)

    # Store trained model
    pickle.dump(reduction_model, open(Path(output_root, 'reduction_model.pkl'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default='data/spectralwaste_segmentation/hyper')
    parser.add_argument('--reduction-method', type=str, choices=['pca', 'fa', 'ica', 'nmf'], default='pca')
    parser.add_argument('--num-components', type=int, default=3)
    parser.add_argument('--normalize', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
