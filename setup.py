import setuptools

setuptools.setup(
    name='spectralwaste_segmentation',
    version=0.1,
    author="",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'numpy<2.0',
        'matplotlib',
        'imageio',
        'easydict',
        'scikit-image',
        'scikit-learn==1.3.2',
        'pandas',
        'opencv-python',
        'torch==2.2.2',
        'torchvision==0.17.2',
        'torchmetrics',
        'wandb'
    ],
)