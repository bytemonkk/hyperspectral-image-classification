from setuptools import setup, find_packages

setup(
    name="MF-HSINet",
    version="1.0.0",
    author="Manoj Kumar Sunkara",
    description="Dual-Branch Hyperspectral Image Classification using Spectral-Spatial Learning and Attention Fusion",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "opencv-python",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "open_clip_torch",
        "einops",
        "tqdm",
        "Pillow",
        "spectral",
        "h5py"
    ],
)