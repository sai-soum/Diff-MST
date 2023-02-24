from pathlib import Path
from setuptools import setup, find_packages

NAME = "mix-style-transfer"
DESCRIPTION = "Mix style transfer with differentiable signal processing"
URL = "https://github.com/sai-soum/mix_style_transfer.git"
EMAIL = "s.s.vanka@qmul.ac.uk"
AUTHOR = "Soumya Sai Vanka"
REQUIRES_PYTHON = ">=3.7.11"
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=[
        "mst",
    ],
    install_requires=[
        "torch",
        "torchaudio",
        "pedalboard",
        "torchvision",
        "auraloss",
        "pytorch_lightning",
        "scipy",
        "matplotlib",
        "numpy",
        "tqdm",
        "tensorboard",
        "librosa",
    ],
    extras_require={
        "asteroid": ["asteroid-filterbanks>=0.3.2"],
        "tests": [
            "pytest",
            "musdb>=0.4.0",
            "museval>=0.4.0",
            "asteroid-filterbanks>=0.3.2",
            "onnx",
            "tqdm",
        ],
        "stempeg": ["stempeg"],
        "evaluation": ["musdb>=0.4.0", "museval>=0.4.0"],
    },
    # entry_points={"console_scripts": ["umx=openunmix.cli:separate"]},
    # packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
