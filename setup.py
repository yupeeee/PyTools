from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="yupeeee-pytools",
    version="0.1.3",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juyeop Kim",
    author_email="juyeopkim@yonsei.ac.kr",
    url="https://github.com/yupeeee/PyTools",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "grad-cam",
        "matplotlib",
        "ml_collections",
        "more_itertools",
        "numpy",
        "opencv-python == 4.5.5.64",
        "pandas",
        "Pillow",
        "pyyaml",
        "scikit_learn",
        "scipy",
        "timm",
        "tqdm",
    ],
)
