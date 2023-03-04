from setuptools import setup, find_packages


setup(
    name="yupeeee-pytools",
    version="0.1.2",
    description="",
    long_description="",
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
