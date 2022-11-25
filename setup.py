import os

from setuptools import find_packages, setup

with open(os.path.join("stable_baselines3", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# SSFC


"""  # noqa:E501


setup(
    name="SSFC",
    packages=[package for package in find_packages() if package.startswith("SSFC")],
    package_data={"SSFC": ["py.typed", "version.txt"]},
    install_requires=[
        "gym==0.21.0",
        "numpy==1.21.6",
        "torch==1.11.0",
        # For saving models
        "cloudpickle==1.5.0",
        # For reading logs
        "pandas==1.3.4",
        # Plotting learning curves
        "matplotlib",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
            # For toy text Gym envs
            "scipy>=1.4.1",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
        "extra": [
            # For render
            "opencv-python",
            # For atari games,
            "atari_py~=0.2.0",
            "pillow",
            # Tensorboard support
            "tensorboard>=2.2.0",
            # Checking memory taken by replay buffer
            "psutil",
        ],
    },
    long_description_content_type="text/markdown",
    version=__version__,
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
