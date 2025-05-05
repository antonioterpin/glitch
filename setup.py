"""Setup script for the SynthPix package."""

from setuptools import find_packages, setup

from src.glitch import __version__ as version

# flake8: noqa: E501
setup(
    name="glitch",
    version=version,
    author="Antonio Terpin",
    description="Transitions generation based on preferences.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax>=0.4.27",
        "tqdm>=4.67.1",
        "ruamel.yaml>=0.18.10",
        "imageio>=2.37.0",
        "matplotlib>=3.10.1",
        "optax>=0.2.4",
        "flax>=0.10.2",
        "torch>=2.7.0",
        "colorama>=0.4.6",
        "wandb",
        "hcnn @ git+ssh://git@github.com/antonioterpin/hcnn.git@feat-autotuning",
    ],
    extras_require={
        "dev": [
            "snowballstemmer==2.2.0",
            "pre_commit==4.0.1",
            "pytest==7.4.4",
        ],
        "cuda12": ["jax[cuda12_pip]"],
        "docs": [
            "Sphinx==7.4.7",
            "sphinx-copybutton==0.5.2",
            "sphinx-rtd-theme==2.0.0",
            "sphinx-tabs==3.4.7",
            "sphinx-togglebutton==0.3.2",
            "sphinxcontrib-applehelp==2.0.0",
            "sphinxcontrib-bibtex==2.6.2",
            "sphinxcontrib-devhelp==2.0.0",
            "sphinxcontrib-htmlhelp==2.1.0",
            "sphinxcontrib-jquery==4.1",
            "sphinxcontrib-jsmath==1.0.1",
            "sphinxcontrib-qthelp==2.0.0",
            "sphinxcontrib-serializinghtml==2.0.0",
        ],
    },
    python_requires=">=3.11",
)
