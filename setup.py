from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyrai",
    version="0.1.0",
    author="PyRAI Contributors",
    author_email="contact@pyrai.org",
    description="A flexible deep learning framework with multiple backend support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/pyrai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "tensorflow>=2.4.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
        "scikit-learn>=0.24.0",
        "pillow>=8.0.0",
        "psutil>=5.8.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "gpu": [
            "pynvml>=11.0",
        ],
        "examples": [
            "torchvision>=0.9.0",
            "torchtext>=0.9.0",
            "tensorflow-datasets>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyrai=pyrai.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/pyrai/issues",
        "Documentation": "https://pyrai.readthedocs.io/",
        "Source Code": "https://github.com/username/pyrai",
    },
) 