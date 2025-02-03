"""PyRAI package setup."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyrai",
    version="0.1.0",
    author="PyRAI Team",
    author_email="team@pyrai.xyz",
    description="Python Reinforced AI Infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyraixyz/pyrai",
    project_urls={
        "Bug Tracker": "https://github.com/pyraixyz/pyrai/issues",
        "Documentation": "https://pyrai.readthedocs.io/",
        "Source Code": "https://github.com/pyraixyz/pyrai",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "tensorflow>=2.6.0",
        "solana>=0.20.0",
        "fastapi>=0.68.0",
        "pydantic>=1.8.0",
        "prometheus-client>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyrai=pyrai.cli:main",
        ],
    },
) 