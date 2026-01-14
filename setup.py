from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optimizer-emmoea",
    version="0.2.0",
    author="Sergei",
    description="Efficient Multi-Objective Surrogate-Assisted Evolutionary Algorithm with Performance Indicator-Based Infill Criterion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/optimizer-emmoea",
    packages=find_packages(exclude=["tests", "examples"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-optimize>=0.9.0",
        "smt==1.3.0",
        "pydacefit>=0.1.0",
        "pymoo>=0.5.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.10.0"],
    },
)
