# setup.py
from setuptools import setup, find_packages

# read the contents of README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# read the contents of requirements.txt for the install_requires
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="ntf",
    version="0.1.0",
    author="Yuqiao Gong",
    author_email="gyq123@sjtu.edu.cn",
    description="Neural Transcriptomic Field (NTF) for 3D spatial omics reconstruction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/your_username/NTF_Project",
    packages=find_packages(where=".", exclude=["scripts", "configs", "data"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bioinformatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)