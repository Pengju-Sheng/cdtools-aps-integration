import setuptools
import os
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

# read version from src/cdtools-aps-integration/_version.py
version_file = os.path.join("src/cdtools-aps-integration", "_version.py")
print(version_file)
with open(version_file) as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
if not version_match:
    raise RuntimeError("Unable to find version string.")
version = version_match.group(1)
    
setuptools.setup(
    name="cdtools-aps-integration",
    version=version,
    python_requires='>3.8', # recommended minimum version for pytorch 2.3.0
    author="Pengju Sheng",
    author_email="pengju.sheng@psi.ch",
    description="Functions to integrate cdtools with the systems at APS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pengju-Sheng/cdtools-aps-integration.git",
    install_requires=[
        "cdtools-py>=0.3.2",
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0", 
        "torch>=2.3.0",
        "h5py>=2.1",
        "tqdm",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

