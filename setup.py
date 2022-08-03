import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="multiSyncPy",
    version="0.1.0",
    description="Functions to quantify multivariate synchrony",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/cslab-hub/multiSyncPy",
    author="Dan Hudson",
    author_email="daniel.dominic.hudson@uni-osnabrueck.de",
    license="GNU LGPL",
    classifiers=[
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["multiSyncPy"],
    ##include_package_data=True,
    install_requires=["numpy", "scipy", "sklearn", "seaborn", "matplotlib"]
)
