#!/usr/bin/env python

import os

from setuptools import setup


def get_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()
    return long_description


long_description = get_long_description()

setup(
    name="omero-cli-zarr",
    version="0.0.8",
    packages=["omero_zarr", "omero.plugins"],
    package_dir={"": "src"},
    description="Plugin for exporting images in zarr format.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU General Public License v2 " "or later (GPLv2+)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    author="The Open Microscopy Team",
    author_email="",
    python_requires=">=3",
    install_requires=["omero-py>=5.6.0", "ome-zarr"],
    long_description=long_description,
    keywords=["OMERO.CLI", "plugin"],
    url="https://github.com/ome/omero-cli-zarr/",
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "src/omero_zarr/_version.py"},
)
