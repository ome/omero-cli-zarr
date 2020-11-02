#!/usr/bin/env python

from setuptools import setup

setup(
    name="omero-cli-zarr",
    version="0.0.6",
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
    keywords=["OMERO.CLI", "plugin"],
    url="https://github.com/ome/omero-cli-zarr/",
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "src/omero_zarr/_version.py"},
)
