#!/usr/bin/env python

# Copyright (C) 2023 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os

from setuptools import setup


def get_long_description() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()
    return long_description


long_description = get_long_description()

setup(
    name="omero-cli-zarr",
    version="0.6.1",
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
    install_requires=["omero-py>=5.6.0", "ome-zarr>=0.5.0,<0.12.0"],
    long_description=long_description,
    keywords=["OMERO.CLI", "plugin"],
    url="https://github.com/ome/omero-cli-zarr/",
    setup_requires=["setuptools_scm==7.1.0"],
    use_scm_version={"write_to": "src/omero_zarr/_version.py"},
    tests_require=["omero-py>=5.18.0", "pytest", "omero-rois"],
)
