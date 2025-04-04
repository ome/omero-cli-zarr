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

import time
from typing import Any, Dict, List

import dask.array as da
import numpy as np
import skimage.transform
from omero.gateway import ImageWrapper
from zarr.storage import FSStore


def print_status(t0: int, t: int, count: int, total: int) -> None:
    """Prints percent done and ETA.
    t0: start timestamp in seconds
    t: current timestamp in seconds
    count: number of tasks done
    total: total number of tasks
    """
    percent_done = float(count) * 100 / total
    dt = t - t0
    if dt > 0:
        rate = float(count) / (t - t0)
        eta_f = float(total - count) / rate
        eta = time.strftime("%H:%M:%S", time.gmtime(eta_f))
    else:
        eta = "NA"
    status = f"{percent_done:.2f}% done, ETA: {eta}"
    print(status, end="\r", flush=True)


def open_store(name: str) -> FSStore:
    """
    Create an FSStore instance that supports nested storage of chunks.
    """
    return FSStore(
        name,
        auto_mkdir=True,
        key_separator="/",
        normalize_keys=False,
        mode="w",
    )


def marshal_pixel_sizes(image: ImageWrapper) -> Dict[str, Dict]:
    pixel_sizes: Dict[str, Dict] = {}
    pix_size_x = image.getPixelSizeX(units=True)
    pix_size_y = image.getPixelSizeY(units=True)
    pix_size_z = image.getPixelSizeZ(units=True)
    # All OMERO units.lower() are valid UDUNITS-2 and therefore NGFF spec
    if pix_size_x is not None:
        pixel_sizes["x"] = {
            "unit": str(pix_size_x.getUnit()).lower(),
            "value": pix_size_x.getValue(),
        }
    if pix_size_y is not None:
        pixel_sizes["y"] = {
            "unit": str(pix_size_y.getUnit()).lower(),
            "value": pix_size_y.getValue(),
        }
    if pix_size_z is not None:
        pixel_sizes["z"] = {
            "unit": str(pix_size_z.getUnit()).lower(),
            "value": pix_size_z.getValue(),
        }
    return pixel_sizes


def marshal_axes(image: ImageWrapper) -> List[Dict]:
    # Prepare axes and transformations info...
    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_t = image.getSizeT()
    pixel_sizes = marshal_pixel_sizes(image)

    axes = []
    if size_t > 1:
        axes.append({"name": "t", "type": "time"})
    if size_c > 1:
        axes.append({"name": "c", "type": "channel"})
    if size_z > 1:
        axes.append({"name": "z", "type": "space"})
        if pixel_sizes and "z" in pixel_sizes:
            axes[-1]["unit"] = pixel_sizes["z"]["unit"]
    # last 2 dimensions are always y and x
    for dim in ("y", "x"):
        axes.append({"name": dim, "type": "space"})
        if pixel_sizes and dim in pixel_sizes:
            axes[-1]["unit"] = pixel_sizes[dim]["unit"]

    return axes


def marshal_transformations(
    image: ImageWrapper, levels: int = 1, multiscales_zoom: float = 2.0
) -> List[List[Dict]]:
    axes = marshal_axes(image)
    pixel_sizes = marshal_pixel_sizes(image)

    # Each path needs a transformations list...
    transformations = []
    zooms = {"x": 1.0, "y": 1.0, "z": 1.0, "c": 1.0, "t": 1.0}
    for level in range(levels):
        # {"type": "scale", "scale": [1, 1, 0.3, 0.5, 0.5]
        scales = []
        for index, axis in enumerate(axes):
            pixel_size = 1
            if axis["name"] in pixel_sizes:
                pixel_size = pixel_sizes[axis["name"]].get("value", 1)
            scales.append(zooms[axis["name"]] * pixel_size)
        # ...with a single 'scale' transformation each
        transformations.append([{"type": "scale", "scale": scales}])
        # NB we rescale X and Y for each level, but not Z, C, T
        zooms["x"] = zooms["x"] * multiscales_zoom
        zooms["y"] = zooms["y"] * multiscales_zoom

    return transformations


def resize(
    image: da.Array, output_shape: tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    r"""
    Wrapped copy of "skimage.transform.resize"
    Resize image to match a certain size.
    :type image: :class:`dask.array`
    :param image: The dask array to resize
    :type output_shape: tuple
    :param output_shape: The shape of the resize array
    :type \*args: list
    :param \*args: Arguments of skimage.transform.resize
    :type \*\*kwargs: dict
    :param \*\*kwargs: Keyword arguments of skimage.transform.resize
    :return: Resized image.
    """
    factors = np.array(output_shape) / np.array(image.shape).astype(float)
    # Rechunk the input blocks so that the factors achieve an output
    # blocks size of full numbers.
    better_chunksize = tuple(
        np.maximum(1, np.round(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    # If E.g. we resize image from 6675 by 0.5 to 3337, factor is 0.49992509 so each
    # chunk of size e.g. 1000 will resize to 499. When assumbled into a new array, the
    # array will now be of size 3331 instead of 3337 because each of 6 chunks was
    # smaller by 1. When we compute() this, dask will read 6 chunks of 1000 and expect
    # last chunk to be 337 but instead it will only be 331.
    # So we use ceil() here (and in resize_block) to round 499.925 up to chunk of 500
    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    # Map overlap
    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        # if the input block is smaller than a 'regular' chunk (e.g. edge of image)
        # we need to calculate target size for each chunk...
        chunk_output_shape = tuple(
            np.ceil(np.array(image_block.shape) * factors).astype(int)
        )
        return skimage.transform.resize(
            image_block, chunk_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)
