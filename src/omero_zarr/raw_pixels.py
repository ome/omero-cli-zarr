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

import argparse
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import omero.clients  # noqa
import omero.gateway  # required to allow 'from omero_zarr import raw_pixels'
from ome_zarr.dask_utils import resize as da_resize
from ome_zarr.writer import (
    write_multiscales_metadata,
    write_plate_metadata,
    write_well_metadata,
)
from omero.model import Channel
from omero.model.enums import (
    PixelsTypedouble,
    PixelsTypefloat,
    PixelsTypeint8,
    PixelsTypeint16,
    PixelsTypeint32,
    PixelsTypeuint8,
    PixelsTypeuint16,
    PixelsTypeuint32,
)
from omero.rtypes import unwrap
from zarr.hierarchy import Group, open_group

from . import __version__
from . import ngff_version as VERSION
from .util import (
    get_zarr_name,
    marshal_axes,
    marshal_transformations,
    open_store,
    print_status,
)


def image_to_zarr(image: omero.gateway.ImageWrapper, args: argparse.Namespace) -> None:
    tile_width = args.tile_width
    tile_height = args.tile_height
    name = get_zarr_name(image, args.output, args.name_by)
    print(f"Exporting to {name} ({VERSION})")
    store = open_store(name)
    root = open_group(store)
    add_image(image, root, tile_width=tile_width, tile_height=tile_height)
    add_omero_metadata(root, image)
    add_toplevel_metadata(root)
    print("Finished.")


def add_image(
    image: omero.gateway.ImageWrapper,
    parent: Group,
    tile_width: Optional[int] = None,
    tile_height: Optional[int] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Adds an OMERO image pixel data as array to the given parent zarr group.
    Returns the number of resolution levels generated for the image.
    """

    size_x = image.getSizeX()
    size_y = image.getSizeY()

    # Target size for smallest multiresolution
    TARGET_SIZE = 96
    level_count = 1
    longest = max(size_x, size_y)
    while longest > TARGET_SIZE:
        longest = longest // 2
        level_count += 1

    paths = add_raw_image(image, parent, level_count, tile_width, tile_height)

    axes = marshal_axes(image)
    transformations = marshal_transformations(image, len(paths))

    datasets: List[Dict[Any, Any]] = [{"path": path} for path in paths]
    for dataset, transform in zip(datasets, transformations):
        dataset["coordinateTransformations"] = transform

    write_multiscales_metadata(parent, datasets, axes=axes)

    return (level_count, axes)


def add_raw_image(
    image: omero.gateway.ImageWrapper,
    parent: Group,
    level_count: int,
    tile_width: Optional[int] = None,
    tile_height: Optional[int] = None,
) -> List[str]:
    pixels = image.getPrimaryPixels()
    omero_dtype = image.getPixelsType()
    pixelTypes = {
        PixelsTypeint8: ["b", np.int8],
        PixelsTypeuint8: ["B", np.uint8],
        PixelsTypeint16: ["h", np.int16],
        PixelsTypeuint16: ["H", np.uint16],
        PixelsTypeint32: ["i", np.int32],
        PixelsTypeuint32: ["I", np.uint32],
        PixelsTypefloat: ["f", np.float32],
        PixelsTypedouble: ["d", np.float64],
    }
    d_type = pixelTypes[omero_dtype][1]
    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_x = image.getSizeX()
    size_y = image.getSizeY()
    size_t = image.getSizeT()

    if tile_width is None:
        tile_width = 1024
    if tile_height is None:
        tile_height = 1024
    tile_width = int(tile_width)
    tile_height = int(tile_height)

    print(
        "sizes x: %s, y: %s, z: %s, c: %s, t: %s"
        % (size_x, size_y, size_z, size_c, size_t)
    )
    print(f"tile_width: {tile_width}, tile_height: {tile_height}")

    chunk_count_x = math.ceil(size_x / tile_width)
    chunk_count_y = math.ceil(size_y / tile_height)

    # create "0" array if it doesn't exist
    path = "0"
    dims = [dim for dim in [size_t, size_c, size_z] if dim != 1]
    shape = tuple(dims + [size_y, size_x])
    chunks = tuple([1] * len(dims) + [tile_height, tile_width])
    zarray = parent.require_dataset(
        path,
        shape=shape,
        exact=True,
        chunks=chunks,
        dtype=d_type,
    )

    # Need to be sure that dims match (if array already existed)
    assert zarray.shape == shape
    msg = f"Chunks mismatch: existing {zarray.chunks} requested {chunks}"
    assert zarray.chunks == chunks, msg

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                for chk_y in range(chunk_count_y):
                    for chk_x in range(chunk_count_x):
                        print(
                            "t, c, z, chk_x, chk_y %s %s %s %s %s"
                            % (t, c, z, chk_y, chk_x)
                        )
                        x = tile_width * chk_x
                        y = tile_height * chk_y

                        y_max = min(size_y, y + tile_height)
                        x_max = min(size_x, x + tile_width)

                        tile_dims = (x, y, x_max - x, y_max - y)

                        indices = []
                        if size_t > 1:
                            indices.append(t)
                        if size_c > 1:
                            indices.append(c)
                        if size_z > 1:
                            indices.append(z)

                        indices.append(np.s_[y:y_max:])
                        indices.append(np.s_[x:x_max:])

                        # Check if chunk exists. If not load from OMERO
                        existing_data = zarray[tuple(indices)]
                        if existing_data.max() == 0:
                            print("loading Tile...")
                            tile = pixels.getTile(z, c, t, tile_dims)
                            zarray[tuple(indices)] = tile

    paths = [str(level) for level in range(level_count)]

    downsample_pyramid_on_disk(parent, paths)
    return paths


def downsample_pyramid_on_disk(parent: Group, paths: List[str]) -> List[str]:
    """
    Takes a high-resolution Zarr array at paths[0] in the zarr group
    and down-samples it by a factor of 2 for each of the other paths
    """
    group_path = parent.store.path
    image_path = os.path.join(group_path, parent.path)
    print("downsample_pyramid_on_disk", image_path)
    for count, path in enumerate(paths[1:]):
        target_path = os.path.join(image_path, path)
        if os.path.exists(target_path):
            print("path exists: %s" % target_path)
            continue
        # open previous resolution from disk via dask...
        path_to_array = os.path.join(image_path, paths[count])
        dask_image = da.from_zarr(path_to_array)

        # resize in X and Y
        dims = list(dask_image.shape)
        dims[-1] = dims[-1] // 2
        dims[-2] = dims[-2] // 2
        output = da_resize(
            dask_image, tuple(dims), preserve_range=True, anti_aliasing=False
        )

        # write to disk
        da.to_zarr(
            arr=output,
            url=image_path,
            component=path,
            dimension_separator=parent._store._dimension_separator,
        )

    return paths


def marshal_acquisition(acquisition: omero.gateway._PlateAcquisitionWrapper) -> Dict:
    """Marshal a PlateAcquisitionWrapper to JSON"""
    acq = {
        "id": acquisition.id,
        "name": acquisition.name,
        "maximumfieldcount": acquisition.maximumFieldCount,
    }
    if acquisition.description:
        acq["description"] = acquisition.description
    if acquisition.startTime:
        acq["starttime"] = acquisition.startTime
    if acquisition.endTime:
        acq["endtime"] = acquisition.endTime
    return acq


def plate_to_zarr(plate: omero.gateway._PlateWrapper, args: argparse.Namespace) -> None:
    """
    Exports a plate to a zarr file using the hierarchy discussed here ('Option 3'):
    https://github.com/ome/omero-ms-zarr/issues/73#issuecomment-706770955
    """
    gs = plate.getGridSize()
    n_rows = gs["rows"]
    n_cols = gs["columns"]
    n_fields = plate.getNumberOfFields()
    total = n_rows * n_cols * (n_fields[1] - n_fields[0] + 1)
    name = get_zarr_name(plate, args.output, args.name_by)

    store = open_store(name)
    print(f"Exporting to {name} ({VERSION})")
    root = open_group(store)

    count = 0
    max_fields = 0
    t0 = time.time()

    well_paths = set()

    col_names = [str(name) for name in plate.getColumnLabels()]
    row_names = [str(name) for name in plate.getRowLabels()]

    # Add acquisitions key if at least one plate acquisition exists
    acquisitions = list(plate.listPlateAcquisitions())
    plate_acq = None
    if acquisitions:
        plate_acq = [marshal_acquisition(x) for x in acquisitions]

    wells = plate.listChildren()
    # sort by row then column...
    wells = sorted(wells, key=lambda x: (x.row, x.column))

    for well in wells:
        row = plate.getRowLabels()[well.row]
        col = plate.getColumnLabels()[well.column]
        fields = []
        for field in range(n_fields[0], n_fields[1] + 1):
            ws = well.getWellSample(field)
            if ws and ws.getImage():
                ac = ws.getPlateAcquisition()
                field_name = "%d" % field
                count += 1
                img = ws.getImage()
                well_paths.add(f"{row}/{col}")
                field_info = {"path": f"{field_name}"}
                if ac:
                    field_info["acquisition"] = ac.id
                fields.append(field_info)
                row_group = root.require_group(row)
                col_group = row_group.require_group(col)
                field_group = col_group.require_group(field_name)
                add_image(img, field_group)
                add_omero_metadata(field_group, img)
                # Update Well metadata after each image
                write_well_metadata(col_group, fields)
                max_fields = max(max_fields, field + 1)
            print_status(int(t0), int(time.time()), count, total)

        # Update plate_metadata after each Well
        if len(well_paths) == 0:
            continue
        write_plate_metadata(
            root,
            row_names,
            col_names,
            wells=list(well_paths),
            field_count=max_fields,
            acquisitions=plate_acq,
            name=plate.name,
        )

    add_toplevel_metadata(root)
    print("Finished.")


def add_omero_metadata(zarr_root: Group, image: omero.gateway.ImageWrapper) -> None:
    image_data = {
        "id": 1,
        "channels": [channelMarshal(c) for c in image.getChannels()],
        "rdefs": {
            "model": (image.isGreyscaleRenderingModel() and "greyscale" or "color"),
            "defaultZ": image._re.getDefaultZ(),
            "defaultT": image._re.getDefaultT(),
        },
        "version": VERSION,
    }
    zarr_root.attrs["omero"] = image_data
    image._closeRE()


def add_toplevel_metadata(zarr_root: Group) -> None:
    zarr_root.attrs["_creator"] = {"name": "omero-zarr", "version": __version__}


def channelMarshal(channel: Channel) -> Dict[str, Any]:
    return {
        "label": channel.getLabel(),
        "color": channel.getColor().getHtml(),
        "inverted": channel.isInverted(),
        "family": unwrap(channel.getFamily()),
        "coefficient": unwrap(channel.getCoefficient()),
        "window": {
            "min": channel.getWindowMin(),
            "max": channel.getWindowMax(),
            "start": channel.getWindowStart(),
            "end": channel.getWindowEnd(),
        },
        "active": channel.isActive(),
    }
