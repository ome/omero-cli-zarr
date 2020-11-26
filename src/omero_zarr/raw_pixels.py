import argparse
import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy
import numpy as np
import omero.clients  # noqa
from omero.rtypes import unwrap
from zarr.hierarchy import Group, open_group

from . import __version__
from .util import print_status


def image_to_zarr(image: omero.gateway.ImageWrapper, args: argparse.Namespace) -> None:
    target_dir = args.output
    cache_dir = target_dir if args.cache_numpy else None

    name = os.path.join(target_dir, "%s.zarr" % image.id)
    print(f"Exporting to {name}")
    root = open_group(name, mode="w")
    n_levels = add_image(image, root, cache_dir=cache_dir)
    add_group_metadata(root, image, n_levels)
    add_toplevel_metadata(root)
    print("Finished.")


def add_image(
    image: omero.gateway.ImageWrapper, parent: Group, cache_dir: Optional[str] = None
) -> int:
    """ Adds the image pixel data as array to the given parent zarr group.
        Optionally caches the pixel data in the given cache_dir directory.
        Returns the number of resolution levels generated for the image.
    """
    if cache_dir is not None:
        cache = True
        os.makedirs(os.path.join(cache_dir, str(image.id)), mode=511, exist_ok=True)
    else:
        cache = False
        cache_dir = ""

    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_x = image.getSizeX()
    size_y = image.getSizeY()
    size_t = image.getSizeT()
    d_type = image.getPixelsType()

    zct_list = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                if cache:
                    # We only want to load from server if not cached locally
                    filename = os.path.join(
                        cache_dir, str(image.id), f"{z:03d}-{c:03d}-{t:03d}.npy",
                    )
                    if not os.path.exists(filename):
                        zct_list.append((z, c, t))
                else:
                    zct_list.append((z, c, t))

    pixels = image.getPrimaryPixels()

    def planeGen() -> np.ndarray:
        planes = pixels.getPlanes(zct_list)
        yield from planes

    planes = planeGen()

    # Target size for smallest multiresolution
    TARGET_SIZE = 96
    level_count = 1
    longest = max(size_x, size_y)
    while longest > TARGET_SIZE:
        longest = longest // 2
        level_count += 1

    field_groups = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                if cache:
                    filename = os.path.join(
                        cache_dir, str(image.id), f"{z:03d}-{c:03d}-{t:03d}.npy",
                    )
                    if os.path.exists(filename):
                        plane = numpy.load(filename)
                    else:
                        plane = next(planes)
                        numpy.save(filename, plane)
                else:
                    plane = next(planes)
                for level in range(level_count):
                    size_y = plane.shape[0]
                    size_x = plane.shape[1]
                    # If on first plane, create a new group for this resolution level
                    if t == 0 and c == 0 and z == 0:
                        field_groups.append(
                            parent.create(
                                str(level),
                                shape=(size_t, size_c, size_z, size_y, size_x),
                                chunks=(1, 1, 1, size_y, size_x),
                                dtype=d_type,
                            )
                        )

                    # field_group = field_groups[level]
                    field_groups[level][t, c, z, :, :] = plane

                    if (level + 1) < level_count:
                        # resize for next level...
                        plane = cv2.resize(
                            plane,
                            dsize=(size_x // 2, size_y // 2),
                            interpolation=cv2.INTER_NEAREST,
                        )
    return level_count


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

    target_dir = args.output
    cache_dir = target_dir if args.cache_numpy else None
    name = os.path.join(target_dir, "%s.zarr" % plate.id)
    print(f"Exporting to {name}")
    root = open_group(name, mode="w")

    count = 0
    max_fields = 0
    t0 = time.time()

    well_paths = set()

    col_names = plate.getColumnLabels()
    row_names = plate.getRowLabels()

    plate_metadata = {
        "name": plate.name,
        "rows": [{"name": str(name)} for name in row_names],
        "columns": [{"name": str(name)} for name in col_names],
        "version": "0.1",
    }
    # Add acquisitions key if at least one plate acquisition exists
    acquisitions = list(plate.listPlateAcquisitions())
    if acquisitions:
        plate_metadata["acquisitions"] = [marshal_acquisition(x) for x in acquisitions]
    root.attrs["plate"] = plate_metadata

    for well in plate.listChildren():
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
                n_levels = add_image(img, field_group, cache_dir=cache_dir)
                add_group_metadata(field_group, img, n_levels)
                # Update Well metadata after each image
                col_group.attrs["well"] = {"images": fields, "version": "0.1"}
                max_fields = max(max_fields, field + 1)
            print_status(int(t0), int(time.time()), count, total)

        # Update plate_metadata after each Well
        plate_metadata["wells"] = [{"path": x} for x in well_paths]
        plate_metadata["field_count"] = max_fields
        root.attrs["plate"] = plate_metadata

    add_toplevel_metadata(root)
    print("Finished.")


def add_group_metadata(
    zarr_root: Group, image: omero.gateway.ImageWrapper, resolutions: int = 1
) -> None:

    image_data = {
        "id": 1,
        "channels": [channelMarshal(c) for c in image.getChannels()],
        "rdefs": {
            "model": (image.isGreyscaleRenderingModel() and "greyscale" or "color"),
            "defaultZ": image._re.getDefaultZ(),
            "defaultT": image._re.getDefaultT(),
        },
        "version": "0.1",
    }
    multiscales = [
        {"version": "0.1", "datasets": [{"path": str(r)} for r in range(resolutions)]}
    ]
    zarr_root.attrs["multiscales"] = multiscales
    zarr_root.attrs["omero"] = image_data

    image._closeRE()


def add_toplevel_metadata(zarr_root: Group) -> None:

    zarr_root.attrs["_creator"] = {"name": "omero-zarr", "version": __version__}


def channelMarshal(channel: omero.model.Channel) -> Dict[str, Any]:
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
