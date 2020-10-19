import argparse
import os
from typing import Any, Dict

import numpy
import numpy as np
import omero.clients  # noqa
from natsort import natsorted
from omero.rtypes import unwrap
from zarr.hierarchy import Group, open_group


def image_to_zarr(image: omero.gateway.Image, args: argparse.Namespace) -> None:

    cache_numpy = args.cache_numpy
    target_dir = args.output

    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_x = image.getSizeX()
    size_y = image.getSizeY()
    size_t = image.getSizeT()

    # dir for caching .npy planes
    if cache_numpy:
        os.makedirs(os.path.join(target_dir, str(image.id)), mode=511, exist_ok=True)
    name = os.path.join(target_dir, "%s.zarr" % image.id)
    za = None
    pixels = image.getPrimaryPixels()

    zct_list = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                # We only want to load from server if not cached locally
                filename = os.path.join(
                    target_dir, str(image.id), f"{z:03d}-{c:03d}-{t:03d}.npy",
                )
                if not os.path.exists(filename):
                    zct_list.append((z, c, t))

    def planeGen() -> np.ndarray:
        planes = pixels.getPlanes(zct_list)
        yield from planes

    planes = planeGen()

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                filename = os.path.join(
                    target_dir, str(image.id), f"{z:03d}-{c:03d}-{t:03d}.npy",
                )
                if os.path.exists(filename):
                    print(f"plane (from disk) c:{c}, t:{t}, z:{z}")
                    plane = numpy.load(filename)
                else:
                    print(f"loading plane c:{c}, t:{t}, z:{z}")
                    plane = next(planes)
                    if cache_numpy:
                        print(f"cached at {filename}")
                        numpy.save(filename, plane)
                if za is None:
                    # store = zarr.NestedDirectoryStore(name)
                    # root = zarr.group(store=store, overwrite=True)
                    root = open_group(name, mode="w")
                    za = root.create(
                        "0",
                        shape=(size_t, size_c, size_z, size_y, size_x),
                        chunks=(1, 1, 1, size_y, size_x),
                        dtype=plane.dtype,
                    )
                za[t, c, z, :, :] = plane
        add_group_metadata(root, image)
    print("Created", name)

def add_image(image: omero.gateway.Image, parent: Group, field_index="0") -> None:
    """Adds the image pixel data as array to the given parent zarr group."""
    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_x = image.getSizeX()
    size_y = image.getSizeY()
    size_t = image.getSizeT()
    d_type = image.getPixelsType()

    group = parent.create(
        field_index,
        shape=(size_t, size_c, size_z, size_y, size_x),
        chunks=(1, 1, 1, size_y, size_x),
        dtype=d_type,
    )

    zct_list = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                zct_list.append((z, c, t))

    pixels = image.getPrimaryPixels()
    def planeGen() -> np.ndarray:
        planes = pixels.getPlanes(zct_list)
        yield from planes

    planes = planeGen()

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = next(planes)
                group[t, c, z, :, :] = plane


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
    print(f"Plate size: rows={n_rows} x cols={n_cols} x fields={n_fields}")

    wells = {}
    for well in plate.listChildren():
        pos = well.getWellPos()
        row = pos[0]
        col = pos[1:]
        if row not in wells:
            wells[row] = {}
        wells[row][col] = well

    target_dir = args.output
    name = os.path.join(target_dir, "%s.zarr" % plate.id)
    root = open_group(name, mode="w")
    count = 0
    for row in natsorted(wells.keys()):
        row_wells = wells[row]
        row_group = root.create_group(row)
        for col in natsorted(row_wells.keys()):
            well = row_wells[col]
            col_group = row_group.create_group(col)
            for field in range(n_fields[0], n_fields[1] + 1):
                add_image(well.getImage(field), col_group, "Field_{}".format(field + 1))
                count += 1
                status = "row={}, col={}, field={} ({:.2f}% done)".format(
                    row, col, field, (count * 100 / total)
                )
                print(status, end="\r", flush=True)


def add_group_metadata(
    zarr_root: Group, image: omero.gateway.Image, resolutions: int = 1
) -> None:

    image_data = {
        "id": 1,
        "channels": [channelMarshal(c) for c in image.getChannels()],
        "rdefs": {
            "model": (image.isGreyscaleRenderingModel() and "greyscale" or "color"),
            "defaultZ": image._re.getDefaultZ(),
            "defaultT": image._re.getDefaultT(),
        },
    }
    multiscales = [
        {"version": "0.1", "datasets": [{"path": str(r)} for r in range(resolutions)]}
    ]
    zarr_root.attrs["multiscales"] = multiscales
    zarr_root.attrs["omero"] = image_data


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
