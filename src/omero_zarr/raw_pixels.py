import argparse
import os
import time
from typing import Any, Dict

import cv2
import numpy
import numpy as np
import omero.clients  # noqa
from omero.rtypes import unwrap
from zarr.hierarchy import Group, open_group


def image_to_zarr(image: omero.gateway.ImageWrapper, args: argparse.Namespace) -> None:

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


def add_image(
    image: omero.gateway.ImageWrapper, parent: Group, field_index: str = "0"
) -> None:
    """Adds the image pixel data as array to the given parent zarr group."""
    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_x = image.getSizeX()
    size_y = image.getSizeY()
    size_t = image.getSizeT()
    d_type = image.getPixelsType()

    field_group = parent.require_group(field_index)

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

    # Target size for smallest multiresolution
    TARGET_SIZE = 96
    level_count = 1
    longest = max(size_x, size_y)
    while longest > TARGET_SIZE:
        longest = longest // 2
        level_count += 1

    add_group_metadata(field_group, image, level_count)

    field_groups = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = next(planes)
                for level in range(level_count):
                    size_y = plane.shape[0]
                    size_x = plane.shape[1]
                    # If on first plane, create a new group for this resolution level
                    if t == 0 and c == 0 and z == 0:
                        field_groups.append(
                            field_group.create(
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


def print_status(t0: float, t: float, count: int, total: int) -> None:
    """ Prints percent done and ETA """
    percent_done = count * 100 / total
    rate = count / (t - t0)
    eta = (total - count) / rate
    status = "{:.2f}% done, ETA: {}".format(
        percent_done, time.strftime("%H:%M:%S", time.gmtime(eta))
    )
    print(status, end="\r", flush=True)


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
    name = os.path.join(target_dir, "%s.zarr" % plate.id)
    print(f"Exporting to {name}")
    root = open_group(name, mode="w")
    plate_metadata = {"rows": n_rows, "columns": n_cols}
    root.attrs["plate"] = plate_metadata

    count = 0
    t0 = time.time()

    for well in plate.listChildren():
        row = plate.getRowLabels()[well.row]
        col = plate.getColumnLabels()[well.column]
        for field in range(n_fields[0], n_fields[1] + 1):
            ws = well.getWellSample(field)
            field_name = "Field_{}".format(field + 1)
            count += 1
            if ws and ws.getImage():
                img = ws.getImage()
                ac = ws.getPlateAcquisition()
                ac_name = ac.getName() if ac else "0"
                ac_group = root.require_group(ac_name)
                row_group = ac_group.require_group(row)
                col_group = row_group.require_group(col)
                add_image(img, col_group, field_name)
            print_status(t0, time.time(), count, total)
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
