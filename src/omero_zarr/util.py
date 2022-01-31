import time
from typing import Dict, List

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
            "units": str(pix_size_x.getUnit()).lower(),
            "value": pix_size_x.getValue(),
        }
    if pix_size_y is not None:
        pixel_sizes["y"] = {
            "units": str(pix_size_y.getUnit()).lower(),
            "value": pix_size_y.getValue(),
        }
    if pix_size_z is not None:
        pixel_sizes["z"] = {
            "units": str(pix_size_z.getUnit()).lower(),
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
            axes[-1]["units"] = pixel_sizes["z"]["units"]
    # last 2 dimensions are always y and x
    for dim in ("y", "x"):
        axes.append({"name": dim, "type": "space"})
        if pixel_sizes and dim in pixel_sizes:
            axes[-1]["units"] = pixel_sizes[dim]["units"]

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
