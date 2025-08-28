#!/usr/bin/env python

#
# Copyright (C) 2025 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

import omero
import zarr
from numpy import finfo, iinfo
from omero.gateway import BlitzGateway, ImageWrapper
from omero.model import ExternalInfoI
from omero.model.enums import (
    PixelsTypecomplex,
    PixelsTypedouble,
    PixelsTypefloat,
    PixelsTypeint8,
    PixelsTypeint16,
    PixelsTypeint32,
    PixelsTypeuint8,
    PixelsTypeuint16,
    PixelsTypeuint32,
)
from omero.rtypes import rbool, rdouble, rint, rlong, rstring, rtime
from zarr.core import Array
from zarr.creation import open_array
from zarr.errors import ArrayNotFoundError, GroupNotFoundError
from zarr.hierarchy import open_group
from zarr.storage import FSStore

from .import_xml import full_import

# TODO: support Zarr v3 - imports for get_omexml_bytes()
# from zarr.core.buffer import default_buffer_prototype
# from zarr.core.sync import sync


AWS_DEFAULT_ENDPOINT = "s3.us-east-1.amazonaws.com"

PIXELS_TYPE = {
    "int8": PixelsTypeint8,
    "int16": PixelsTypeint16,
    "uint8": PixelsTypeuint8,
    "uint16": PixelsTypeuint16,
    "int32": PixelsTypeint32,
    "uint32": PixelsTypeuint32,
    "float_": PixelsTypefloat,
    "float8": PixelsTypefloat,
    "float16": PixelsTypefloat,
    "float32": PixelsTypefloat,
    "float64": PixelsTypedouble,
    "complex_": PixelsTypecomplex,
    "complex64": PixelsTypecomplex,
}


def get_omexml_bytes(store: zarr.storage.Store) -> Optional[bytes]:
    # Zarr v3 get() is async. Need to sync to get the bytes
    # rsp = store.get("OME/METADATA.ome.xml", prototype=default_buffer_prototype())
    # result = sync(rsp)
    # if result is None:
    #     return None
    # return result.to_bytes()

    # Zarr v2
    return store.get("OME/METADATA.ome.xml")


def format_s3_uri(uri: str, endpoint: str) -> str:
    """
    Combine endpoint and uri
    """
    parsed_uri = urlsplit(uri)
    url = f"{parsed_uri.netloc}"
    if endpoint:
        parsed_endpoint = urlsplit(endpoint)
        endpoint = f"{parsed_endpoint.netloc}"
    else:
        endpoint = AWS_DEFAULT_ENDPOINT
    return f"{parsed_uri.scheme}" + "://" + endpoint + "/" + url + f"{parsed_uri.path}"


def load_array(store: zarr.storage.Store, path: Optional[str] = None) -> Array:
    return open_array(store=store, mode="r", path=path)


def load_attrs(store: zarr.storage.StoreLike, path: Optional[str] = None) -> dict:
    """
    Load the attrs from the root group or path subgroup
    """
    root = open_group(store=store, mode="r", path=path)
    attrs = root.attrs.asdict()
    if "ome" in attrs:
        attrs = attrs["ome"]
    return attrs


def parse_image_metadata(
    store: zarr.storage.Store, img_attrs: dict, image_path: Optional[str] = None
) -> tuple:
    """
    Parse the image metadata
    """
    multiscale_attrs = img_attrs["multiscales"][0]
    array_path = multiscale_attrs["datasets"][0]["path"]
    if image_path is not None:
        array_path = image_path.rstrip("/") + "/" + array_path
    # load .zarray from path to know the dimension
    array_data = load_array(store, array_path)
    sizes = {}
    shape = array_data.shape
    axes = multiscale_attrs.get("axes")
    # Need to check the older version
    if axes:
        for axis, size in zip(axes, shape):
            if isinstance(axis, str):
                sizes[axis] = size  # v0.3
            else:
                sizes[axis["name"]] = size

    pixels_type = array_data.dtype.name
    return sizes, pixels_type


def create_image(
    conn: BlitzGateway,
    store: zarr.storage.Store,
    image_attrs: dict,
    object_name: str,
    families: list,
    models: list,
    kwargs: dict,
    image_path: Optional[str] = None,
) -> tuple:
    """
    Create an Image/Pixels object
    """
    query_service = conn.getQueryService()
    pixels_service = conn.getPixelsService()
    sizes, pixels_type = parse_image_metadata(store, image_attrs, image_path)
    size_t = sizes.get("t", 1)
    size_z = sizes.get("z", 1)
    size_x = sizes.get("x", 1)
    size_y = sizes.get("y", 1)

    channels = list(range(sizes.get("c", 1)))
    omero_pixels_type = query_service.findByQuery(
        "from PixelsType as p where p.value='%s'" % PIXELS_TYPE[pixels_type], None
    )
    iid = pixels_service.createImage(
        size_x,
        size_y,
        size_z,
        size_t,
        channels,
        omero_pixels_type,
        object_name,
        "",
        conn.SERVICE_OPTS,
    )
    iid = iid.getValue()

    rnd_def = None
    image = conn.getObject("Image", iid)
    # Set rendering settings and channel names if omero_attrs is provided
    rnd_def = set_rendering_settings(
        conn, image, image_attrs, pixels_type, families, models
    )

    img_obj = image._obj
    set_external_info(img_obj, kwargs, image_path)

    return img_obj, rnd_def


def hex_to_rgba(hex_color: str) -> list:
    """
    Converts a hex color code to an RGB array.
    """
    if len(hex_color) == 3:
        hex_color = hex_color[0] * 2 + hex_color[1] * 2 + hex_color[2] * 2
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r, g, b]


def get_channels(omero_info: dict) -> list:
    """
    Find the name of the channels if specified
    """
    channel_names = []
    if omero_info is None:
        return channel_names
    for index, entry in enumerate(omero_info.get("channels", [])):
        channel_names.append(entry.get("label", index))
    return channel_names


def set_channel_names(conn: BlitzGateway, iid: int, omero_attrs: dict) -> None:
    channel_names = get_channels(omero_attrs)
    nameDict = {i + 1: name for i, name in enumerate(channel_names)}
    conn.setChannelNames("Image", [iid], nameDict)


def set_rendering_settings(
    conn: BlitzGateway,
    image: ImageWrapper,
    image_attrs: dict,
    pixels_type: str,
    families: Optional[list] = None,
    models: Optional[list] = None,
) -> Optional[omero.model.RenderingDefI]:
    """
    Extract the rendering settings and the channels information
    """
    omero_info = image_attrs.get("omero", None)
    if omero_info is None:
        return None
    set_channel_names(conn, image.id, omero_info)

    if families is None:
        families = load_families(conn)
    if models is None:
        models = load_models(conn)

    pixels_id = image.getPrimaryPixels().getId()

    if omero_info is None:
        return None
    rdefs = omero_info.get("rdefs", None)
    if rdefs is None:
        rdefs = dict()
    rnd_def = omero.model.RenderingDefI()
    rnd_def.version = rint(0)
    rnd_def.defaultZ = rint(rdefs.get("defaultZ", 0))
    rnd_def.defaultT = rint(rdefs.get("defaultT", 0))
    value = rdefs.get("model", "rgb")
    if value == "color":
        value = "rgb"
    ref_model = None
    for m in models:
        mv = m.getValue()._val
        if mv == "rgb":
            ref_model = m
        if mv == value:
            rnd_def.model = m
    if rnd_def.model is None:
        rnd_def.model = ref_model

    q_def = omero.model.QuantumDefI()
    q_def.cdStart = rint(0)
    q_def.cdEnd = rint(255)
    # Flag to select a 8-bit depth (<i>=2^8-1</i>) output interval
    q_def.bitResolution = rint(255)
    rnd_def.quantization = q_def
    rnd_def.pixels = omero.model.PixelsI(pixels_id, False)

    if pixels_type.startswith("float"):
        pixels_min = finfo(pixels_type).min
        pixels_max = finfo(pixels_type).max
    else:
        pixels_min = iinfo(pixels_type).min
        pixels_max = iinfo(pixels_type).max
    for entry in omero_info.get("channels", []):
        cb = omero.model.ChannelBindingI()
        rnd_def.addChannelBinding(cb)
        cb.coefficient = rdouble(entry.get("coefficient", 1.0))
        cb.active = rbool(entry.get("active", False))
        value = entry.get("family", "linear")
        ref_family = None
        for f in families:
            fv = f.getValue()._val
            if fv == "linear":
                ref_family = f
            if fv == value:
                cb.family = f
        if cb.family is None:
            cb.family = ref_family

        # convert color to rgba
        rgb = hex_to_rgba(
            entry.get("color", "000000").lstrip("#")
        )  # default to black is no color set
        cb.red = rint(rgb[0])
        cb.green = rint(rgb[1])
        cb.blue = rint(rgb[2])
        cb.alpha = rint(255)
        cb.noiseReduction = rbool(False)

        window = entry.get("window", {})
        try:
            cb.inputStart = rdouble(float(window.get("start", pixels_min)))
        except TypeError:
            cb.inputStart = rdouble(pixels_min)
        try:
            cb.inputEnd = rdouble(float(window.get("end", pixels_max)))
        except TypeError:
            cb.inputEnd = rdouble(pixels_max)
        inverted = entry.get("inverted", False)
        if inverted:  # add codomain
            ric = omero.model.ReverseIntensityContextI()
            ric.reverse = rbool(inverted)
            cb.addCodomainMapContext(ric)
    return rnd_def


def load_families(conn: BlitzGateway) -> list:
    ctx = {"omero.group": "-1"}
    return conn.getQueryService().findAllByQuery("select f from Family as f", None, ctx)


def load_models(conn: BlitzGateway) -> list:
    ctx = {"omero.group": "-1"}
    return conn.getQueryService().findAllByQuery(
        "select f from RenderingModel as f", None, ctx
    )


def register_image(
    conn: BlitzGateway,
    store: zarr.storage.Store,
    kwargs: dict,
    img_attrs: Optional[dict] = None,
    image_path: Optional[str] = None,
) -> tuple:
    """
    Register the ome.zarr image in OMERO.
    """

    update_service = conn.getUpdateService()
    families = load_families(conn)
    models = load_models(conn)

    if img_attrs is None:
        img_attrs = load_attrs(store, image_path)
    if kwargs and kwargs.get("name"):
        image_name = kwargs["name"]
    elif "name" in img_attrs:
        image_name = img_attrs["name"]
    else:
        image_name = kwargs.get("uri", "").rstrip("/").split("/")[-1]
        if image_path is not None:
            image_name = f"{image_name} [{image_path}]"
    image, rnd_def = create_image(
        conn,
        store,
        img_attrs,
        image_name,
        families,
        models,
        kwargs=kwargs,
        image_path=image_path,
    )
    update_service.saveAndReturnObject(image)
    if rnd_def is not None:
        update_service.saveAndReturnObject(rnd_def)

    print("Created Image", image.id.val)
    return image


def determine_naming(values: list[dict]) -> str:
    """
    Determine the name of columns or rows of a plate
    """
    if len(values) > 0:
        value = values[0]["name"]
        if value.isdigit():
            return "number"
    return "letter"


def create_plate_acquisition(pa: dict) -> omero.model.PlateAcquisitionI:
    """
    Create a plate acquisition object
    """
    plate_acquisition = omero.model.PlateAcquisitionI()
    if pa.get("name"):
        plate_acquisition.name = rstring(pa.get("name"))
    else:
        plate_acquisition.name = rstring(pa.get("id"))
    if pa.get("maximumfieldcount"):
        plate_acquisition.maximumFieldCount = rint(pa.get("maximumfieldcount"))
    if pa.get("starttime"):
        plate_acquisition.startTime = rtime(pa.get("starttime"))
    if pa.get("endtime"):
        plate_acquisition.endTime = rtime(pa.get("endtime"))
    return plate_acquisition


def register_plate(
    conn: BlitzGateway, store: zarr.storage.Store, attrs: dict, kwargs: dict
) -> omero.model.PlateI:
    """
    Register a plate
    """

    plate_attrs = attrs["plate"]

    object_name = kwargs.get("name", None)
    if object_name is None:
        object_name = plate_attrs.get("name", None)
    if object_name is None:
        object_name = kwargs["uri"].rstrip("/").split("/")[-1].split(".")[0]

    update_service = conn.getUpdateService()
    families = load_families(conn)
    models = load_models(conn)

    # Create a plate
    plate = omero.model.PlateI()
    plate.name = rstring(object_name)
    plate.columnNamingConvention = rstring(determine_naming(plate_attrs["columns"]))
    plate.rowNamingConvention = rstring(determine_naming(plate_attrs["rows"]))
    plate.rows = rint(len(plate_attrs["rows"]))
    plate.columns = rint(len(plate_attrs["columns"]))

    acquisitions = plate_attrs.get("acquisitions")
    plate_acquisitions = {}
    if acquisitions is not None and len(acquisitions) > 1:
        for pa in acquisitions:
            plate_acquisition = create_plate_acquisition(pa)
            plate.addPlateAcquisition(plate_acquisition)

    plate = update_service.saveAndReturnObject(plate)
    print("Plate created with id:", plate.id.val)

    # load the new plate acquisitions and map them to the original IDs
    if acquisitions is not None and len(acquisitions) > 1:
        pwrapper = conn.getObject("Plate", plate.id.val)
        pas = list(pwrapper.listPlateAcquisitions())
        for pa, saved in zip(acquisitions, pas):
            plate_acquisitions[pa["id"]] = saved.id
        print("plate_acquisitions", plate_acquisitions)

    # for bug in omero-cli-zarr - need to handle dupliate Wells!
    well_paths = []

    well_count = len(plate_attrs["wells"])
    for well_index, well_attrs in enumerate(plate_attrs["wells"]):
        images_to_save = []
        rnd_defs = []
        # read metadata
        row_index = well_attrs["rowIndex"]
        column_index = well_attrs["columnIndex"]
        well_path = well_attrs["path"]
        if well_path in well_paths:
            continue
        else:
            well_paths.append(well_path)
        print("well_path", well_path, f"({well_index}/{well_count})")
        # create OMERO object
        well = omero.model.WellI()
        well.plate = omero.model.PlateI(plate.getId(), False)
        well.column = rint(column_index)
        well.row = rint(row_index)

        well_attrs = load_attrs(store, well_path)
        well_samples_attrs = well_attrs["well"]["images"]

        for sample_attrs in well_samples_attrs:
            image_path = f"{well_path}/{sample_attrs['path']}/"

            img_attrs = load_attrs(store, image_path)
            image_name = img_attrs.get("name", f"{well_path}/{sample_attrs['path']}")

            image, rnd_def = create_image(
                conn, store, img_attrs, image_name, families, models, kwargs, image_path
            )

            images_to_save.append(image)
            if rnd_def is not None:
                rnd_defs.append(rnd_def)
            # Link well sample and plate acquisition
            ws = omero.model.WellSampleI()
            if "acquisition" in sample_attrs:
                acquisition_id = sample_attrs["acquisition"]
                pa_id = plate_acquisitions.get(acquisition_id)
                if pa is not None:
                    ws.plateAcquisition = omero.model.PlateAcquisitionI(pa_id, False)
            ws.image = omero.model.ImageI(image.id.val, False)
            ws.well = well
            well.addWellSample(ws)

        # Save each Well and Images as we go...
        update_service.saveObject(well)
        update_service.saveAndReturnArray(images_to_save)
        if len(rnd_defs) > 0:
            update_service.saveAndReturnIds(rnd_defs)

    print("Plate created with id:", plate.id.val)
    return plate


def set_external_info(
    image: omero.model.ImageI,
    kwargs: dict,
    image_path: Optional[str] = None,
) -> None:
    """
    Create the external info and link it to the image
    """
    extinfo = ExternalInfoI()
    # non-nullable properties
    setattr(extinfo, "entityId", rlong(3))
    setattr(extinfo, "entityType", rstring("com.glencoesoftware.ngff:multiscales"))

    uri = kwargs.get("uri", "")
    endpoint = kwargs.get("endpoint", "")
    nosignrequest = kwargs.get("nosignrequest", False)

    if image_path is not None:
        uri = uri.rstrip("/") + "/" + image_path
    parsed_uri = urlsplit(uri)
    scheme = f"{parsed_uri.scheme}"

    if "http" in scheme:
        endpoint = "https://" + f"{parsed_uri.netloc}"
        nosignrequest = True
        path = f"{parsed_uri.path}"
        if path.startswith("/"):
            path = path[1:]
        uri = "s3://" + path

    if not uri.startswith("/"):
        uri = format_s3_uri(uri, endpoint)
    if nosignrequest:
        if not uri.endswith("/"):
            uri = uri + "/"
        uri = uri + "?anonymous=true"
    setattr(extinfo, "lsid", rstring(uri))
    print("lsid:", uri)
    image.details.externalInfo = extinfo


def validate_uri(uri: str) -> str:
    """
    Check that the protocol is valid and the URI ends with "/"
    """
    parsed_uri = urlsplit(uri)
    scheme = f"{parsed_uri.scheme}"
    if "s3" not in scheme:
        raise Exception("Protocol should be s3. Protocol specified is: " + scheme)
    # Check if ends with / otherwise add one
    path = f"{parsed_uri.path}"
    if path.endswith("/"):
        return uri
    return uri + "/"


def validate_endpoint(endpoint: Optional[str]) -> None:
    """
    Check that the protocol is valid
    """
    if endpoint is None or not endpoint:
        return
    parsed_endpoint = urlsplit(endpoint)
    scheme = f"{parsed_endpoint.scheme}"
    if "https" not in scheme:
        raise Exception("Protocol should be https. Protocol specified is: " + scheme)


def link_to_target(
    conn: BlitzGateway, obj: Union[omero.model.PlateI, omero.model.ImageI], kwargs: dict
) -> None:
    is_plate = isinstance(obj, omero.model.PlateI)

    if kwargs.get("target"):
        if is_plate:
            screen_id = kwargs["target"]
            if screen_id.startswith("Screen:"):
                screen_id = screen_id.split(":")[1]
            target = conn.getObject("Screen", attributes={"id": int(screen_id)})
        else:
            dataset_id = kwargs["target"]
            if dataset_id.startswith("Dataset:"):
                dataset_id = dataset_id.split(":")[1]
            target = conn.getObject("Dataset", attributes={"id": int(dataset_id)})
    elif kwargs.get("target_by_name"):
        tname = kwargs["target_by_name"]
        if is_plate:
            target = conn.getObject("Screen", attributes={"name": tname})
        else:
            target = conn.getObject("Dataset", attributes={"name": tname})

    if target is None:
        print("Target not found")
        return

    if is_plate:
        link = omero.model.ScreenPlateLinkI()
        link.parent = omero.model.ScreenI(target.getId(), False)
        link.child = omero.model.PlateI(obj.getId(), False)
        conn.getUpdateService().saveObject(link)
        print("Linked to Screen", target.getId())
    else:
        link = omero.model.DatasetImageLinkI()
        link.parent = omero.model.DatasetI(target.getId(), False)
        link.child = omero.model.ImageI(obj.getId(), False)
        conn.getUpdateService().saveObject(link)
        print("Linked to Dataset", target.getId())


def register_zarr(conn: BlitzGateway, uri: str, **kwargs: Any) -> None:
    # All connection params are in kwargs so they can be easily
    # passed through to e.g. set_external_info
    kwargs = kwargs.copy()  # avoid modifying caller's dict
    kwargs["uri"] = uri

    endpoint = kwargs.get("endpoint")
    if endpoint is not None:
        endpoint = str(endpoint)
    nosignrequest = kwargs.get("nosignrequest", False)
    validate_endpoint(endpoint)
    store = None
    if uri.startswith("/"):
        store = zarr.storage.LocalStore(uri, read_only=True)
    else:
        storage_options: Dict[str, Any] = {}
        if nosignrequest:
            storage_options["anon"] = True

        if endpoint:
            storage_options["client_kwargs"] = {"endpoint_url": endpoint}

        # if FsspecStore is not None:
        #     store = FsspecStore.from_url(
        #         uri, read_only=True, storage_options=storage_options
        #     )
        # else:
        store = FSStore(uri, mode="r", **storage_options)

    zattrs = load_attrs(store)
    objs = []
    if "plate" in zattrs:
        print("Registering: Plate")
        objs = [register_plate(conn, store, zattrs, kwargs)]
    else:
        if zattrs.get("bioformats2raw.layout") == 3:
            print("Registering: bioformats2raw.layout")
            zarr_name = kwargs.get("uri", "").rstrip("/").split("/")[-1]
            if kwargs.get("name"):
                zarr_name = kwargs.get("name")
            # try to load OME/METADATA.ome.xml
            omexml_bytes = get_omexml_bytes(store)
            if omexml_bytes is not None:
                print("Importing OME/METADATA.ome.xml")
                rsp = full_import(conn.c, omexml_bytes, kwargs.get("wait", -1))
                for series, p in enumerate(rsp.pixels):
                    # set external info.
                    # NB: order of pixels MUST match the series 0, 1, 2...
                    image = conn.getObject("Image", p.image.id.val)
                    image_path = str(series)
                    image_attrs = load_attrs(store, image_path)
                    # pixels_type is only used if we have *incomplete* `omero` metadata
                    sizes, pixels_type = parse_image_metadata(
                        store, image_attrs, image_path
                    )
                    rnd_def = set_rendering_settings(
                        conn, image, image_attrs, pixels_type
                    )
                    if rnd_def is not None:
                        conn.getUpdateService().saveAndReturnObject(rnd_def)
                    set_external_info(image._obj, kwargs, image_path=image_path)
                    # default name is METADATA.ome.xml [series], based on clientPath?
                    new_name = image.name.replace("METADATA.ome.xml", zarr_name)
                    print("Imported Image:", image.id)
                    image.setName(new_name)
                    image.save()  # save Name and ExternalInfo
                    objs.append(image)
            else:
                print("OME/METADATA.ome.xml Not Found")
                series = 0
                series_exists = True
                while series_exists:
                    try:
                        print("Checking for series:", series)
                        obj = register_image(
                            conn, store, kwargs, None, image_path=str(series)
                        )
                        objs.append(obj)
                    except (ArrayNotFoundError, GroupNotFoundError):
                        # FIXME: FileNotFoundError (zarr v3) or
                        # zarr.errors.PathNotFoundError (zarr v2)
                        series_exists = False
                    series += 1
        else:
            print("Registering: Image")
            objs = [register_image(conn, store, kwargs, zattrs)]

    if kwargs.get("target") or kwargs.get("target_by_name"):
        for obj in objs:
            link_to_target(conn, obj, kwargs=kwargs)
