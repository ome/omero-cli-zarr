import re
from typing import Any, Generator

from omero.gateway import BlitzGateway, BlitzObjectWrapper, ImageWrapper
from omero.model import Dataset, ExternalInfoI, Image, ImageI, Plate, Project, Screen
from omero.rtypes import rlong, rstring
from omero_sys_ParametersI import ParametersI

# Regex to match well positions (eg. A1)
WELL_POS_RE = re.compile(r"(?P<row>\D+)(?P<col>\d+)")


def get_extinfo(conn: BlitzGateway, image: ImageWrapper) -> ExternalInfoI:
    """
    Get the external info for an OMERO image.

    Args:
        conn (BlitzGateway): Active OMERO gateway connection
        image (ImageWrapper): OMERO image

    Returns:
        ExternalInfoI: External info object

    Raises:
        Exception: If the query fails.
    """

    details = image.getDetails()
    if details and details._externalInfo:
        params = ParametersI()
        params.addId(details._externalInfo._id)
        query = """
            select e from ExternalInfo as e
            where e.id = :id
        """
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        extinfo = conn.getQueryService().findByQuery(query, params)
        return extinfo
    return None


def _get_path(conn: BlitzGateway, image_id: int) -> str:
    """
    Retrieve the (first) original file path for a given OMERO image.

    Args:
        conn (BlitzGateway): Active OMERO gateway connection
        image_id (int): OMERO image id

    Returns:
        str: path of the image file

    Raises:
        Exception: If the query fails.
    """
    params = ParametersI()
    params.addId(image_id)
    query = """
        select fs from Fileset as fs
        join fetch fs.images as image
        left outer join fetch fs.usedFiles as usedFile
        join fetch usedFile.originalFile as f
        join fetch f.hasher
        where image.id = :id
    """
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    fs = conn.getQueryService().findByQuery(query, params)
    path = fs._getUsedFiles()[0]._clientPath._val
    return path


def _lookup(conn: BlitzGateway, _type: str, oid: int) -> BlitzObjectWrapper:
    """
    Look up an OMERO object by its type and ID.

    Args:
        conn (BlitzGateway): Active OMERO gateway connection
        _type (str): Type of OMERO object (e.g., "Screen", "Plate", "Image")
        oid (int): Object ID to look up

    Returns:
        BlitzObjectWrapper: Wrapped OMERO object

    Raises:
        ValueError: If the object doesn't exist
    """
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    obj = conn.getObject(_type, oid)
    if not obj:
        raise ValueError(f"No such {_type}: {oid}")
    return obj


def get_images(conn: BlitzGateway, obj: Any) -> Generator[ImageWrapper, str, int]:
    """
    Generator that yields images from any OMERO container object.

    Recursively traverses OMERO container hierarchies
    (Screen/Plate/Project/Dataset) to find all contained images.

    | Args:
    |    conn (BlitzGateway): Active OMERO gateway connection
    |    obj: OMERO container object (Screen, Plate, Project, Dataset, Image)
    |        or a list of such objects

    Yields:
        tuple: Contains:
            - ImageWrapper: Image object
            - str | None: Well position (eg. A1) if from plate, None otherwise
            - int | None: Well sample index if from plate, None otherwise

    Raises:
        ValueError: If given an unsupported object type
    """
    if isinstance(obj, list):
        for x in obj:
            yield from get_images(conn, x)
    elif isinstance(obj, Screen):
        scr = _lookup(conn, "Screen", obj.id)
        for plate in scr.listChildren():
            yield from get_images(conn, plate._obj)
    elif isinstance(obj, Plate):
        plt = _lookup(conn, "Plate", obj.id)
        for well in plt.listChildren():
            for idx in range(0, well.countWellSample()):
                img = well.getImage(idx)
                yield img, well.getWellPos(), idx
    elif isinstance(obj, Project):
        prj = _lookup(conn, "Project", obj.id)
        for ds in prj.listChildren():
            yield from get_images(conn, ds._obj)
    elif isinstance(obj, Dataset):
        ds = _lookup(conn, "Dataset", obj.id)
        for img in ds.listChildren():
            yield img, None, None
    elif isinstance(obj, Image):
        img = _lookup(conn, "Image", obj.id)
        yield img, None, None
    else:
        raise ValueError(f"Unsupported type: {obj.__class__.__name__}")

    return 0


def set_external_info(
    conn: BlitzGateway,
    img: ImageI,
    well: str,
    idx: int,
    lsid: str,
    entityType: str,
    entityId: int,
) -> ImageI:
    """
    Set the external info for an OMERO image.

    | Args:
    |    conn (BlitzGateway): Active OMERO gateway connection
    |    img (ImageI): OMERO image
    |    well (str | None): Optional well position (e.g. 'A1')
    |    idx (int | None): Optional well sample field index
    |    lsid (str | None): Optional custom LSID path. If None, path is
    |        derived from image's clientpath
    |    entityType (str | None): Optional entity type. Defaults to
    |        'com.glencoesoftware.ngff:multiscales'
    |    entityId (int | None): Optional entity ID. Defaults to 3

    Returns:
        ImageI: Updated OMERO image with external info set

    Raises:
        ValueError: If the path cannot be determined from clientpath and no
            lsid is provided, or if the well position format is invalid

    """
    if not entityType:
        entityType = "com.glencoesoftware.ngff:multiscales"
    if not entityId:
        entityId = 3
    if lsid:
        path = lsid
    else:
        path = _get_path(conn, img.id)
        if path.endswith("OME/METADATA.ome.xml"):
            if well:
                match = WELL_POS_RE.match(well)
                if match:
                    col = match.group("col")
                    row = match.group("row")
                    path = path.replace("OME/METADATA.ome.xml", f"{row}")
                    path = f"/{path}/{col}/{idx}"
                else:
                    raise ValueError(f"Couldn't parse well position: {well}")
            else:
                series = img.getSeries()._val
                path = path.replace("OME/METADATA.ome.xml", f"{series}")
                path = f"/{path}"
        else:
            raise ValueError(f"Doesn't seem to be an ome.zarr: {path}")

    info = ExternalInfoI()
    info.entityType = rstring(entityType)
    info.entityId = rlong(entityId)
    info.lsid = rstring(path)
    img.details.externalInfo = info
    return img


def _checkNone(obj: Any) -> str:
    """
    Helper function to safely get string value from OMERO rtype objects.

    Args:
        obj: OMERO rtype object that may have a _val attribute

    Returns:
        str: The value of obj._val if it exists, otherwise "None"
    """
    if obj and obj._val:
        return obj._val
    return "None"


def external_info_str(extinfo: ExternalInfoI) -> str:
    """
    Format ExternalInfo object as a human-readable string.

    Args:
        extinfo (ExternalInfoI): OMERO ExternalInfo object

    Returns:
        str: Formatted string containing entityType, entityId and lsid,
            or "None" if extinfo is None
    """
    if extinfo is not None:
        return (
            f"[entityType={_checkNone(extinfo.entityType)}\n"
            f" entityId={_checkNone(extinfo.entityId)}\n"
            f" lsid={_checkNone(extinfo.lsid)}]"
        )
    return "None"
