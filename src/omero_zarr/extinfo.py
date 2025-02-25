from omero.gateway import BlitzGateway
from omero.gateway import BlitzObjectWrapper
from omero.model import ExternalInfoI, ImageI
from omero.rtypes import rstring, rlong
from omero_sys_ParametersI import ParametersI
from omero.model import Image
from omero.model import Plate
from omero.model import Screen
from omero.model import Dataset
from omero.model import Project


def get_path(conn: BlitzGateway, image_id: int) -> str:
    params = ParametersI()
    params.addId(image_id)
    query = """
        select fs from Fileset as fs
        join fetch fs.images as image
        left outer join fetch fs.usedFiles as usedFile
        join fetch usedFile.originalFile as f
        join fetch f.hasher where image.id = :id
    """
    fs = conn.getQueryService().findByQuery(query, params)
    res = fs._getUsedFiles()[0]._clientPath._val
    return res


def set_external_info(img: ImageI, path: str) -> ImageI:
    info = ExternalInfoI()
    info.entityType = rstring("com.glencoesoftware.ngff:multiscales")
    info.entityId = rlong(3)
    info.lsid = rstring(path)
    img.details.externalInfo = info
    return img


def _lookup(conn: BlitzGateway, type: str, oid: int) -> BlitzObjectWrapper:
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    obj = conn.getObject(type, oid)
    if not obj:
        raise ValueError(f"No such {type}: {oid}")
    return obj


def get_images(conn: BlitzGateway, object):
    if isinstance(object, list):
        for x in object:
            yield from get_images(conn, x)
    elif isinstance(object, Screen):
        scr = _lookup(conn, "Screen", object.id)
        for plate in scr.listChildren():
            yield from get_images(conn, plate._obj)
    elif isinstance(object, Plate):
        plt = _lookup(conn, "Plate", object.id)
        for well in plt.listChildren():
            for idx in range(0, well.countWellSample()):
                img = well.getImage(idx)
                yield img
    elif isinstance(object, Project):
        prj = _lookup(conn, "Project", object.id)
        for ds in prj.listChildren():
            yield from get_images(conn, ds._obj)
    elif isinstance(object, Dataset):
        ds = _lookup(conn, "Dataset", object.id)
        for img in ds.listChildren():
            yield img
    elif isinstance(object, Image):
        img = _lookup(conn, "Image", object.id)
        yield img
    else:
        raise ValueError(f"Unsupported type: {object.__class__.__name__}")
