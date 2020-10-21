from omero_zarr.cli import HELP, ZarrControl

register("zarr", ZarrControl, HELP)  # type: ignore # noqa
