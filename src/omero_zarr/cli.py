import argparse
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from omero.cli import CLI, BaseControl, Parser, ProxyStringType
from omero.gateway import BlitzGateway, BlitzObjectWrapper
from omero.model import ImageI, PlateI

from .masks import MASK_DTYPE_SIZE, image_masks_to_zarr
from .raw_pixels import image_to_zarr, plate_to_zarr

HELP = """Export data in zarr format.

Subcommands
===========

 - export
 - masks

"""
EXPORT_HELP = """Export an image in zarr format.

In order to use bioformats2raw for the actual export,
make sure the bioformats2raw binary is in the $PATH.
"""

MASKS_HELP = """Export ROI Masks on the Image in zarr format.

Options
-------

  --style

     'labeled': 5D integer values (default but overlaps are not supported!)
     'split': one group per ROI

"""


def gateway_required(func: Callable) -> Callable:
    """
  Decorator which initializes a client (self.client),
  a BlitzGateway (self.gateway), and makes sure that
  all services of the Blitzgateway are closed again.
  """

    @wraps(func)
    def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Callable:
        self.client = self.ctx.conn(*args)
        self.gateway = BlitzGateway(client_obj=self.client)

        try:
            return func(self, *args, **kwargs)
        finally:
            if self.gateway is not None:
                self.gateway.close(hard=False)
                self.gateway = None
                self.client = None  # type: ignore

    return _wrapper


class ZarrControl(BaseControl):

    gateway = None
    client = None

    def _configure(self, parser: Parser) -> None:
        parser.add_login_arguments()

        parser.add_argument(
            "--output", type=str, default="", help="The output directory"
        )

        parser.add_argument(
            "--cache_numpy",
            action="store_true",
            help="Save planes as .npy files in case of connection loss",
        )

        # Subcommands
        sub = parser.sub()
        masks = parser.add(sub, self.masks, MASKS_HELP)
        masks.add_argument(
            "object",
            type=ProxyStringType("Image"),
            help="The Image from which to export Masks.",
        )
        masks.add_argument(
            "--source-image",
            help=(
                "Path to the multiscales group containing the source image. "
                "By default, use the output directory"
            ),
            default=None,
        )
        masks.add_argument(
            "--label-path",
            help=(
                "Subdirectory of the image location for storing labels. "
                "[breaks ome-zarr]"
            ),
            default="labels",
        )
        masks.add_argument(
            "--label-name",
            help=("Name of the array that will be stored. Ignored for --style=split"),
            default="0",
        )
        masks.add_argument(
            "--style",
            choices=("split", "labeled"),
            default="labeled",
            help=("Choice of storage for ROIs [breaks ome-zarr]"),
        )
        masks.add_argument(
            "--label-bits",
            default=str(max(MASK_DTYPE_SIZE.keys())),
            choices=[str(s) for s in sorted(MASK_DTYPE_SIZE.keys())],
            help=(
                "Integer bit size for each label pixel, use 1 for a binary "
                "label, default %(default)s"
            ),
        )
        masks.add_argument(
            "--label-map",
            help=(
                "File in format: ID,NAME,ROI_ID which is used to separate "
                "overlapping labels"
            ),
        )

        export = parser.add(sub, self.export, EXPORT_HELP)
        export.add_argument(
            "--bf", action="store_true", help="Use bioformats2raw to export the image.",
        )
        export.add_argument(
            "--tile_width", default=None, help="For use with bioformats2raw"
        )
        export.add_argument(
            "--tile_height", default=None, help="For use with bioformats2raw"
        )
        export.add_argument(
            "--resolutions", default=None, help="For use with bioformats2raw"
        )
        export.add_argument(
            "--max_workers", default=None, help="For use with bioformats2raw"
        )
        export.add_argument(
            "object", type=ProxyStringType("Image"), help="The Image to export.",
        )

    @gateway_required
    def masks(self, args: argparse.Namespace) -> None:
        """Export masks on the Image as zarr files."""
        if isinstance(args.object, ImageI):
            image_id = args.object.id
            image = self._lookup(self.gateway, "Image", image_id)
            self.ctx.out("Export Masks on Image: %s" % image.name)
            image_masks_to_zarr(image, args)

    @gateway_required
    def export(self, args: argparse.Namespace) -> None:
        if isinstance(args.object, ImageI):
            image = self._lookup(self.gateway, "Image", args.object.id)
            inplace = image.getInplaceImport()

            if args.bf:
                if self.client is None:
                    raise Exception("This cannot happen")  # mypy is confused
                prx, desc = self.client.getManagedRepository(description=True)
                repo_path = Path(desc._path._val) / Path(desc._name._val)
                if inplace:
                    for p in image.getImportedImageFilePaths()["client_paths"]:
                        self._bf_export(Path("/") / Path(p), args)
                else:
                    for p in image.getImportedImageFilePaths()["server_paths"]:
                        self._bf_export(repo_path / p, args)
            else:
                image_to_zarr(image, args)
        elif isinstance(args.object, PlateI):
            plate = self._lookup(self.gateway, "Plate", args.object.id)
            plate_to_zarr(plate, args)

    def _lookup(
        self, gateway: BlitzGateway, otype: str, oid: int
    ) -> BlitzObjectWrapper:
        """Find object of type by ID."""
        gateway.SERVICE_OPTS.setOmeroGroup("-1")
        obj = gateway.getObject(otype, oid)
        if not obj:
            self.ctx.die(110, f"No such {otype}: {oid}")
        return obj

    def _bf_export(self, abs_path: Path, args: argparse.Namespace) -> None:
        target = (Path(args.output) or Path.cwd()) / Path(abs_path).name
        target.mkdir(exist_ok=True)

        options = "--file_type=zarr"
        if args.tile_width:
            options += " --tile_width=" + args.tile_width
        if args.tile_height:
            options += " --tile_height=" + args.tile_height
        if args.resolutions:
            options += " --resolutions=" + args.resolutions
        if args.max_workers:
            options += " --max_workers=" + args.max_workers

        self.ctx.dbg(
            f"bioformats2raw {options} {abs_path.resolve()} {target.resolve()}"
        )
        process = subprocess.Popen(
            ["bioformats2raw", options, abs_path.resolve(), target.resolve()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if stderr:
            self.ctx.err(stderr)
        else:
            self.ctx.out(f"Image exported to {target.resolve()}")


try:
    register("zarr", ZarrControl, HELP)  # type: ignore
except NameError:
    if __name__ == "__main__":
        cli = CLI()
        cli.register("zarr", ZarrControl, HELP)
        cli.invoke(sys.argv[1:])
