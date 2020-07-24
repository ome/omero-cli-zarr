import sys
import subprocess
from pathlib import Path
from functools import wraps

from omero.cli import BaseControl
from omero.cli import CLI
from omero.cli import ProxyStringType
from omero.gateway import BlitzGateway
from omero.model import ImageI

from .raw_pixels import image_to_zarr
from .masks import image_masks_to_zarr, MASK_DTYPE_SIZE

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

     'labelled': 5D integer values (default but overlaps are not supported!)
     '6d': masks are stored in a 6D array
     'split': one group per ROI

"""


def gateway_required(func):
    """
  Decorator which initializes a client (self.client),
  a BlitzGateway (self.gateway), and makes sure that
  all services of the Blitzgateway are closed again.
  """

    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        self.client = self.ctx.conn(*args)
        self.gateway = BlitzGateway(client_obj=self.client)

        try:
            return func(self, *args, **kwargs)
        finally:
            if self.gateway is not None:
                self.gateway.close(hard=False)
                self.gateway = None
                self.client = None

    return _wrapper


class ZarrControl(BaseControl):

    gateway = None
    client = None

    def _configure(self, parser):
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
            "--mask-path",
            help=(
                "Subdirectory of the image location for storing masks. "
                "[breaks ome-zarr]"
            ),
            default="masks",
        )
        masks.add_argument(
            "--mask-name",
            help=(
                "Name of the array that will be stored. "
                "Ignored for --style=split"
            ),
            default="0",
        )
        masks.add_argument(
            "--style",
            choices=("6d", "split", "labelled"),
            default="labelled",
            help=("Choice of storage for ROIs [breaks ome-zarr]"),
        )
        masks.add_argument(
            "--mask-bits",
            default=str(max(MASK_DTYPE_SIZE.keys())),
            choices=[str(s) for s in sorted(MASK_DTYPE_SIZE.keys())],
            help=(
                "Integer bit size for each mask pixel, use 1 for a binary "
                "mask, default %(default)s"
            ),
        )
        masks.add_argument(
            "--mask-map",
            help=(
                "File in format: ID,NAME,ROI_ID which is used to separate "
                "overlapping masks"
            ),
        )

        export = parser.add(sub, self.export, EXPORT_HELP)
        export.add_argument(
            "--bf",
            action="store_true",
            help="Use bioformats2raw to export the image.",
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
            "object",
            type=ProxyStringType("Image"),
            help="The Image to export.",
        )

    @gateway_required
    def masks(self, args):
        """Export masks on the Image as zarr files."""
        if isinstance(args.object, ImageI):
            image_id = args.object.id
            image = self._lookup(self.gateway, "Image", image_id)
            self.ctx.out("Export Masks on Image: %s" % image.name)
            image_masks_to_zarr(image, args)

    @gateway_required
    def export(self, args):
        if isinstance(args.object, ImageI):
            image = self._lookup(self.gateway, "Image", args.object.id)
            inplace = image.getInplaceImport()

            if args.bf:
                prx, desc = self.client.getManagedRepository(description=True)
                repo_path = Path(desc._path._val) / Path(desc._name._val)
                if inplace:
                    for path in image.getImportedImageFilePaths()['client_paths']:
                        self._bf_export(Path('/') / Path(path), args)
                else:
                    for path in image.getImportedImageFilePaths()['server_paths']:
                        self._bf_export(repo_path / path, args)
            else:
                image_to_zarr(image, args)

    def _lookup(self, gateway, type, oid):
        """Find object of type by ID."""
        gateway.SERVICE_OPTS.setOmeroGroup("-1")
        obj = gateway.getObject(type, oid)
        if not obj:
            self.ctx.die(110, "No such %s: %s" % (type, oid))
        return obj

    def _bf_export(self, abs_path, args):
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

        self.ctx.dbg("bioformats2raw %s %s %s" % (options, abs_path.resolve(),
                                                  target.resolve()))
        process = subprocess.Popen(
            ["bioformats2raw", options, abs_path.resolve(),
             target.resolve()], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stderr:
            self.ctx.err(stderr)
        else:
            self.ctx.out("Image exported to {}".format(target.resolve()))


try:
    register("zarr", ZarrControl, HELP)
except NameError:
    if __name__ == "__main__":
        cli = CLI()
        cli.register("zarr", ZarrControl, HELP)
        cli.invoke(sys.argv[1:])
