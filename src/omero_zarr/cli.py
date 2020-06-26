import sys
import os
import subprocess
from pathlib import Path
from functools import wraps

import omero
from omero.cli import BaseControl
from omero.cli import CLI
from omero.cli import ProxyStringType
from omero.gateway import BlitzGateway
from omero.rtypes import rlong
from omero.model import ImageI

from .raw_pixels import image_to_zarr
from .masks import image_masks_to_zarr, MASK_DTYPE_SIZE

HELP = "Export data in zarr format."
EXPORT_HELP = "Export an image in zarr format."
MASKS_HELP = "Export ROI Masks on the Image in zarr format."


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

        parser.add_argument(
            "--bf",
            action="store_true",
            help="Use bioformats2raw to read from managed repo",
        )
        parser.add_argument(
            "--tile_width", default=None, help="For use with bioformats2raw"
        )
        parser.add_argument(
            "--tile_height", default=None, help="For use with bioformats2raw"
        )
        parser.add_argument(
            "--resolutions", default=None, help="For use with bioformats2raw"
        )
        parser.add_argument(
            "--max_workers", default=None, help="For use with bioformats2raw"
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
            "--split-masks",
            action="store_true",
            help=(
                "Store each ROI in a separate group, required if masks "
                "overlap"
            ),
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

        export = parser.add(sub, self.export, EXPORT_HELP)
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
            image_id = args.object.id
            image = self._lookup(self.gateway, "Image", image_id)
            self.ctx.out("Export image: %s" % image.name)

            if args.bf:
                path, name = self._get_path(image_id)
                if path:
                    self._do_export(path, name, args)
                else:
                    print(
                        "Couldn't find managed repository path for this image."
                    )
            else:
                image_to_zarr(image, args)

    def _lookup(self, gateway, type, oid):
        """Find object of type by ID."""
        gateway.SERVICE_OPTS.setOmeroGroup("-1")
        obj = gateway.getObject(type, oid)
        if not obj:
            self.ctx.die(110, "No such %s: %s" % (type, oid))
        return obj

    def _do_export(self, path, name, args):
        abs_path = Path(os.environ["MANAGED_REPO"]) / path / name

        bf2raw = Path(os.environ["BF2RAW"])

        target = Path(args.output) / name
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

        print(options)
        process = subprocess.Popen(
            ["bin/bioformats2raw", options, abs_path, target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=bf2raw,
        )
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr)
        else:
            print("Image exported to {}".format(target))

    def _get_path(self, image_id):
        query = """
        select org from Image i left outer join i.fileset as fs left
        outer join fs.usedFiles as uf left outer join uf.originalFile as org
        where i.id = :iid
        """
        qs = self.client.sf.getQueryService()
        params = omero.sys.Parameters()
        params.map = {"iid": rlong(image_id)}
        results = qs.findAllByQuery(query, params)
        for res in results:
            name = res.name._val
            path = res.path._val
            if not (
                name.endswith(".log")
                or name.endswith(".txt")
                or name.endswith(".xml")
            ):
                return path, name


try:
    register("zarr", ZarrControl, HELP)
except NameError:
    if __name__ == "__main__":
        cli = CLI()
        cli.register("zarr", ZarrControl, HELP)
        cli.invoke(sys.argv[1:])
