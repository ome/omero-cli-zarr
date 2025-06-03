#!/usr/bin/env python

# Copyright (C) 2023 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List

from omero.cli import CLI, BaseControl, Parser, ProxyStringType
from omero.gateway import BlitzGateway, BlitzObjectWrapper
from omero.model import ImageI, PlateI
from zarr.hierarchy import open_group
from zarr.storage import FSStore

from .masks import (
    MASK_DTYPE_SIZE,
    MaskSaver,
    image_shapes_to_zarr,
    plate_shapes_to_zarr,
)
from .raw_pixels import (
    add_omero_metadata,
    add_toplevel_metadata,
    image_to_zarr,
    plate_to_zarr,
)

HELP = """Export data in zarr format.

Subcommands
===========

 - export
 - masks

"""
EXPORT_HELP = """Export an image in zarr format.

Using bioformats2raw
--------------------

Rather than download pixels from OMERO using the API, two options allow
converting files directly using bioformats2raw:

  --bf

     Server-side option which queries the server for the location
     of the file within the managed repository (e.g. /OMERO/ManagedRepository)
     and then calls bioformats2raw directly on that file.

  --bfpath=local/file/path

     If a copy of the fileset is available locally, then this option
     calls bioformats2raw directly on the argument. No checks are performed
     that the fileset is the correct one.

bioformats2raw executable
-------------------------

In order to use bioformats2raw for the actual export,
make sure the bioformats2raw binary is in the $PATH.

bioformats2raw options
----------------------

  --max_workers

     Maximum number of workers to use for parallel generation of pyramids

  --tile_width / --tile_height

     Maximum tile width or height to read

  --resolutions

     Number of pyramid resolutions to generate

"""

MASKS_HELP = """Export ROI Masks on the Image in zarr format.

Options
-------

  --style

     'labeled': 5D integer values (default but overlaps are not supported!)
     'split': one group per ROI

"""

POLYGONS_HELP = """Export ROI Polygons on the Image or Plate in zarr format"""


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

        # Subcommands
        sub = parser.sub()
        polygons = parser.add(sub, self.polygons, POLYGONS_HELP)
        polygons.add_argument(
            "object",
            type=ProxyStringType("Image"),
            help="The Image from which to export Polygons.",
        )
        polygons.add_argument(
            "--label-bits",
            default=str(max(MASK_DTYPE_SIZE.keys())),
            choices=[str(s) for s in sorted(MASK_DTYPE_SIZE.keys())],
            help=(
                "Integer bit size for each label pixel, use 1 for a binary "
                "label, default %(default)s"
            ),
        )
        polygons.add_argument(
            "--style",
            choices=("split", "labeled"),
            default="labeled",
            help=("Choice of storage for ROIs [breaks ome-zarr]"),
        )
        polygons.add_argument(
            "--label-path",
            help=(
                "Subdirectory of the image location for storing labels. "
                "[breaks ome-zarr]"
            ),
            default="labels",
        )
        polygons.add_argument(
            "--source-image",
            help=(
                "Path to the multiscales group containing the source image/plate. "
                "By default, use the output directory"
            ),
            default=None,
        )
        polygons.add_argument(
            "--label-map",
            help=(
                "File in format: ID,NAME,ROI_ID which is used to separate "
                "overlapping labels"
            ),
        )
        polygons.add_argument(
            "--label-name",
            help=("Name of the array that will be stored. Ignored for --style=split"),
            default="0",
        )
        polygons.add_argument(
            "--name_by",
            default="id",
            choices=["id", "name"],
            help=(
                "How the existing Image or Plate zarr is named. Default 'id' is "
                "[ID].ome.zarr. 'name' is [NAME].ome.zarr"
            ),
        )

        masks = parser.add(sub, self.masks, MASKS_HELP)
        masks.add_argument(
            "object",
            type=ProxyStringType("Image"),
            help="The Image from which to export Masks.",
        )
        masks.add_argument(
            "--source-image",
            help=(
                "Path to the multiscales group containing the source image/plate. "
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
        masks.add_argument(
            "--name_by",
            default="id",
            choices=["id", "name"],
            help=(
                "How the existing Image or Plate zarr is named. Default 'id' is "
                "[ID].ome.zarr. 'name' is [NAME].ome.zarr"
            ),
        )

        export = parser.add(sub, self.export, EXPORT_HELP)
        export.add_argument(
            "--bf",
            action="store_true",
            help="Use bioformats2raw on the server to export images. Requires"
            " bioformats2raw 0.3.0 or higher and access to the managed repo.",
        )
        export.add_argument(
            "--bfpath",
            help="Use bioformats2raw on a local copy of a file. Requires"
            " bioformats2raw 0.4.0 or higher.",
        )
        export.add_argument(
            "--tile_width",
            default=None,
            help="Maximum tile width",
        )
        export.add_argument(
            "--tile_height",
            default=None,
            help="Maximum tile height",
        )
        export.add_argument(
            "--resolutions",
            default=None,
            help="Number of pyramid resolutions to generate"
            " (only for use with bioformats2raw)",
        )
        export.add_argument(
            "--max_workers",
            default=None,
            help="Maximum number of workers (only for use with bioformats2raw)",
        )
        export.add_argument(
            "--name_by",
            default="id",
            choices=["id", "name"],
            help=(
                "How to name the Image or Plate zarr. Default 'id' is [ID].ome.zarr. "
                "'name' is [NAME].ome.zarr"
            ),
        )
        export.add_argument(
            "object",
            type=ProxyStringType("Image"),
            help="The Image to export.",
        )

        for subcommand in (polygons, masks, export):
            subcommand.add_argument(
                "--output", type=str, default="", help="The output directory"
            )
        for subcommand in (polygons, masks):
            subcommand.add_argument(
                "--overlaps",
                type=str,
                default=MaskSaver.OVERLAPS[0],
                choices=MaskSaver.OVERLAPS,
                help="To allow overlapping shapes, use 'dtype_max':"
                " All overlapping regions will be set to the"
                " max value for the dtype",
            )

    @gateway_required
    def masks(self, args: argparse.Namespace) -> None:
        """Export masks on the Image as zarr files."""
        if isinstance(args.object, ImageI):
            image_id = args.object.id
            image = self._lookup(self.gateway, "Image", image_id)
            self.ctx.out("Export Masks on Image: %s" % image.name)
            image_shapes_to_zarr(image, ["Mask"], args)
        elif isinstance(args.object, PlateI):
            plate = self._lookup(self.gateway, "Plate", args.object.id)
            plate_shapes_to_zarr(plate, ["Mask"], args)

    @gateway_required
    def polygons(self, args: argparse.Namespace) -> None:
        """Export polygons on the Plate or Image as zarr files."""
        if isinstance(args.object, ImageI):
            image_id = args.object.id
            image = self._lookup(self.gateway, "Image", image_id)
            self.ctx.out("Export Polygons on Image: %s" % image.name)
            image_shapes_to_zarr(image, ["Polygon"], args)
        elif isinstance(args.object, PlateI):
            plate = self._lookup(self.gateway, "Plate", args.object.id)
            plate_shapes_to_zarr(plate, ["Polygon"], args)

    @gateway_required
    def export(self, args: argparse.Namespace) -> None:
        if isinstance(args.object, ImageI):
            image = self._lookup(self.gateway, "Image", args.object.id)
            if args.bf or args.bfpath:
                self._bf_export(image, args)
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

    def _bf_export(self, image: BlitzObjectWrapper, args: argparse.Namespace) -> None:
        if args.bfpath:
            abs_path = Path(args.bfpath)
        elif image.getInplaceImport():
            p = image.getImportedImageFilePaths()["client_paths"][0]
            abs_path = Path("/") / Path(p)
        else:
            if self.client is None:
                raise Exception("This cannot happen")  # mypy is confused
            prx, desc = self.client.getManagedRepository(description=True)
            p = image.getImportedImageFilePaths()["server_paths"][0]
            abs_path = Path(desc._path._val) / Path(desc._name._val) / Path(p)
        temp_target = (Path(args.output) or Path.cwd()) / f"{image.id}.tmp"
        image_target = (Path(args.output) or Path.cwd()) / f"{image.id}.zarr"

        if temp_target.exists():
            self.ctx.die(111, f"{temp_target.resolve()} already exists")
        if image_target.exists():
            self.ctx.die(111, f"{image_target.resolve()} already exists")

        cmd: List[str] = [
            "bioformats2raw",
            str(abs_path.resolve()),
            str(temp_target.resolve()),
        ]

        if args.tile_width:
            cmd.append(f"--tile_width={args.tile_width}")
        if args.tile_height:
            cmd.append(f"--tile_height={args.tile_height}")
        if args.resolutions:
            cmd.append(f"--resolutions={args.resolutions}")
        if args.max_workers:
            cmd.append(f"--max_workers={args.max_workers}")
        cmd.append(f"--series={image.series}")
        cmd.append("--no-root-group")
        cmd.append("--no-ome-meta-export")

        self.ctx.dbg(" ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if stderr:
            self.ctx.err(stderr.decode("utf-8"))
        if process.returncode == 0:
            image_source = temp_target / "0"
            image_source.rename(image_target)
            temp_target.rmdir()
            self.ctx.out(f"Image exported to {image_target.resolve()}")

        # Add OMERO metadata
        store = FSStore(
            str(image_target.resolve()),
            auto_mkdir=False,
            normalize_keys=False,
            mode="w",
        )
        root = open_group(store)
        add_omero_metadata(root, image)
        add_toplevel_metadata(root)


try:
    register("zarr", ZarrControl, HELP)  # type: ignore
except NameError:
    if __name__ == "__main__":
        cli = CLI()
        cli.register("zarr", ZarrControl, HELP)
        cli.invoke(sys.argv[1:])
