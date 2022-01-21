#!/usr/bin/env python

#
# Copyright (C) 2022 University of Dundee. All Rights Reserved.
# Use is subject to license terms supplied in LICENSE.txt
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

"""
   Test of the omero zarr export plugin.
"""

from pathlib import Path

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from omero.gateway import BlitzGateway
from omero.testlib import ITest
from omero_zarr.raw_pixels import image_to_zarr


class TestZarrExport(ITest):
    def test_export(self, tmp_path: Path) -> None:
        dir_name = "test_export"
        kwargs = {"sizeX": 50, "sizeY": 25, "sizeZ": 2, "sizeC": 3, "sizeT": 4}
        export_dir = tmp_path / dir_name
        export_dir.mkdir()
        imgs = self.import_fake_file(**kwargs)

        assert imgs is not None
        image_id = imgs[0].id.val
        conn = BlitzGateway(client_obj=self.client)
        image = conn.getObject("Image", image_id)

        # Do the export...
        image_to_zarr(image, str(export_dir))

        new_dirs = [str(zdir) for zdir in export_dir.iterdir()]
        print("new_dirs", new_dirs)
        assert len(new_dirs) == 1
        zarr_path = new_dirs[0]
        # Expect a new IMAGE_ID.zarr directory
        assert zarr_path.endswith(f"{dir_name}/{image_id}.zarr")

        # try to read...
        reader = Reader(parse_url(zarr_path))
        nodes = list(reader())
        assert len(nodes) == 1

        print("data", nodes[0].data)
        shape = [kwargs.get("size" + dim) for dim in "TCZYX"]
        assert nodes[0].data[0].shape == tuple(shape)
