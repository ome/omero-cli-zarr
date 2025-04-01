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

import pytest
from cli import CLITest
from omero.gateway import BlitzGateway
from omero_zarr.cli import ZarrControl


class TestRender(CLITest):

    def setup_method(self, method: str) -> None:
        """Set up the test."""
        super().setup_method(method)
        self.cli.register("zarr", ZarrControl, "TEST")
        self.delete_args = self.args + ["delete"]
        self.args += ["zarr"]

    def create_image(self, sizec: int = 4, sizez: int = 1, sizet: int = 1) -> None:
        """Create a test image with the given dimensions."""
        self.gw = BlitzGateway(client_obj=self.client)

        images = self.import_fake_file(
            images_count=2, sizeZ=sizez, sizeT=sizet, sizeC=sizec, client=self.client
        )
        self.idonly = "%s" % images[0].id.val
        self.imageid = "Image:%s" % images[0].id.val
        self.source = "Image:%s" % images[1].id.val
        for image in images:
            img = self.gw.getObject("Image", image.id.val)
            img.getThumbnail(size=(96,), direct=False)

    # export tests
    # ========================================================================

    def test_export_zarr(self, capsys: pytest.CaptureFixture) -> None:
        """Test export of a Zarr image."""
        self.create_image(sizec=1)
        # Run test as self and as root
        self.cli.invoke(self.args + ["export", self.imageid], strict=True)
        out, err = capsys.readouterr()
        lines = out.split("\n")
        print(lines)
        assert "ok" in lines[0]
        assert "ok" in lines[1]
