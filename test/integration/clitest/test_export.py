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

import json
from pathlib import Path

import pytest
from omero.model import RoiI
from omero.testlib.cli import AbstractCLITest
from omero_rois import mask_from_binary_image
from omero_zarr.cli import ZarrControl


class TestRender(AbstractCLITest):

    def setup_method(self, method: str) -> None:
        """Set up the test."""
        self.args = self.login_args()
        self.cli.register("zarr", ZarrControl, "TEST")
        self.args += ["zarr"]

    # export tests
    # ========================================================================

    def test_export_zarr(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Test export of a Zarr image."""
        sizec = 2
        images = self.import_fake_file(sizeC=sizec, client=self.client)
        img_id = images[0].id.val
        self.cli.invoke(
            self.args + ["export", f"Image:{img_id}", "--output", str(tmp_path)],
            strict=True,
        )
        out, err = capsys.readouterr()
        lines = out.split("\n")
        print(lines)
        all_lines = ", ".join(lines)
        assert "Exporting to" in all_lines
        assert "Finished" in all_lines

        assert len(list(tmp_path.iterdir())) == 1
        assert (tmp_path / f"{img_id}.zarr").is_dir()

        attrs_text = (tmp_path / f"{img_id}.zarr" / ".zattrs").read_text(
            encoding="utf-8"
        )
        attrs_json = json.loads(attrs_text)
        print(attrs_json)
        assert "multiscales" in attrs_json
        assert len(attrs_json["omero"]["channels"]) == sizec
        assert attrs_json["omero"]["channels"][0]["window"]["min"] == 0
        assert attrs_json["omero"]["channels"][0]["window"]["max"] == 255

        arr_text = (tmp_path / f"{img_id}.zarr" / "0" / ".zarray").read_text(
            encoding="utf-8"
        )
        arr_json = json.loads(arr_text)
        assert arr_json["shape"] == [sizec, 512, 512]

    def test_export_plate(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:

        plates = self.import_plates(
            client=self.client,
            plates=1,
            plate_acqs=1,
            plate_cols=2,
            plate_rows=2,
            fields=1,
        )
        plate_id = plates[0].id.val
        self.cli.invoke(
            self.args + ["export", f"Plate:{plate_id}", "--output", str(tmp_path)],
            strict=True,
        )
        out, err = capsys.readouterr()
        lines = out.split("\n")
        print(lines)
        all_lines = ", ".join(lines)
        assert "Exporting to" in all_lines
        assert "Finished" in all_lines
        assert (tmp_path / f"{plate_id}.zarr").is_dir()
        attrs_text = (tmp_path / f"{plate_id}.zarr" / ".zattrs").read_text(
            encoding="utf-8"
        )
        attrs_json = json.loads(attrs_text)
        print(attrs_json)
        assert len(attrs_json["plate"]["wells"]) == 4
        assert attrs_json["plate"]["rows"] == [{"name": "A"}, {"name": "B"}]
        assert attrs_json["plate"]["columns"] == [{"name": "1"}, {"name": "2"}]

        arr_text = (
            tmp_path / f"{plate_id}.zarr" / "A" / "1" / "0" / "0" / ".zarray"
        ).read_text(encoding="utf-8")
        arr_json = json.loads(arr_text)
        assert arr_json["shape"] == [512, 512]

    def test_export_masks(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Test export of a Zarr image."""
        images = self.import_fake_file(sizeC=2, client=self.client)
        img_id = images[0].id.val
        size_xy = 512

        # Create a mask
        from skimage.data import binary_blobs

        blobs = binary_blobs(length=size_xy, volume_fraction=0.1, n_dim=2).astype(
            "int8"
        )
        red = [255, 0, 0, 255]
        mask = mask_from_binary_image(blobs, rgba=red, z=0, c=0, t=0)

        roi = RoiI()
        roi.setImage(images[0])
        roi.addShape(mask)
        updateService = self.client.sf.getUpdateService()
        updateService.saveAndReturnObject(roi)

        print("tmp_path", tmp_path)

        img_args = [f"Image:{img_id}", "--output", str(tmp_path)]
        self.cli.invoke(
            self.args + ["export"] + img_args,
            strict=True,
        )

        self.cli.invoke(
            self.args + ["masks"] + img_args,
            strict=True,
        )

        out, err = capsys.readouterr()
        lines = out.split("\n")
        print(lines)
        all_lines = ", ".join(lines)
        assert "Exporting to" in all_lines
        assert "Finished" in all_lines
        assert "Found 1 mask shapes in 1 ROIs" in all_lines

        labels_text = (
            tmp_path / f"{img_id}.zarr" / "labels" / "0" / ".zattrs"
        ).read_text(encoding="utf-8")
        labels_json = json.loads(labels_text)
        assert labels_json["image-label"]["colors"] == [{"label-value": 1, "rgba": red}]

        arr_text = (
            tmp_path / f"{img_id}.zarr" / "labels" / "0" / "0" / ".zarray"
        ).read_text(encoding="utf-8")
        arr_json = json.loads(arr_text)
        assert arr_json["shape"] == [1, 512, 512]
