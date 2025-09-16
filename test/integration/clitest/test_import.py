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

from random import random
from typing import Any, Dict

import pytest
from omero.gateway import BlitzGateway
from omero.testlib.cli import AbstractCLITest
from omero_zarr.cli import ZarrControl
from omero_zarr.zarr_import import import_zarr

SAMPLES: Dict[str, Dict[str, Any]] = {
    "6001240.zarr": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr",
        "dataset_name": "Test Import 6001240",
    },
    "13457227.zarr": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457227.zarr",
    },
    "13457227.zarr (s3)": {
        "url": "s3://idr/zarr/v0.4/idr0101A/13457227.zarr",
        "args": "--endpoint https://uk1s3.embassy.ebi.ac.uk/ --nosignrequest",
    },
    "CMU-1.ome.zarr": {
        "url": (
            "https://s3.us-east-1.amazonaws.com/"
            "gs-public-zarr-archive/CMU-1.ome.zarr/0"
        ),
    },
    # TODO: Haven't got this sample working with zarr v2 yet.
    # See https://github.com/BioNGFF/omero-import-utils/pull/24
    # "CMU-1.ome.zarr (s3)": {
    #     "url": "s3://gs-public-zarr-archiv§e/CMU-1.ome.zarr/0",
    #     "args": "--nosignrequest",
    #     "series_count": 1,
    # },
    "LacZ_ctrl.zarr": {
        "url": (
            "https://storage.googleapis.com/"
            "jax-public-ngff/example_v2/LacZ_ctrl.zarr"
        ),
        "pixel_sizes_x": [0.23, 0.45, 0.91],
        "series_count": 3,
        "dataset_name": "Test Import LacZ_ctrl.zarr",
    },
    "9846151.zarr": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0048A/9846151.zarr/",
    },
}


class TestImport(AbstractCLITest):

    def setup_method(self, method: str) -> None:
        """Set up the test."""
        self.args = self.login_args()
        self.cli.register("zarr", ZarrControl, "TEST")
        self.args += ["zarr"]

    # import tests
    # ========================================================================

    @pytest.mark.parametrize("sample_id", SAMPLES.keys())
    @pytest.mark.parametrize("invoke", ["api", "cli"])
    def test_register_images(
        self, capsys: pytest.CaptureFixture, sample_id: str, invoke: str
    ) -> None:
        """Test register of a Zarr image."""

        sample: Dict[str, Any] = SAMPLES[sample_id]
        conn = BlitzGateway(client_obj=self.client)

        # We test Dataset by Name or ID...
        ds_name = sample.get("dataset_name", "Test Register Images")
        ds_name += str(random())
        dataset = self.make_dataset(ds_name)

        if invoke == "cli":
            exp_args = [
                "import",
                sample["url"],
            ]
            if "args" in sample:
                url_args = sample["args"].split(" ")
                exp_args += url_args

            if "dataset_name" in sample:
                exp_args += ["--target-by-name", ds_name]
            else:
                # If no dataset name, we use the ID of the dataset we created
                exp_args += ["--target", str(dataset.getId().val)]

            self.cli.invoke(
                self.args + exp_args,
                strict=True,
            )
            out, err = capsys.readouterr()
            lines = out.split("\n")
            print(lines)
            image_ids = []
            for line in lines:
                if "Created Image" in line or "Imported Image" in line:
                    image_id = int(line.split(" ")[-1])
                    image_ids.append(image_id)

        else:
            # import via api
            kwargs = {}
            if "args" in sample:
                url_args = sample["args"].split(" ")
                if url_args[0] == "--endpoint":
                    kwargs["endpoint"] = url_args[1]
                if "--nosignrequest" in url_args:
                    kwargs["nosignrequest"] = True
            if "dataset_name" in sample:
                kwargs["target_by_name"] = ds_name
            else:
                # If no dataset name, we use the ID of the dataset we created
                kwargs["target"] = str(dataset.getId().val)

            objs = import_zarr(conn, sample["url"], **kwargs)
            image_ids = [obj.id.val for obj in objs]

        # check we have created the expected number of images
        assert len(image_ids) == sample.get("series_count", 1)
        for img_id in image_ids:
            image = self.query.get("Image", img_id)
            assert image is not None, f"Image {img_id} not found"

        # check images are linked to dataset
        dataset = conn.getObject("Dataset", dataset.id.val)
        assert dataset is not None, "Dataset not found"
        ds_imgs = [img.id for img in dataset.listChildren()]
        for img_id in image_ids:
            assert img_id in ds_imgs

        if "pixel_sizes_x" in sample:
            # check pixel sizes
            for i, img_id in enumerate(image_ids):
                image = conn.getObject("Image", img_id)
                pixels = image.getPrimaryPixels()
                size_x = pixels.getSizeX()
                phys_size_x = pixels.getPhysicalSizeX().getValue()
                exp_size_x = sample["pixel_sizes_x"][i]
                assert abs(phys_size_x - exp_size_x) < 0.01, (
                    f"Image {img_id} sizeX {size_x} physSizeX {phys_size_x} != "
                    f"expected {exp_size_x}"
                )
