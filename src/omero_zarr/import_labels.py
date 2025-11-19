#!/usr/bin/env python

# Copyright (C) 2025 University of Dundee & Open Microscopy Environment.
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

from typing import Optional

import numpy as np
from omero.gateway import BlitzGateway, ColorHolder
from omero.model import ImageI, MaskI, RoiI
from omero.rtypes import rdouble, rint, rstring
from zarr.creation import open_array
from zarr.errors import GroupNotFoundError
from zarr.hierarchy import open_group
from zarr.storage import Store, StoreLike


def load_attrs(store: StoreLike, path: Optional[str] = None) -> dict:
    """
    Load the attrs from the root group or path subgroup
    """
    root = open_group(store=store, mode="r", path=path)
    attrs = root.attrs.asdict()
    if "ome" in attrs:
        attrs = attrs["ome"]
    return attrs


def masks_from_labels_nd(
    labels_nd: np.ndarray, axes: list[str], label_props: dict
) -> dict:
    rois = {}

    colors_by_value = {}
    if "colors" in label_props:
        for color in label_props["colors"]:
            pixel_value = color.get("label-value", None)
            rgba = color.get("rgba", None)
            if pixel_value and rgba and len(rgba) == 4:
                colors_by_value[pixel_value] = rgba

    text_by_value = {}
    if "properties" in label_props:
        for props in label_props["properties"]:
            pixel_value = props.get("label-value", None)
            text = props.get("omero:text", None)
            if pixel_value and text:
                text_by_value[pixel_value] = text

    # For each label value, we create an ROI that
    # contains 2D masks for each time point, channel, and z-slice.
    for i in range(1, int(labels_nd.max()) + 1):
        if not np.any(labels_nd == i):
            continue

        masks = []
        bin_img = labels_nd == i

        sizes = {dim: labels_nd.shape[axes.index(dim)] for dim in axes}
        size_t = sizes.get("t", 1)
        size_c = sizes.get("c", 1)
        size_z = sizes.get("z", 1)

        for t in range(size_t):
            for c in range(size_c):
                for z in range(size_z):

                    indices = []
                    if "t" in axes:
                        indices.append(t)
                    if "c" in axes:
                        indices.append(c)
                    if "z" in axes:
                        indices.append(z)

                    # indices.append(np.s_[::])
                    # indices.append(np.s_[x:x_max:])

                    # slice down to 2D plane
                    plane = bin_img[tuple(indices)]

                    if not np.any(plane):
                        continue

                    # plane = plane.compute()

                    # Find bounding box to minimise size of mask
                    xmask = plane.sum(0).nonzero()[0]
                    ymask = plane.sum(1).nonzero()[0]
                    # if any(xmask) and any(ymask):
                    x0 = min(xmask)
                    w = max(xmask) - x0 + 1
                    y0 = min(ymask)
                    h = max(ymask) - y0 + 1
                    submask = plane[y0 : (y0 + h), x0 : (x0 + w)]

                    mask = MaskI()
                    mask.setBytes(np.packbits(np.asarray(submask, dtype=int)))
                    mask.setWidth(rdouble(w))
                    mask.setHeight(rdouble(h))
                    mask.setX(rdouble(x0))
                    mask.setY(rdouble(y0))

                    if i in colors_by_value:
                        ch = ColorHolder.fromRGBA(*colors_by_value[i])
                        mask.setFillColor(rint(ch.getInt()))
                    if "z" in axes:
                        mask.setTheZ(rint(z))
                    if "c" in axes:
                        mask.setTheC(rint(c))
                    if "t" in axes:
                        mask.setTheT(rint(t))
                    if i in text_by_value:
                        mask.setTextValue(rstring(text_by_value[i]))

                    masks.append(mask)

        rois[i] = masks

    return rois


def rois_from_labels_nd(
    conn: BlitzGateway,
    image_id: int,
    labels_nd: np.ndarray,
    axes: list[str],
    label_props: dict,
) -> None:
    # Text is set on Mask shapes, not ROIs
    rois = masks_from_labels_nd(labels_nd, axes, label_props)

    for label, masks in rois.items():
        if len(masks) > 0:
            create_roi(conn, image_id, shapes=masks)


def create_roi(conn: BlitzGateway, image_id: int, shapes: list) -> RoiI:
    # create an ROI, link it to Image
    roi = RoiI()
    roi.setImage(ImageI(image_id, False))
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    print(f"Save ROI for image: {image_id}")
    return conn.getUpdateService().saveAndReturnObject(roi)


def create_labels(
    conn: BlitzGateway, store: Store, image_id: int, image_path: Optional[str] = None
) -> None:
    """
    Create labels for the image
    """
    if image_path is None:
        image_path = ""
    labels_path = image_path.rstrip("/") + "/labels"
    try:
        labels_attrs = load_attrs(store, labels_path)
    except GroupNotFoundError:
        print("No zarr group at", labels_path)
        return
    if "labels" not in labels_attrs:
        print("No labels found at", labels_path)
        return
    for name in labels_attrs["labels"]:
        print("Found label:", name)
        label_path = f"{labels_path}/{name}"
        print("Loading label from:", label_path)

        label_image = load_attrs(store, label_path)

        axes = label_image["multiscales"][0]["axes"]
        axes_names = [axis["name"] for axis in axes]
        label_props = label_image.get("image-label", {})

        ds_path = label_image["multiscales"][0]["datasets"][0]["path"]
        array_path = f"{label_path}/{ds_path}/"
        labels_nd = open_array(store=store, mode="r", path=array_path)
        labels_data = labels_nd[slice(None)]

        # Create ROIs from the labels
        rois_from_labels_nd(conn, image_id, labels_data, axes_names, label_props)
