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
import logging
import os
import re
import time
from collections import defaultdict
from fileinput import input as finput
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import omero.clients  # noqa
from ome_zarr.conversions import int_to_rgba_255
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Node
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import write_multiscale_labels
from omero.model import MaskI, PolygonI
from omero.rtypes import unwrap
from skimage.draw import polygon as sk_polygon
from zarr.hierarchy import open_group

from .util import (
    get_zarr_name,
    marshal_axes,
    marshal_transformations,
    open_store,
    print_status,
)

LOGGER = logging.getLogger("omero_zarr.masks")

# Mapping of dimension names to axes in the Zarr
DIMENSION_ORDER: Dict[str, int] = {
    "T": 0,
    "C": 1,
    "Z": 2,
    "Y": 3,
    "X": 4,
}

MASK_DTYPE_SIZE: Dict[int, np.dtype] = {
    1: bool,
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
}

OME_MODEL_POINT_LIST_RE = re.compile(r"([\d.]+),([\d.]+)")

SHAPE_TYPES = {"Mask": MaskI, "Polygon": PolygonI}


def plate_shapes_to_zarr(
    plate: omero.gateway.PlateWrapper, shape_types: List[str], args: argparse.Namespace
) -> None:
    """
    Export shapes of type "Mask" or "Polygon" on a Plate to OME-Zarr labels

    @param shape_types      e.g. ["Mask", "Polygon"]
    """

    gs = plate.getGridSize()
    n_rows = gs["rows"]
    n_cols = gs["columns"]
    n_fields = plate.getNumberOfFields()
    total = n_rows * n_cols * (n_fields[1] - n_fields[0] + 1)

    dtype = MASK_DTYPE_SIZE[int(args.label_bits)]
    saver = MaskSaver(
        plate,
        None,
        dtype,
        args.label_path,
        args.style,
        args.source_image,
        args.overlaps,
        args.output,
        args.name_by,
    )

    count = 0
    t0 = time.time()
    for well in plate.listChildren():
        row = plate.getRowLabels()[well.row]
        col = plate.getColumnLabels()[well.column]
        for field in range(n_fields[0], n_fields[1] + 1):
            ws = well.getWellSample(field)
            field_name = "%d" % field
            count += 1
            if ws and ws.getImage():
                img = ws.getImage()
                plate_path = f"{row}/{col}/{field_name}"
                saver.set_image(img, plate_path)
                masks = get_shapes(img, shape_types)
                if masks:
                    if args.label_map:
                        label_map = get_label_map(masks, args.label_map)
                        for name, values in label_map.items():
                            print(f"Label map: {name} (count: {len(values)})")
                            saver.save(values, name)
                    else:
                        saver.save(list(masks.values()), args.label_name)
                print_status(int(t0), int(time.time()), count, total)


def get_label_map(masks: Dict, label_map_arg: str) -> Dict:
    label_map = defaultdict(list)
    roi_map = {}
    for roi_id, roi in masks.items():
        roi_map[roi_id] = roi

    try:
        for line in finput(label_map_arg):
            line = line.strip()
            sid, name, roi = line.split(",")
            label_map[name].append(roi_map[int(roi)])
    except (KeyError, ValueError) as e:
        # KeyError: if a roi is missing from the roi_map
        # ValueError: if there aren't enough separators
        print(f"Error parsing {label_map_arg}: {e}")
    return label_map


def get_shapes(image: omero.gateway.ImageWrapper, shape_types: List[str]) -> Dict:
    shape_classes = []
    for klass in shape_types:
        if klass in SHAPE_TYPES:
            shape_classes.append(SHAPE_TYPES[klass])

    conn = image._conn
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(image.id, None, {"omero.group": "-1"})

    masks = {}
    shape_count = 0
    for roi in result.rois:
        mask_shapes = []
        for s in roi.copyShapes():
            if isinstance(s, tuple(shape_classes)):
                mask_shapes.append(s)

        if len(mask_shapes) > 0:
            masks[roi.id.val] = mask_shapes
            shape_count += len(mask_shapes)

    print(f"Found {shape_count} mask shapes in {len(masks)} ROIs")
    return masks


def image_shapes_to_zarr(
    image: omero.gateway.ImageWrapper, shape_types: List[str], args: argparse.Namespace
) -> None:
    """
    Export shapes of type "Mask" or "Polygon" on an Image to OME-Zarr labels

    @param shape_types      e.g. ["Mask", "Polygon"]
    """

    masks = get_shapes(image, shape_types)

    dtype = MASK_DTYPE_SIZE[int(args.label_bits)]

    if args.style == "labeled" and args.label_bits == "1":
        print("Boolean type makes no sense for labeled. Using 64")
        dtype = MASK_DTYPE_SIZE[64]

    if masks:
        saver = MaskSaver(
            None,
            image,
            dtype,
            args.label_path,
            args.style,
            args.source_image,
            args.overlaps,
            args.output,
            args.name_by,
        )

        if args.style == "split":
            for roi_id, roi in masks.items():
                saver.save([roi], str(roi_id))
        else:
            if args.label_map:
                label_map = get_label_map(masks, args.label_map)
                for name, values in label_map.items():
                    print(f"Label map: {name} (count: {len(values)})")
                    saver.save(values, name)
            else:
                saver.save(list(masks.values()), args.label_name)
    else:
        print("No masks found on Image")


class MaskSaver:
    """
    Action class containing the parameters needed for mapping from
    masks to zarr groups/arrays.
    """

    OVERLAPS = ("error", "dtype_max")

    def __init__(
        self,
        plate: Optional[omero.gateway.PlateWrapper],
        image: Optional[omero.gateway.ImageWrapper],
        dtype: np.dtype,
        path: str = "labels",
        style: str = "labeled",
        source: str = "..",
        overlaps: str = "error",
        output: Optional[str] = None,
        name_by: str = "id",
    ) -> None:
        self.dtype = dtype
        self.path = path
        self.style = style
        self.source_image = source
        self.plate = plate
        self.plate_path = Optional[str]
        self.overlaps = overlaps
        self.output = output
        self.name_by = name_by
        if image:
            self.image = image
            self.size_t = image.getSizeT()
            self.size_c = image.getSizeC()
            self.size_z = image.getSizeZ()
            self.size_y = image.getSizeY()
            self.size_x = image.getSizeX()
            self.image_shape = (
                self.size_t,
                self.size_c,
                self.size_z,
                self.size_y,
                self.size_x,
            )

    def set_image(
        self, image: omero.gateway.ImageWrapper, plate_path: Optional[str]
    ) -> None:
        """
        Set the current image information, in case of plate
        MaskSaver.
        :param image: The image
        :param plate_path: The zarr path to the image
        :return: None
        """
        self.image = image
        self.size_t = image.getSizeT()
        self.size_c = image.getSizeC()
        self.size_z = image.getSizeZ()
        self.size_y = image.getSizeY()
        self.size_x = image.getSizeX()
        self.image_shape = (
            self.size_t,
            self.size_c,
            self.size_z,
            self.size_y,
            self.size_x,
        )
        if plate_path:
            self.plate_path = plate_path

    def save(self, masks: List[omero.model.Shape], name: str) -> None:
        """
        Save the masks/labels. In case of plate, make sure to set_image first.
        :param masks: The masks
        :param name: The name
        :return: None
        """

        # Figure out whether we can flatten some dimensions
        unique_dims: Dict[str, Set[int]] = {
            "T": {unwrap(mask.theT) for shapes in masks for mask in shapes},
            "Z": {unwrap(mask.theZ) for shapes in masks for mask in shapes},
        }
        ignored_dimensions: Set[str] = set()
        # We always ignore the C dimension
        ignored_dimensions.add("C")
        print(f"Unique dimensions: {unique_dims}")

        for d in "TZ":
            if unique_dims[d] == {None}:
                ignored_dimensions.add(d)

        filename = get_zarr_name(self.plate or self.image, self.output, self.name_by)

        # Verify that we are linking this mask to a real ome-zarr
        source_image = self.source_image
        print(f"source_image ??? needs to be None to use filename: {source_image}")
        print(f"filename: {filename}", self.output, self.name_by)
        source_image_link = self.source_image
        if source_image is None:
            # Assume that we're using the output directory
            source_image = filename
            source_image_link = "../.."  # Drop "labels/0"

        if self.plate:
            assert self.plate_path, "Need image path within the plate"
            source_image = f"{source_image}/{self.plate_path}"

        print(f"source_image {source_image}")
        image_path = source_image
        if self.output:
            image_path = os.path.join(self.output, source_image)
        src = parse_url(image_path)
        assert src, f"Source image does not exist at {image_path}"
        input_pyramid = Node(src, [])
        assert input_pyramid.load(Multiscales), "No multiscales metadata found"
        input_pyramid_levels = len(input_pyramid.data)

        store = open_store(image_path)
        label_group = open_group(store)

        _mask_shape: List[int] = list(self.image_shape)
        mask_shape: Tuple[int, ...] = tuple(_mask_shape)
        for d in ignored_dimensions:
            _mask_shape[DIMENSION_ORDER[d]] = 1
            mask_shape = tuple(_mask_shape)
        del _mask_shape
        print(f"Ignoring dimensions {ignored_dimensions}")

        if self.style not in ("labeled", "split"):
            assert False, "6d has been removed"

        # Create and store binary data
        labels, fill_colors, properties = self.masks_to_labels(
            masks,
            mask_shape,
            ignored_dimensions,
        )

        axes = marshal_axes(self.image)

        # For v0.3+ ngff we want to reduce the number of dimensions to
        # match the dims of the Image.
        dims_to_squeeze = []
        for dim, size in enumerate(self.image_shape):
            if size == 1:
                dims_to_squeeze.append(dim)
        labels = np.squeeze(labels, axis=tuple(dims_to_squeeze))

        scaler = Scaler(max_layer=input_pyramid_levels)
        label_pyramid = scaler.nearest(labels)
        transformations = marshal_transformations(self.image, levels=len(label_pyramid))

        # Specify and store metadata
        image_label_colors: List[JSONDict] = []
        label_properties: List[JSONDict] = []
        image_label = {
            "colors": image_label_colors,
            "properties": label_properties,
            "source": {"image": source_image_link},
        }
        if properties:
            for label_value, props_dict in sorted(properties.items()):
                new_dict: Dict = {"label-value": label_value, **props_dict}
                label_properties.append(new_dict)
        if fill_colors:
            for label_value, rgba_int in sorted(fill_colors.items()):
                image_label_colors.append(
                    {"label-value": label_value, "rgba": int_to_rgba_255(rgba_int)}
                )

        write_multiscale_labels(
            label_pyramid,
            label_group,
            name,
            axes=axes,
            coordinate_transformations=transformations,
            label_metadata=image_label,
        )

    def shape_to_binim_yx(
        self, shape: omero.model.Shape
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        if isinstance(shape, MaskI):
            return self._mask_to_binim_yx(shape)
        return self._polygon_to_binim_yx(shape)

    def _mask_to_binim_yx(
        self, mask: omero.model.Shape
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """
        :param mask MaskI: An OMERO mask

        :return: tuple of
                - Binary mask
                - (T, C, Z, Y, X, w, h) tuple of mask settings (T, C, Z may be
                None)

        TODO: Move to https://github.com/ome/omero-rois/
        """

        t = unwrap(mask.theT)
        c = unwrap(mask.theC)
        z = unwrap(mask.theZ)

        x = int(mask.x.val)
        y = int(mask.y.val)
        w = int(mask.width.val)
        h = int(mask.height.val)

        mask_packed = mask.getBytes()
        # convert bytearray into something we can use
        intarray = np.frombuffer(mask_packed, dtype=np.uint8)
        binarray = np.unpackbits(intarray).astype(self.dtype)
        # truncate and reshape
        binarray = np.reshape(binarray[: (w * h)], (h, w))

        return binarray, (t, c, z, y, x, h, w)

    def _polygon_to_binim_yx(
        self, polygon: omero.model.Shape
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        t = unwrap(polygon.theT)
        c = unwrap(polygon.theC)
        z = unwrap(polygon.theZ)

        # "10,20, 50,150, 200,200, 250,75"
        points = unwrap(polygon.points).strip()
        coords = OME_MODEL_POINT_LIST_RE.findall(points)
        x_coords = np.array([int(round(float(xy[0]))) for xy in coords])
        y_coords = np.array([int(round(float(xy[1]))) for xy in coords])

        # bounding box of polygon
        x = x_coords.min()
        y = y_coords.min()
        w = x_coords.max() - x
        h = y_coords.max() - y

        img = np.zeros((h, w), dtype=self.dtype)

        # coords *within* bounding box
        x_coords = x_coords - x
        y_coords = y_coords - y

        pixels = sk_polygon(y_coords, x_coords, img.shape)
        img[pixels] = 1

        return img, (t, c, z, y, x, h, w)

    def _get_indices(
        self, ignored_dimensions: Set[str], d: str, d_value: int, d_size: int
    ) -> List[int]:
        """
        Figures out which Z/C/T-planes a mask should be copied to
        """
        if d in ignored_dimensions:
            return [0]
        if d_value is not None:
            return [d_value]
        return range(d_size)

    def masks_to_labels(
        self,
        masks: List[omero.model.Mask],
        mask_shape: Tuple[int, ...],
        ignored_dimensions: Optional[Set[str]] = None,
        check_overlaps: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[int, str], Dict[int, Dict]]:
        """
        :param masks [MaskI]: Iterable container of OMERO masks
        :param mask_shape 5-tuple: the image dimensions (T, C, Z, Y, X), taking
            into account `ignored_dimensions`

        :param ignored_dimensions set(char): Ignore these dimensions and set
            size to 1

        :param check_overlaps bool: Whether to check for overlapping masks or
            not

        :return: Label image with size `mask_shape` as well as color metadata
            and dict of other properties.


        TODO: Move to https://github.com/ome/omero-rois/
        """

        # FIXME: hard-coded dimensions
        assert len(mask_shape) > 3
        size_t: int = mask_shape[0]
        size_c: int = mask_shape[1]
        size_z: int = mask_shape[2]
        ignored_dimensions = ignored_dimensions or set()

        roi_ids = [shape.roi.id.val for mask in masks for shape in mask]
        sorted_roi_ids = list(set(roi_ids))
        sorted_roi_ids.sort()

        if check_overlaps is None:
            # If overlaps isn't 'dtype_max', an exception is thrown
            # if any overlaps exist
            check_overlaps = self.overlaps != "dtype_max"

        # label values are 1...n
        max_value = len(sorted_roi_ids)
        # find most suitable dtype...
        labels_dtype = np.int64
        sorted_dtypes = [kv for kv in MASK_DTYPE_SIZE.items()]
        sorted_dtypes.sort(key=lambda x: x[0])
        # ignore first dtype (bool)
        for int_dtype in sorted_dtypes[1:]:
            dtype = int_dtype[1]
            # choose first dtype that handles max_value
            if np.iinfo(dtype).max >= max_value:
                labels_dtype = dtype
                break
        LOGGER.debug("Exporting labels to dtype %s", labels_dtype)
        labels = np.zeros(mask_shape, labels_dtype)

        for d in "TCZYX":
            if d in ignored_dimensions:
                assert (
                    labels.shape[DIMENSION_ORDER[d]] == 1
                ), f"Ignored dimension {d} should be size 1"
            assert (
                labels.shape == mask_shape
            ), f"Invalid label shape: {labels.shape}, expected {mask_shape}"

        fillColors: Dict[int, str] = {}
        properties: Dict[int, Dict] = {}

        for count, shapes in enumerate(masks):
            for shape in shapes:
                # Using ROI ID allows stitching label from multiple images
                # into a Plate and not creating duplicates from different iamges.
                # All shapes will be the same value (color) for each ROI
                shape_value = sorted_roi_ids.index(shape.roi.id.val) + 1
                properties[shape_value] = {
                    "omero:shapeId": shape.id.val,
                    "omero:roiId": shape.roi.id.val,
                }
                if shape.textValue:
                    properties[shape_value]["omero:text"] = unwrap(shape.textValue)
                if shape.fillColor:
                    fillColors[shape_value] = unwrap(shape.fillColor)
                binim_yx, (t, c, z, y, x, h, w) = self.shape_to_binim_yx(shape)
                for i_t in self._get_indices(ignored_dimensions, "T", t, size_t):
                    for i_c in self._get_indices(ignored_dimensions, "C", c, size_c):
                        for i_z in self._get_indices(
                            ignored_dimensions, "Z", z, size_z
                        ):
                            overlap = np.logical_and(
                                labels[i_t, i_c, i_z, y : (y + h), x : (x + w)].astype(
                                    bool
                                ),
                                binim_yx,
                            )
                            # ADD to the array, so zeros in our binarray don't
                            # wipe out previous shapes
                            labels[i_t, i_c, i_z, y : (y + h), x : (x + w)] += (
                                binim_yx * shape_value
                            )

                            if np.any(overlap):
                                if check_overlaps:
                                    raise Exception(
                                        f"Shape {shape.roi.id.val} overlaps "
                                        "with existing labels"
                                    )
                                else:
                                    # set overlapping region to max(dtype)
                                    labels[i_t, i_c, i_z, y : (y + h), x : (x + w)][
                                        overlap
                                    ] = np.iinfo(labels_dtype).max
        return labels, fillColors, properties
