import omero.clients  # noqa
from omero.model import MaskI
from omero.rtypes import unwrap
import numpy as np
import zarr


# Mapping of dimension names to axes in the Zarr
DIMENSION_ORDER = {
    "T": 0,
    "C": 1,
    "Z": 2,
    "Y": 3,
    "X": 4,
}

MASK_DTYPE_SIZE = {
    1: np.bool,
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
}


def image_masks_to_zarr(image, args):

    conn = image._conn
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(image.id, None)

    masks = {}
    shape_count = 0
    for roi in result.rois:
        mask_shapes = []
        for s in roi.copyShapes():
            if isinstance(s, MaskI):
                mask_shapes.append(s)

        if len(mask_shapes) > 0:
            masks[roi.id.val] = mask_shapes
            shape_count += len(mask_shapes)

    print(f"Found {shape_count} mask shapes in {len(masks)} ROIs")

    dtype = MASK_DTYPE_SIZE[int(args.mask_bits)]

    if args.style == "labelled" and args.mask_bits == "1":
        print("Boolean type makes no sense for labelled. Using 64")
        dtype = MASK_DTYPE_SIZE[64]

    if masks:
        saver = MaskSaver(image, dtype, args.mask_path, args.style)
        if args.style == "split":
            for (roi_id, roi) in masks.items():
                saver.save([roi], str(roi_id))
        else:
            saver.save(masks.values(), args.mask_name)
    else:
        print("No masks found on Image")


class MaskSaver:
    def __init__(self, image, dtype, path="masks", style="6d"):
        self.image = image
        self.dtype = dtype
        self.path = path
        self.style = style
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

    def save(self, masks, name):

        # Figure out whether we can flatten some dimensions
        unique_dims = {
            "T": set(),
            "C": set(),
            "Z": set(),
        }
        for shapes in masks:
            for mask in shapes:
                unique_dims["T"].add(unwrap(mask.theT))
                unique_dims["C"].add(unwrap(mask.theC))
                unique_dims["Z"].add(unwrap(mask.theZ))
        ignored_dimensions = set()
        print(f"Unique dimensions: {unique_dims}")

        for d in "TCZ":
            if unique_dims[d] == {None}:
                ignored_dimensions.add(d)

        filename = f"{self.image.id}.zarr"
        root = zarr.open(filename)
        if self.path in root.group_keys():
            out_masks = getattr(root, self.path)
        else:
            out_masks = root.create_group(self.path)

        mask_shape = list(self.image_shape)
        for d in ignored_dimensions:
            mask_shape[DIMENSION_ORDER[d]] = 1
        print("Ignoring dimensions {}".format(ignored_dimensions))

        if self.style in ("labelled", "split"):

            za = out_masks.create_dataset(
                name,
                shape=mask_shape,
                chunks=(1, 1, 1, self.size_y, self.size_x),
                dtype=self.dtype,
                overwrite=True,
            )

            self.masks_to_labels(
                masks,
                mask_shape,
                ignored_dimensions,
                check_overlaps=True,
                labels=za,
            )

        else:
            assert self.style == "6d"
            za = out_masks.create_dataset(
                name,
                shape=tuple([len(masks)] + mask_shape),
                chunks=(1, 1, 1, 1, self.size_y, self.size_x),
                dtype=self.dtype,
                overwrite=True,
            )

            self.stack_masks(
                masks,
                mask_shape,
                ignored_dimensions,
                check_overlaps=True,
                target=za,
            )

        # Setting za.attrs[] doesn't work, so go via parent
        if "0" in root:
            image_name = "../../0"
        else:
            image_name = "omero://{}.zarr".format(self.image.id)
        out_masks[name].attrs["image"] = {
            "array": image_name,
            "source": {
                # 'ts': [],
                # 'cs': [],
                # 'zs': [],
                # 'ys': [],
                # 'xs': [],
            },
        }

        print(f"Created {filename}/{self.path}/{name}")

    def _mask_to_binim_yx(self, mask):
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
        intarray = np.fromstring(mask_packed, dtype=np.uint8)
        binarray = np.unpackbits(intarray).astype(self.dtype)
        # truncate and reshape
        binarray = np.reshape(binarray[: (w * h)], (h, w))

        return binarray, (t, c, z, y, x, h, w)

    def _get_indices(self, ignored_dimensions, d, d_value, d_size):
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
        masks,
        mask_shape,
        ignored_dimensions=None,
        check_overlaps=True,
        labels=None,
    ):
        """
        :param masks [MaskI]: Iterable container of OMERO masks
        :param mask_shape 5-tuple: the image dimensions (T, C, Z, Y, X), taking
            into account `ignored_dimensions`
        :param ignored_dimensions set(char): Ignore these dimensions and set
            size to 1
        :param check_overlaps bool: Whether to check for overlapping masks or
            not
        :param labels nd-array: The optional output array, pass this if you
            have already created the array and want to fill it.

        :return: Label image with size `mask_shape`

        TODO: Move to https://github.com/ome/omero-rois/
        """

        size_t, size_c, size_z, size_y, size_x = mask_shape
        ignored_dimensions = ignored_dimensions or set()
        mask_shape = tuple(mask_shape)

        if not labels:
            # TODO: Set np.int size based on number of labels
            labels = np.zeros(mask_shape, np.int64)

        for d in "TCZYX":
            if d in ignored_dimensions:
                assert (
                    labels.shape[DIMENSION_ORDER[d]] == 1
                ), "Ignored dimension {} should be size 1".format(d)
            assert (
                labels.shape == mask_shape
            ), "Invalid label shape: {}, expected {}".format(
                labels.shape, mask_shape
            )

        for count, shapes in enumerate(masks):
            # All shapes same color for each ROI
            print(count)
            for mask in shapes:
                binim_yx, (t, c, z, y, x, h, w) = self._mask_to_binim_yx(mask)
                for i_t in self._get_indices(
                    ignored_dimensions, "T", t, size_t
                ):
                    for i_c in self._get_indices(
                        ignored_dimensions, "C", c, size_c
                    ):
                        for i_z in self._get_indices(
                            ignored_dimensions, "Z", z, size_z
                        ):
                            if check_overlaps and np.any(
                                np.logical_and(
                                    labels[
                                        i_t, i_c, i_z, y : (y + h), x : (x + w)
                                    ].astype(np.bool),
                                    binim_yx,
                                )
                            ):
                                raise Exception(
                                    (
                                        f"Mask {count} overlaps "
                                        "with existing labels"
                                    )
                                )
                            # ADD to the array, so zeros in our binarray don't
                            # wipe out previous masks
                            labels[
                                i_t, i_c, i_z, y : (y + h), x : (x + w)
                            ] += (
                                binim_yx * (count + 1)  # Prevent zeroing
                            )

        return labels

    def stack_masks(
        self,
        masks,
        mask_shape,
        ignored_dimensions=None,
        check_overlaps=True,
        target=None,
    ):
        """
        :param masks [MaskI]: Iterable container of OMERO masks
        :param mask_shape 5-tuple: the image dimensions (T, C, Z, Y, X), taking
            into account `ignored_dimensions`
        :param ignored_dimensions set(char): Ignore these dimensions and set
            size to 1
        :param check_overlaps bool: Whether to check for overlapping masks or
            not
        :param labels nd-array: The optional output array, pass this if you
            have already created the array and want to fill it.

        :return: Array with one extra dimension than `mask_shape`

        TODO: Move to https://github.com/ome/omero-rois/
        """

        size_t, size_c, size_z, size_y, size_x = mask_shape
        ignored_dimensions = ignored_dimensions or set()
        mask_shape = tuple(mask_shape)

        if not target:
            raise Exception("No target")

        for d in "TCZYX":
            if d in ignored_dimensions:
                assert (
                    target.shape[DIMENSION_ORDER[d] + 1] == 1
                ), "Ignored dimension {} should be size 1".format(d)
            assert target.shape == tuple(
                [len(masks)] + list(mask_shape)
            ), "Invalid label shape: {}, expected {}".format(
                target.shape, mask_shape
            )
            assert True

        for count, shapes in enumerate(masks):
            # All shapes same color for each ROI
            print(count)
            for mask in shapes:
                binim_yx, (t, c, z, y, x, h, w) = self._mask_to_binim_yx(mask)
                for i_t in self._get_indices(
                    ignored_dimensions, "T", t, size_t
                ):
                    for i_c in self._get_indices(
                        ignored_dimensions, "C", c, size_c
                    ):
                        for i_z in self._get_indices(
                            ignored_dimensions, "Z", z, size_z
                        ):
                            target[
                                count, i_t, i_c, i_z, y : (y + h), x : (x + w)
                            ] += (
                                binim_yx  # Here one could assign probabilities
                            )

        return target
