import omero.clients  # noqa
from omero.model import MaskI
from omero.rtypes import unwrap
import numpy as np
import zarr


def image_masks_to_zarr(image, args):

    size_t = image.getSizeT()
    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_y = image.getSizeY()
    size_x = image.getSizeX()
    image_shape = (size_t, size_c, size_z, size_y, size_x)

    conn = image._conn
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(image.id, None)

    masks = {}
    for roi in result.rois:
        mask_shapes = []
        for s in roi.copyShapes():
            if isinstance(s, MaskI):
                mask_shapes.append(s)

        if len(mask_shapes) > 0:
            masks[roi.id.val] = mask_shapes

    print(f"Found {len(masks)} masks")

    if masks:
        name = f"{image.id}.zarr"
        root = zarr.open(name)
        if "masks" in root.group_keys():
            out_masks = root.masks
        else:
            out_masks = root.create_group("masks")

        # TODO: Make each ROI a separate group and use Roi.id as name?
        roi_name = "0"
        za = out_masks.create_dataset(
            roi_name,
            shape=image_shape,
            chunks=(1, 1, size_z, size_y, size_x),
            dtype=np.int16,
            overwrite=True,
        )
        masks_to_labels(masks, image_shape, check_overlaps=False, labels=za)
        # Setting za.attrs[] doesn't work, so go via parent
        if "0" in root:
            image_name = "0"
        else:
            image_name = "omero://{}.zarr".format(image.id)
        out_masks[roi_name].attrs["image"] = {
            "array": image_name,
            "source": {
                # 'ts': [],
                # 'cs': [],
                # 'zs': [],
                # 'ys': [],
                # 'xs': [],
            },
        }

        print("Created {}/{}".format(name, roi_name))
    else:
        print("No masks found on Image")


def mask_to_binim(mask, image_shape):
    """
    :param mask MaskI: An OMERO mask
    :param image_shape 5-tuple: the image dimensions (T, C, Z, Y, X)

    :return: Binary mask with the same dimensions as the image
             If `T`, `C` or `Z` are not set on the mask the mask is expanded to
             all planes in that dimension

    TODO: Move to https://github.com/ome/omero-rois/
    """
    size_t, size_c, size_z, size_y, size_x = image_shape

    def get_indicies(d, size_d):
        if d is not None:
            return [d]
        return range(size_d)

    # Create an nd-array same size as image
    binim = np.zeros(image_shape, np.bool)
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
    binarray = np.unpackbits(intarray)
    # truncate and reshape
    binarray = np.reshape(binarray[: (w * h)], (h, w))

    for i_t in get_indicies(t, size_t):
        for i_c in get_indicies(c, size_c):
            for i_z in get_indicies(z, size_z):
                binim[i_t, i_c, i_z, y : (y + h), x : (x + w)] = binarray

    return binim


def masks_to_labels(masks, image_shape, check_overlaps, labels=None):
    """
    :param masks [MaskI]: Iterable container of OMERO masks
    :param image_shape 5-tuple: the image dimensions (T, C, Z, Y, X)
    :param check_overlaps bool: Whether to check for overlapping masks or not
    :param labels nd-array: The optional output array, pass this if you have
           already created the array and want to fill it.

    :return: Label image with the same dimensions as the image
             If `T`, `C` or `Z` are not set on the mask the labels are expanded
             to all planes in that dimension

    TODO: Move to https://github.com/ome/omero-rois/
    """

    # Create np nd-array same size as image
    if not labels:
        # TODO: Set np.int size based on number of labels
        labels = np.zeros(image_shape, np.int16)
    assert (
        labels.shape == image_shape
    ), "labels must have the same shape as the image"

    for count, shapes in enumerate(masks.values()):
        # All shapes same color for each ROI
        print(count)
        for mask in shapes:
            binim = mask_to_binim(mask, image_shape)
            if check_overlaps and np.any(
                np.logical_and(labels.astype(np.bool), binim)
            ):
                raise Exception(
                    "Mask {} overlaps with existing labels".format(count)
                )

            # ADD to the array, so zeros in our binarray don't wipe out
            # previous masks
            labels[:] += binim * count

    return labels
