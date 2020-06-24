import omero.clients  # noqa
from omero.model import MaskI
from omero.rtypes import unwrap
import numpy as np
import zarr


def image_masks_to_zarr(image, args):

    size_x = image.getSizeX()
    size_y = image.getSizeY()

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
        stack = masks_to_zarr(masks, image)
        name = f"{image.id}_masks.zarr"
        root = zarr.open_group(name, mode="w")
        za = root.create(
            "0",
            shape=stack.shape,
            chunks=(1, 1, size_y, size_x),
            dtype=stack.dtype,
        )

        za[:, :, :, :] = stack

        print("Created", name)
    else:
        print("No masks found on Image")


def masks_to_zarr(masks, image):

    # Create np nd-array same size as image
    size_t = image.getSizeT()
    size_z = image.getSizeZ()
    size_x = image.getSizeX()
    size_y = image.getSizeY()

    labels = np.zeros((size_t, size_z, size_y, size_x))
    for count, shapes in enumerate(masks.values()):
        # All shapes same color for each ROI
        for mask in shapes:
            t = unwrap(mask.theT) or 0
            z = unwrap(mask.theZ) or 0
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
            # ADD to the array, so zeros in our binarray don't wipe out
            # previous masks
            labels[t, z, y : (y + h), x : (x + w)] += binarray * (count)

    return labels
