"""Module for downsampling numpy arrays via various methods.
Copied from ome-zarr-py

See the :class:`~ome_zarr.scale.Scaler` class for details.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

import dask.array as da
import numpy as np
from skimage.transform import resize

from .util import resize as dask_resize

LOGGER = logging.getLogger("ome_zarr.scale")

ListOfArrayLike = Union[list[da.Array], list[np.ndarray]]
ArrayLike = Union[da.Array, np.ndarray]


@dataclass
class Scaler:
    """Helper class for performing various types of downsampling.

    A method can be chosen by name such as "nearest".

    Attributes:
        downscale:
            Downscaling factor.
        max_layer:
            The maximum number of downsampled layers to create.

    >>> import numpy as np
    >>> data = np.zeros((1, 1, 1, 64, 64))
    >>> scaler = Scaler()
    >>> downsampling = scaler.nearest(data)
    >>> for x in downsampling:
    ...     print(x.shape)
    (1, 1, 1, 64, 64)
    (1, 1, 1, 32, 32)
    (1, 1, 1, 16, 16)
    (1, 1, 1, 8, 8)
    (1, 1, 1, 4, 4)
    """

    downscale: int = 2
    max_layer: int = 4

    def resize_image(self, image: ArrayLike) -> ArrayLike:
        """
        Resize a numpy array OR a dask array to a smaller array (not pyramid)
        """
        if isinstance(image, da.Array):

            def _resize(image: ArrayLike, out_shape: tuple, **kwargs: Any) -> ArrayLike:
                return dask_resize(image, out_shape, **kwargs)

        else:
            _resize = resize

        # only down-sample in X and Y dimensions for now...
        new_shape = list(image.shape)
        new_shape[-1] = image.shape[-1] // self.downscale
        new_shape[-2] = image.shape[-2] // self.downscale
        out_shape = tuple(new_shape)

        dtype = image.dtype
        image = _resize(
            image.astype(float), out_shape, order=1, mode="reflect", anti_aliasing=False
        )
        return image.astype(dtype)

    def nearest(self, base: np.ndarray) -> list[np.ndarray]:
        """
        Downsample using :func:`skimage.transform.resize`.
        """
        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: ArrayLike, sizeY: int, sizeX: int) -> np.ndarray:
        """Apply the 2-dimensional transformation."""
        if isinstance(plane, da.Array):

            def _resize(
                image: ArrayLike, output_shape: tuple, **kwargs: Any
            ) -> ArrayLike:
                return dask_resize(image, output_shape, **kwargs)

        else:
            _resize = resize

        return _resize(
            plane,
            output_shape=(sizeY // self.downscale, sizeX // self.downscale),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(plane.dtype)

    #
    # Helpers
    #

    def _by_plane(
        self,
        base: np.ndarray,
        func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:
        """Loop over 3 of the 5 dimensions and apply the func transform."""

        rv = [base]
        for i in range(self.max_layer):
            stack_to_scale = rv[-1]
            shape_5d = (*(1,) * (5 - stack_to_scale.ndim), *stack_to_scale.shape)
            T, C, Z, Y, X = shape_5d

            # If our data is already 2D, simply resize and add to pyramid
            if stack_to_scale.ndim == 2:
                rv.append(func(stack_to_scale, Y, X))
                continue

            # stack_dims is any dims over 2D
            stack_dims = stack_to_scale.ndim - 2
            new_stack = None
            for t in range(T):
                for c in range(C):
                    for z in range(Z):
                        dims_to_slice = (t, c, z)[-stack_dims:]
                        # slice nd down to 2D
                        plane = stack_to_scale[(dims_to_slice)][:]
                        out = func(plane, Y, X)
                        # first iteration of loop creates the new nd stack
                        if new_stack is None:
                            zct_dims = shape_5d[:-2]
                            shape_dims = zct_dims[-stack_dims:]
                            new_stack = np.zeros(
                                (*shape_dims, out.shape[0], out.shape[1]),
                                dtype=base.dtype,
                            )
                        # insert resized plane into the stack at correct indices
                        new_stack[(dims_to_slice)] = out
            rv.append(new_stack)
        return rv
