from ome_zarr.format import CurrentFormat

from ._version import version as __version__

ngff_version = CurrentFormat().version

__all__ = [
    "__version__",
]
