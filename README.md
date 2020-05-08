OMERO CLI Zarr plugin
=====================

This OMERO command-line plugin allows you to export images from
OMERO as zarr files, according to the spec at 
https://github.com/ome/omero-ms-zarr/blob/master/spec.md.

It supports export using 2 alternative methods:

- By default the OMERO API is used to load planes as numpy arrays
  and the zarr file is created from this data. NB: currently, large
  tiled images are not supported by this method.

- Alternatively, if you can read directly from the OMERO binary
  repository and have installed https://github.com/glencoesoftware/bioformats2raw
  then you can use this to create zarr files.


# Usage

To export images via the OMERO API:

```
# Image will be saved in current directory as 1.zarr
$ omero zarr Image:1

# Specify an output directory
$ omero zarr Image:1 --output /home/user/zarr_files

# Cache each plane as a numpy file.npy. If connection is lost, and you need
# to export again, we can use these instead of downloading again
# omero zarr Image:1 --cache_numpy

```

To export images via bioformats2raw:

```
export MANAGED_REPO=/var/omero/data/ManagedRepository
export BF2RAW=/opt/tools/bioformats2raw-0.2.0-SNAPSHOT

$ omero zarr 1 --output /home/user/zarr_files
Image exported to /home/user/zarr_files/2chZT.lsm
```
