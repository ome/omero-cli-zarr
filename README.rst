.. image:: https://github.com/ome/omero-cli-zarr/workflows/Precommit/badge.svg
   :target: https://github.com/ome/omero-cli-zarr/actions

.. image:: https://badge.fury.io/py/omero-cli-zarr.svg
    :target: https://badge.fury.io/py/omero-cli-zarr

OMERO CLI Zarr plugin
=====================

This OMERO command-line plugin allows you to export Images and Plates
from OMERO as zarr files, according to the spec at
https://github.com/ome/omero-ms-zarr/blob/master/spec.md
as well as Masks associated with Images.

Images are 5D arrays of shape `(t, c, z, y, x)`.
Plates are a hierarchy of `plate/row/column/field(image)`.
Masks are 2D bitmasks which can exist on muliplte planes of an Image.
In `ome-zarr` sets of Masks are collected together into "labels".

It supports export using 2 alternative methods:

- By default the OMERO API is used to load planes as numpy arrays
  and the zarr file is created from this data. NB: currently, large
  tiled images are not supported by this method.

- Alternatively, if you can read directly from the OMERO binary
  repository and have installed https://github.com/glencoesoftware/bioformats2raw
  then you can use this to create zarr files.


Usage
-------

Images and Plates
^^^^^^^^^^^^^^^^^

To export Images or Plates via the OMERO API::


    # Image will be saved in current directory as 1.zarr
    $ omero zarr export Image:1

    # Plate will be saved in current directory as 2.zarr
    $ omero zarr export Plate:2

    # Specify an output directory
    $ omero zarr --output /home/user/zarr_files export Image:1

    # Cache each plane as a numpy file.npy. If connection is lost, and you need
    # to export again, we can use these instead of downloading again
    # omero zarr --cache_numpy export Image:1


To export images via bioformats2raw we use the ```--bf``` flag::

    export MANAGED_REPO=/var/omero/data/ManagedRepository
    export BF2RAW=/opt/tools/bioformats2raw-0.2.0-SNAPSHOT

    $ omero zarr --output /home/user/zarr_files export 1 --bf
    Image exported to /home/user/zarr_files/2chZT.lsm

Masks
^^^^^

To export Masks for an Image or Plate::

    # Saved under 1.zarr/labels/0 - 1.zarr/ must already exist
    $ omero zarr masks Image:1

    # Labels saved under each image. e.g 2.zarr/A/1/0/labels/0
    # Plate should already be exported
    $ omero zarr masks Plate:2

    # Saved under zarr_files/1.zarr/labels/0
    $ omero zarr --output /home/user/zarr_files masks Image:1

    # Specify the label-name. (default is '0')
    # e.g. Export to 1.zarr/labels/A
    $ omero zarr masks Image:1 --label-name=A

The default behaviour is to export all masks on the Image to a single 5D
"labeled" zarr array, with a different value for each mask Shape.
An exception will be thrown if any of the masks overlap.

To handle overlapping masks, split masks into non-overlapping zarr groups
using a "label-map" which is a csv file of that specifies the name of
the zarr group for each ROI on the Image. Columns are ID, NAME, ROI_ID.

For example, to create a group from the `textValue` of each Shape,
you can use this command::

    omero hql --style=plain "select distinct s.textValue, s.roi.id from Shape s where s.roi.image.id = 5514375" --limit=-1 | tee 5514375.rois

This creates a file `5514375.rois` like this::

    0,Cell,1369132
    1,Cell,1369134
    2,Cell,1369136
    ...
    40,Chromosomes,1369131
    41,Chromosomes,1369133
    42,Chromosomes,1369135
    ...

This will create zarr groups of `Cell` and `Chromosomes` under `5514375.zarr/labels/`::

    $ omero zarr masks Image:5514375 --label-map=5514375.rois
