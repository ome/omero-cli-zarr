.. image:: https://github.com/ome/omero-cli-zarr/workflows/Precommit/badge.svg
   :target: https://github.com/ome/omero-cli-zarr/actions

.. image:: https://badge.fury.io/py/omero-cli-zarr.svg
    :target: https://badge.fury.io/py/omero-cli-zarr

OMERO CLI Zarr plugin
=====================

This OMERO command-line plugin allows you to import and export
Images and ROIs from OME-Zarr format into OMERO and vice versa.

Import
------

NB: This feature currently works only with OME-Zarr files hosted
on public object stores OR a local filesystem accessible to the OMERO server.

The import process is similar to an "in-place" import; It does not move the
data into OMERO. Instead the data is accessed via the
`omero-zarr-pixel-buffer <https://github.com/glencoesoftware/omero-zarr-pixel-buffer>`_,
which must be installed on the OMERO server.

Images are created in OMERO and the link to the OME-Zarr data is stored in
the `ExternalInfo` of the Image. Images are created in one of two ways:

- If the OME-Zarr data is in the `bioformats2raw` layout, which includes an
  `OME/METADATA.ome.xml` file, then this file is copied to the the server and
  imported to create a Fileset and one or more Images. This preserves all of
  the OME metadata in the xml file.

- Otherwise, Images are created via the OMERO API, using the metadata
  in the OME-Zarr files.

Export
------

This feature supports export of any regular Images and Plates
from OMERO into the OME-Zarr format, according to the spec at
https://ngff.openmicroscopy.org/0.4/index.html
as well as Masks and Polygons on the Images.

Images are exported as nD arrays of shape, up to `(t, c, z, y, x)`.
Plates are a hierarchy of `plate/row/column/field(image)`.
Masks in OMERO are 2D bitmasks which can exist on multiple planes of an Image.
Polygons are 2D shapes defined by a set of vertices. In both cases,
they are exported as OME-Zarr `labels`, with Polygons being converted to
bitmasks during export.

Export is supported using 2 alternative methods:

- By default the OMERO API is used to load planes as numpy arrays
  and the zarr file is created from this data.

- Alternatively, if you can read directly from the OMERO binary
  repository and have installed https://github.com/glencoesoftware/bioformats2raw
  then you can use this to create zarr files.


Usage
-------

Import OME-Zarr files
^^^^^^^^^^^^^^^^^^^^^

To import a public OME-Zarr data into OMERO, use the `import` command::

    $ omero zarr import https://example.com/path/to/file.ome.zarr

    # Use --target DATASET_ID to import into a Dataset
    $ omero zarr import https://example.com/path/to/file.ome.zarr --target 1234

    # Or import local data which must be accessible to the OMERO server
    $ omero zarr import /path/to/file.ome.zarr


Export Images and Plates
^^^^^^^^^^^^^^^^^^^^^^^^

To export Images or Plates via the OMERO API::


    # Image will be saved in current directory as 1.ome.zarr
    $ omero zarr export Image:1

    # Plate will be saved in current directory as 2.ome.zarr
    $ omero zarr export Plate:2

    # Use the Image or Plate 'name' to save e.g. my_image.ome.zarr
    $ omero zarr --name_by name export Image:1

    # Specify an output directory
    $ omero zarr --output /home/user/zarr_files export Image:1

    # By default, a tile (chunk) size of 1024 is used. Specify values with
    $ omero zarr export Image:1 --tile_width 256 --tile_height 256


NB: If the connection to OMERO is lost and the Image is partially exported,
re-running the command will attempt to complete the export.

To export images via bioformats2raw we use the ```--bf``` flag::

    export MANAGED_REPO=/var/omero/data/ManagedRepository
    export BF2RAW=/opt/tools/bioformats2raw-0.2.0-SNAPSHOT

    $ omero zarr --output /home/user/zarr_files export 1 --bf
    Image exported to /home/user/zarr_files/2chZT.lsm

Export Masks and Polygons
^^^^^^^^^^^^^^^^^^^^^^^^^

To export Masks or Polygons for an Image or Plate, use the `masks` or `polygons` command::

    # Saved under 1.ome.zarr/labels/0
    # 1.ome.zarr/ should already exist...
    $ omero zarr masks Image:1

    # ...or specify path with --source-image
    $ omero zarr masks Image:1 --source-image my_image.ome.zarr

    # Labels saved under each image. e.g 2.ome.zarr/A/1/0/labels/0
    # 2.ome.zarr should already be exported or specify path with --source-image
    $ omero zarr masks Plate:2

    # Saved under zarr_files/1.ome.zarr/labels/0
    $ omero zarr --output /home/user/zarr_files masks Image:1

    # Specify the label-name. (default is '0')
    # e.g. Export to 1.ome.zarr/labels/A
    $ omero zarr masks Image:1 --label-name=A

    # Allow overlapping masks or polygons (overlap will be maximum value of the dtype)
    $ omero zarr polygons Image:1 --overlaps=dtype_max

The default behaviour is to export all masks or polygons on the Image to a single nD
"labeled" zarr array, with a different value for each Shape.
An exception will be thrown if any of the masks overlap, unless the `--overlaps`
option is used as above.

An alternative to handle overlapping masks is to split masks into non-overlapping zarr
groups using a "label-map" which is a csv file that specifies the name of
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

License
-------

This project, similar to many Open Microscopy Environment (OME) projects, is
licensed under the terms of the GNU General Public License (GPL) v2 or later.

Copyright
---------

2020-2023, The Open Microscopy Environment
