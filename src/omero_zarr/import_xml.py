#!/usr/bin/env python

#
# Copyright (C) 2025 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import platform
import sys
import tempfile

import omero
from omero.callbacks import CmdCallbackI
from omero.clients import BaseClient
from omero.cmd import ERR as CmdErr
from omero.grid import ImportProcessPrx, ImportResponse
from omero.model import ChecksumAlgorithmI, NamedValue
from omero.model.enums import ChecksumAlgorithmSHA1160
from omero.rtypes import rbool, rstring
from omero_version import omero_version


def create_fileset() -> omero.model.FilesetI:
    """Create a new Fileset with single OME XML file."""
    fileset = omero.model.FilesetI()
    entry = omero.model.FilesetEntryI()
    # NB: If the clientPath includes .zarr, Bio-Formats tries to import zarr group
    entry.setClientPath(rstring("OME/METADATA.ome.xml"))
    fileset.addFilesetEntry(entry)

    # Fill version info
    system, node, release, version, machine, processor = platform.uname()

    client_version_info = [
        NamedValue("omero.version", omero_version),
        NamedValue("os.name", system),
        NamedValue("os.version", release),
        NamedValue("os.architecture", machine),
    ]

    upload = omero.model.UploadJobI()
    upload.setVersionInfo(client_version_info)
    fileset.linkJob(upload)
    return fileset


def create_settings() -> omero.grid.ImportSettings:
    """Create ImportSettings and set some values."""
    settings = omero.grid.ImportSettings()
    # can't create thumbnails on import since ExternalInfo is not set yet
    settings.doThumbnails = rbool(False)
    settings.noStatsInfo = rbool(False)
    settings.userSpecifiedTarget = None
    settings.userSpecifiedName = None
    settings.userSpecifiedDescription = None
    settings.userSpecifiedAnnotationList = None
    settings.userSpecifiedPixels = None
    settings.checksumAlgorithm = ChecksumAlgorithmI()
    s = rstring(ChecksumAlgorithmSHA1160)
    settings.checksumAlgorithm.value = s
    return settings


def upload_file(
    proc: ImportProcessPrx, omexml_bytes: bytes, client: BaseClient
) -> list[str]:
    """Upload files to OMERO from local filesystem."""
    ret_val = []
    i = 0
    rfs = proc.getUploader(i)
    try:
        offset = 0
        # rfs.write([], offset, 0)  # Touch
        # Write the OME XML file
        rfs.write(omexml_bytes, offset, len(omexml_bytes))

        # create temp file for sha1
        # error: No overload variant of "NamedTemporaryFile" matches argument
        # type "bool" [call-overload]
        with tempfile.NamedTemporaryFile(delete_on_close=False) as fp:  # type: ignore
            fp.write(omexml_bytes)
            fp.close()
            ret_val.append(client.sha1(fp.name))
    finally:
        rfs.close()
    return ret_val


def assert_import(
    client: BaseClient, proc: ImportProcessPrx, omexml_bytes: bytes, wait: int
) -> ImportResponse:
    """Wait and check that we imported an image."""
    hashes = upload_file(proc, omexml_bytes, client)
    # print ('Hashes:\n  %s' % '\n  '.join(hashes))
    handle = proc.verifyUpload(hashes)
    cb = CmdCallbackI(client, handle)

    # https://github.com/openmicroscopy/openmicroscopy/blob/v5.4.9/components/blitz/src/ome/formats/importer/ImportLibrary.java#L631
    if wait == 0:
        cb.close(False)
        return None
    if wait < 0:
        while not cb.block(2000):
            sys.stdout.write(".")
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        cb.loop(wait, 1000)
    rsp = cb.getResponse()
    print("rsp", rsp.__class__)
    if isinstance(rsp, CmdErr):
        raise Exception(rsp)
    assert len(rsp.pixels) > 0
    return rsp


def full_import(
    client: BaseClient, omexml_bytes: bytes, wait: int = -1
) -> ImportResponse:
    """Re-usable method for a basic import."""
    mrepo = client.getManagedRepository()

    fileset = create_fileset()
    settings = create_settings()

    proc = mrepo.importFileset(fileset, settings)
    print(client.__class__)
    print(proc.__class__)
    try:
        # do the upload and trigger the import
        return assert_import(client, proc, omexml_bytes, wait)
    finally:
        proc.close()
