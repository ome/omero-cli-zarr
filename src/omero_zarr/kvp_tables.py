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
import csv

import omero.clients  # noqa

from .util import get_map_anns, get_zarr_name, map_anns_match


def plate_to_table(
    plate: omero.gateway._PlateWrapper, args: argparse.Namespace
) -> None:
    """
    Exports Well KVPs to a CSV table.
    """
    name = get_zarr_name(plate, args.output, args.name_by)
    skip_wells_map = args.skip_wells_map

    wells = list(plate.listChildren())
    # sort by row then column...
    wells = sorted(wells, key=lambda x: (x.row, x.column))
    well_count = len(wells)

    well_kvps_by_id = get_map_anns(wells)

    if skip_wells_map:
        # skip_wells_map is like MyKey:MyValue.
        # Or wild-card MyKey:* or MyKey:Val*
        wells = [
            well
            for well in wells
            if not map_anns_match(well_kvps_by_id.get(well.id, {}), skip_wells_map)
        ]
        print(
            f"Skipping {well_count - len(wells)} out of {well_count} wells"
            f" with skip_wells_map: {skip_wells_map}"
        )

    keys_set = set()

    for well in wells:
        kvps = well_kvps_by_id.get(well.id, {})
        for key in kvps.keys():
            keys_set.add(key)

    column_names = list(keys_set)
    column_names = sorted(column_names)

    print("column_names", column_names)

    plate_name = plate.getName()

    # write csv file...
    csv_name = name.replace(".ome.zarr", ".csv")
    print(f"Writing CSV file: {csv_name}")
    with open(csv_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Plate", "Well"] + column_names)

        for well in wells:
            kvps = well_kvps_by_id.get(well.id, {})
            row = [plate_name, f"{well.getWellPos()}"]
            for key in column_names:
                row.append(";".join(kvps.get(key, [])))
            writer.writerow(row)
