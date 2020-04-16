OMERO CLI Zarr plugin
=====================

Example usage:
```
export MANAGED_REPO=/var/omero/data/ManagedRepository
export BF2RAW=/opt/tools/bioformats2raw-0.2.0-SNAPSHOT

# Export image with ID 1
> omero zarr 1 /home/user/zarr_files
Using session for root@localhost:4064. Idle timeout: 10 min. Current group: system
Image exported to /home/user/zarr_files/2chZT.lsm
```
