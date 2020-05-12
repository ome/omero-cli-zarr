import sys
import os
import subprocess
from pathlib import Path
from functools import wraps

import omero
from omero.cli import BaseControl
from omero.cli import CLI
from omero.cli import ProxyStringType
from omero.config import ConfigXml
from omero.gateway import BlitzGateway
from omero.rtypes import rlong
from omero.model import ImageI

from raw_pixels import image_to_zarr

HELP = "Export an image in zarr format."

def gateway_required(func):
  """
  Decorator which initializes a client (self.client),
  a BlitzGateway (self.gateway), and makes sure that
  all services of the Blitzgateway are closed again.
  """
  @wraps(func)
  def _wrapper(self, *args, **kwargs):
    self.client = self.ctx.conn(*args)
    self.gateway = BlitzGateway(client_obj=self.client)

    try:
      return func(self, *args, **kwargs)
    finally:
      if self.gateway is not None:
        self.gateway.close(hard=False)
        self.gateway = None
        self.client = None
  return _wrapper


class ZarrControl(BaseControl):

  gateway = None
  client = None

  def _configure(self, parser):
    parser.add_login_arguments()

    ProxyStringType("Image")
    parser.add_argument("object", type=ProxyStringType("Image"),
      help="The Image to export.")
    parser.add_argument("--output", type=str, default="",
                        help="Full path to output directory")

    parser.add_argument(
      "--cache_numpy", action="store_true",
      help="Save planes as .npy files in case of connection loss")

    parser.add_argument("--bf",
                        help="Full path to bioformats2raw base directory.\
                        Use bioformats2raw to read directly from managed repo.")
    parser.add_argument("--tile_width", default=None,
                        help="For use with bioformats2raw")
    parser.add_argument("--tile_height", default=None,
                        help="For use with bioformats2raw")
    parser.add_argument("--resolutions", default=None,
                        help="For use with bioformats2raw")
    parser.add_argument("--max_workers", default=None,
                        help="For use with bioformats2raw")

    parser.set_defaults(func=self.export)


  def _get_repo(self):
    try:
      if 'OMERODIR' in os.environ:
        base_dir = Path(os.environ.get('OMERODIR'))
      else:
        self.ctx.die(1, 'OMERODIR env variable not set')
      grid_dir = base_dir / "etc" / "grid"
      cfg_xml = grid_dir / "config.xml"
      config = ConfigXml(str(cfg_xml))
      return config['omero.data.dir']
    except:
      self.ctx.die(2, "Couldn't find managed repository path.")


  @gateway_required
  def export(self, args):

    if isinstance(args.object, ImageI):
      image_id = args.object.id
      image = self._lookup(self.gateway, "Image", image_id)
      self.ctx.out("Export image: %s" % image.name)

      if args.bf:
        path, name = self._get_path(image_id)
        if path:
          self._do_export(path, name, args)
        else:
          self.ctx.die(3, "Couldn't find managed repository path for this image.")
      else:
        image_to_zarr(image, args)

  def _lookup(self, gateway, type, oid):
        """Find object of type by ID."""
        gateway.SERVICE_OPTS.setOmeroGroup("-1")
        obj = gateway.getObject(type, oid)
        if not obj:
            self.ctx.die(4, "No such %s: %s" % (type, oid))
        return obj

  def _do_export(self, path, name, args):
    repo_path = self._get_repo()
    abs_path = repo_path / Path("ManagedRepository") / path / name
    bf2raw = Path(args.bf)

    target = Path(args.output) / name
    target.mkdir(exist_ok=True)

    options = "--file_type=zarr"
    if args.tile_width:
      options += " --tile_width="+args.tile_width
    if args.tile_height:
      options += " --tile_height="+args.tile_height
    if args.resolutions:
      options += " --resolutions="+args.resolutions
    if args.max_workers:
      options += " --max_workers="+args.max_workers

    process = subprocess.Popen(["bin/bioformats2raw", options, abs_path, target],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=bf2raw)
    stdout, stderr = process.communicate()
    if stderr:
      print(stderr)
    else:
      print("Image exported to {}".format(target))


  def _get_path(self, image_id):
    query = "select org from Image i left outer join i.fileset as fs left outer join fs.usedFiles as uf left outer join uf.originalFile as org where i.id = :iid"
    qs = self.client.sf.getQueryService()
    params = omero.sys.Parameters()
    params.map = {"iid": rlong(image_id)}
    results = qs.findAllByQuery(query, params)
    for res in results:
      name = res.name._val
      path = res.path._val
      if not (name.endswith(".log") or name.endswith(".txt") or name.endswith(".xml")):
        return path, name


try:
  register("zarr", ZarrControl, HELP)
except NameError:
  if __name__ == "__main__":
    cli = CLI()
    cli.register("zarr", ZarrControl, HELP)
    cli.invoke(sys.argv[1:])