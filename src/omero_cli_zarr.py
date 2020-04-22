import sys
import os
import subprocess
from pathlib import Path
from functools import wraps

import omero
from omero.cli import BaseControl
from omero.cli import CLI
from omero.gateway import BlitzGateway
from omero.rtypes import rlong

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

    parser.add_argument("image_id", type=int, help="The Image to export")
    parser.add_argument("target", type=str, help="The target directory")

    parser.add_argument("--tile_width", default=None)
    parser.add_argument("--tile_height", default=None)
    parser.add_argument("--resolutions", default=None)
    parser.add_argument("--max_workers", default=None)

    parser.set_defaults(func=self.export)

  @gateway_required
  def export(self, args):
    path, name = self._get_path(args.image_id)

    if path:
      self._do_export(path, name, args)
    else:
      print("Couldn't find managed repository path for this image.")


  def _do_export(self, path, name, args):
    abs_path = Path(os.environ['MANAGED_REPO']) / path / name

    bf2raw = Path(os.environ['BF2RAW'])

    target = Path(args.target) / name
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

    print(options)
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