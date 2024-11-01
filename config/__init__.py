from yaml import safe_load

from os.path import realpath, dirname
from os import chdir, getcwd, PathLike


def get_yaml_dict(path: str) -> dict:
    with _OpenRelativePath(path) as stream:
        load = safe_load(stream)
    return load


class _OpenRelativePath:
    """
    Context manager for dealing with reading package-level files.
    """

    def __init__(self, localpath: PathLike, mode: str = "rt") -> None:
        self.localpath = localpath
        self.mode = mode

    def __enter__(self):
        self.root = getcwd()
        real = realpath(__file__)
        local = dirname(real)
        chdir(local)
        self.f_handle = open(self.localpath, self.mode)
        return self.f_handle

    def __exit__(self, exc_type, exc_value, traceback):
        self.f_handle.close()
        chdir(self.root)


MODEL_CONFIG = get_yaml_dict("model.yaml")
DATA_CONFIG = get_yaml_dict("data.yaml")
