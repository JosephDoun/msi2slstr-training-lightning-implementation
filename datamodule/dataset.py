import os

from osgeo.gdal import Open
from osgeo.gdal import Dataset as GDataset
from osgeo.gdal import GA_ReadOnly

from torch import Tensor
from torch.utils.data import Dataset


class Image(Dataset):
    """
    Class holding image source and maps indices to tiles.
    """

    def __init__(self, imagepath: str, t_size: int, pad: int = 0) -> None:
        self.dataset: GDataset = Open(imagepath, GA_ReadOnly)
        self.info = imagepath.split(os.sep)
        self.date = self.info[1]
        self.tile = self.info[2]
        self.t_size = t_size
        self.pad = pad

        self.tile_coords = get_array_coords_list(t_size=t_size,
                                                 sizex=self.dataset
                                                 .RasterXSize,
                                                 sizey=self.dataset
                                                 .RasterYSize)

    def _get_tile(self, index):
        """
        Index to tile mapper method.
        """
        return (index  % self.dataset.RasterXSize * self.t_size,
                index // self.dataset.RasterYSize * self.t_size,
                self.t_size,
                self.t_size)

    def __getitem__(self, index):
        """
        Get i-th tile of image.
        """
        return self.dataset.ReadAsArray(*self.tile_coords[index])

    def __len__(self):
        return (self.dataset.RasterXSize // self.t_size) *\
               (self.dataset.RasterYSize // self.t_size)


class PairedImages:
    def __init__(self, *imagepaths: str,
                 t_size: tuple[int, int] = (100, 2),
                 pad: tuple[int, int] = (0, 0)) -> None:
        self.sources = [Image(i, t, p) for i, t, p in
                        zip(imagepaths, t_size, pad, strict=True)]
        
        assert all(map(lambda x: len(x) == len(self.sources[0]),
                       self.sources)),\
        "Image lengths match"
        assert all(map(lambda x: x.date == self.sources[0].date,
                       self.sources)),\
        "Image dates don't match."
        assert all(map(lambda x: x.tile == self.sources[0].tile,
                       self.sources)),\
        "Image tiles don't match."

        self.date = self.sources[0].date
        self.tile = self.sources[0].tile

    def __getitem__(self, index) -> tuple[tuple[Tensor], tuple[int, str, str]]:
        return tuple(tuple((source[index] for source in self.sources)),
                     # Metadata.
                     (index, self.date, self.tile))

    def __len__(self):
        return self.sources[0].__len__()


def get_msi2slstr_data(dirname: str, *,
                       t_size: tuple[int] = (100, 2),
                       pad: int = (0, 0)):
    """
    Gather data sources from msi2slstr directory.
    """
    for dpath, directories, _ in os.walk(dirname):
        for directory in directories:
            sen2filepath = os.path.join(dpath, directory, "S2MSI.tif")
            sen3filepath = os.path.join(dpath, directory, "S3SLSTR.tif")
            if not os.path.exists(sen2filepath): break
            yield PairedImages(sen2filepath, sen3filepath,
                               t_size=t_size, pad=pad)


def get_array_coords_list(
        t_size: int, sizex: int, sizey: int) -> list:
    """
    Returns a list of tile coordinates given the source image dimensions,
    tile size and array stride for sequential indexing.

    :return: A list of (xoffset, yoffset, tile_width, tile_height) values
        in terms of array elements.
    :rtype: list
    """
    xtiles = sizex // t_size
    ytiles = sizey // t_size
    return [(i % xtiles * t_size, i //
             ytiles * t_size, t_size, t_size)
            for i in range(xtiles * ytiles)]


class msi2slstr_dataset(Dataset):
    def __init__(self, dirname: str = "data", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources = [pair for pair in get_msi2slstr_data(dirname=dirname,
                                                            t_size=(100, 2))]

    def _get_source(self, index):
        return index // len(self.source[0][1]), index % len(self.source[0][1])

    def __getitem__(self, index):
        source, index = self._get_source(index)
        sen2, sen3 = self.sources[source]
        return (index, sen2.date, sen2[index], sen3[index])
    
    def __len__(self):
        return sum(map(lambda x: len(x), self.sources))
