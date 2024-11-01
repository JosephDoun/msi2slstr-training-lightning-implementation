import os

from osgeo.gdal import Open
from osgeo.gdal import Dataset as GDataset
from osgeo.gdal import GA_ReadOnly
from osgeo.gdal_array import LoadFile

from torch import Tensor
from torch import from_numpy
from torch import float32

from torch.utils.data import Dataset


class Image(Dataset):
    """
    Class holding image source and maps indices to tiles.
    """

    def __init__(self, imagepath: str, t_size: int, pad: int = 0) -> None:
        self.dataset: GDataset = Open(imagepath, GA_ReadOnly)
        self.imagepath = imagepath
        self.info = imagepath.split(os.sep)
        self.date = self.info[-3]
        self.tile = self.info[-2]
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
        return (from_numpy(LoadFile(self.imagepath, *self.tile_coords[index]))
                .to(float32)
                .clamp(0))

    def __len__(self):
        return (self.dataset.RasterXSize // self.t_size) *\
               (self.dataset.RasterYSize // self.t_size)


class FusionImage(Image):
    def __init__(self, sen3imagepath: str, t_size: int, pad: int = 0) -> None:
        super().__init__(sen3imagepath, t_size, pad)

        Warp("output.tif", self.dataset,
             xRes=10, yRes=10, multithread=True, callback=TermProgress)
        
        self.dataset = Open("output.tif", GA_Update)

        self.tile_coords = get_array_coords_list(t_size=t_size,
                                                 sizex=self.dataset
                                                 .RasterXSize,
                                                 sizey=self.dataset
                                                 .RasterYSize)

    def __call__(self, indices: tuple[int], x: Tensor) -> None:
        """
        Put items.
        """
        for i, sample in zip(indices, x, strict=True):
            self.dataset.WriteArray(sample.numpy(),
                                    **self.tile_coords[i][:2],
                                    band_list=range(x.size(1)))


class M2SPair:
    """
    A pair of images served together conjointly.
    """
    
    def __init__(self, sen2imagepath: str, sen3imagepath: str,
                 t_size: tuple[int, int] = (100, 2),
                 pad: tuple[int, int] = (0, 0)) -> None:
        self.sen2source = Image(sen2imagepath, t_size=t_size[0], pad=pad[0])
        self.sen3source = Image(sen3imagepath, t_size=t_size[1], pad=pad[1])
        
        assert len(self.sen2source) == len(self.sen3source),\
        "Image lengths don't match"
        assert self.sen2source.date == self.sen3source.date,\
        "Image dates don't match."
        assert self.sen2source.tile == self.sen3source.tile,\
        "Image tiles don't match."

        self.date = self.sen2source.date
        self.tile = self.sen2source.tile

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.sen2source[index], self.sen3source[index][:12]

    def __len__(self):
        """
        The length of the pair is the length of a single image.
        """
        return self.sen2source.__len__()


def get_msi2slstr_data(dirname: str, *, t_size: tuple[int] = (100, 2),
                       pad: int = (0, 0)):
    """
    Gather data sources from msi2slstr directory.
    """
    for dpath, directories, _ in os.walk(dirname):
        for directory in directories:
            sen2filepath = os.path.join(dpath, directory, "S2MSI.tif")
            sen3filepath = os.path.join(dpath, directory, "S3SLSTR.tif")
            if not os.path.exists(sen2filepath): continue
            yield M2SPair(sen2filepath, sen3filepath, t_size=t_size, pad=pad)


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
    """
    The msi2slstr dataset assembly class. Serves the dataset from a directory
    populated by the msi2slstr-datagen scripts.

    :param dirname: The root directory path of the msi2slstr data,
        defaults to `data`.
    :type dirname: str
    """
    def __init__(self, dirname: str = "data", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources: list[M2SPair] = [pair for pair in
                                       get_msi2slstr_data(dirname=dirname,
                                                          t_size=(100, 2),
                                                          pad=(0, 0))]

    def _get_source(self, index) -> tuple[int, int]:
        """
        :returns: The index of the data source to probe and the corresponding
            tile index of the selected data source.
        :rtype: tuple[int, int]
        """
        return index // len(self.sources[0]), index % len(self.sources[0])

    def __getitem__(self, index) -> tuple[tuple[Tensor, Tensor],
                                          tuple[int, str, str]]:
        """
        :returns: The two coupled patches of images and metadata related to
            the data sources to be used for logging. The metadata consist of
            the sample `index` value, the acquisition `date` and the 
            corresponding `tile` of the Sentinel-2 grid.
        :rtype: tuple[tuple[Tensor, Tensor], tuple[int, str, str]]
        """
        source, index = self._get_source(index)
        pair = self.sources[source]
        return pair[index], (pair.date, pair.tile)
    
    def __len__(self):
        """
        :returns: Sum of individual source lengths
        :rtype: int
        """
        return sum(map(lambda x: len(x), self.sources))


class Sentinel3_dataset(msi2slstr_dataset):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[Image] = [pair.sen3source for pair in
                                     get_msi2slstr_data("data",
                                                        t_size=(100, 2))]

    def __getitem__(self, index) -> Tensor:
        source, index = self._get_source(index)
        img = self.sources[source]
        return img[index][:12]
