import os

from osgeo.gdal import Open
from osgeo.gdal import Dataset as GDataset
from osgeo.gdal import GA_ReadOnly
from osgeo.gdal import GA_Update
from osgeo.gdal import Warp, Translate
from osgeo.gdal import TermProgress

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


class FusedImage(Image):
    def __init__(self, sen3imagepath: str, t_size: int, pad: int = 0) -> None:
        super().__init__(sen3imagepath, t_size, pad)

        Translate("output.tif", self.dataset, xRes=10, yRes=10, noData=0,
                  callback=TermProgress, creationOptions=["COMPRESS=ZSTD",
                                                          "PREDICTOR=2",
                                                          "BIGTIFF=YES"])
        
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
            self.dataset.WriteArray(sample.cpu().numpy(),
                                    *self.tile_coords[i][:2],
                                    band_list=range(1, x.size(1) + 1))


class M2SPair:
    """
    A pair of images served together conjointly.
    """
    
    def __init__(self, sen2image: Image, sen3image: Image) -> None:
        self.sen2source = sen2image
        self.sen3source = sen3image
        
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


def get_files(dirname: str):
    for dpath, _, _ in os.walk(dirname):
        sen2filepath = os.path.join(dpath, "S2MSI.tif")
        sen3filepath = os.path.join(dpath, "S3SLSTR.tif")
        if not os.path.exists(sen2filepath) \
            or not os.path.exists(sen3filepath): continue
        yield sen2filepath, sen3filepath


def get_msi2slstr_data(dirname: str, *, t_size: tuple[int] = (100, 2),
                       pad: int = (0, 0)):
    """
    Gather data sources from msi2slstr directory.
    """
    for sen2filepath, sen3filepath in get_files(dirname):
        yield M2SPair(Image(sen2filepath, t_size=t_size[0], pad=pad[0]),
                      Image(sen3filepath, t_size=t_size[1], pad=pad[1]))


def get_sentinel3_data(dirname: str, *, t_size: int = 2, pad: int = 0):
    """
    Gather data sources from msi2slstr directory.
    """
    for _, sen3filepath in get_files(dirname):
        yield Image(sen3filepath, t_size=t_size, pad=pad)


def get_sentinel2_data(dirname: str, *, t_size: int = 100, pad: int = 0):
    """
    Gather data sources from msi2slstr directory.
    """
    for sen2filepath, _ in get_files(dirname):
        yield Image(sen2filepath, t_size=t_size, pad=pad)


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
    def __init__(self, dirname: str = "data", t_size: tuple[int] = (100, 2),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources: list[M2SPair] = [pair for pair in
                                       get_msi2slstr_data(dirname=dirname,
                                                          t_size=t_size,
                                                          pad=(0, 0))]

    def _get_source(self, index) -> tuple[int, int]:
        """
        :returns: The index of the data source to probe and the corresponding
            tile index of the selected data source.
        :rtype: tuple[int, int]

        TODO
        Needs written tests.
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
        return len(self.sources) * len(self.sources[0])


class sen3dataset(msi2slstr_dataset):
    def __init__(self, t_size: int = 2) -> None:
        super().__init__()
        self.sources: list[Image] = [image for image in
                                     get_sentinel3_data("data",
                                                        t_size=t_size)]

    def __getitem__(self, index) -> Tensor:
        source, index = self._get_source(index)
        img = self.sources[source]
        return img[index][:12]


class sen2dataset(msi2slstr_dataset):
    def __init__(self, t_size: int = 500) -> None:
        super().__init__()
        self.sources: list[Image] = [image for image in
                                     get_sentinel2_data("data",
                                                        t_size=t_size)]

    def __getitem__(self, index) -> Tensor:
        source, index = self._get_source(index)
        img = self.sources[source]
        return img[index]


class predictor_dataset(msi2slstr_dataset):
    def __init__(self, dirname: str, *args, **kwargs):
        super().__init__(dirname, t_size=(100, 2), *args, **kwargs)
        self.output = FusionImage(self.sources[0].sen3source.imagepath,
                                  self.sources[0].sen2source.t_size)

    def __getitem__(self, index) -> tuple[tuple[int], tuple[Tensor, Tensor]]:
        return index, super().__getitem__(index)[0]

    def __call__(self, indices: tuple[int], x: Tensor) -> None:
        super().__call__()
        self.output(indices, x)


class independent_pairs(Dataset):
    """
    Dataset of same-sized independent pairs of Sentinel-2 and Sentinel-3
    images. For autoencoding.
    """
    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self._sen2 = sen2dataset(t_size=size)
        self._sen3 = sen3dataset(t_size=size)

    def __len__(self):
        return max([len(self._sen2), len(self._sen3)])

    def __getitem__(self, index) -> tuple:
        return (self._sen2[index % len(self._sen2)],
                self._sen3[index % len(self._sen3)])
