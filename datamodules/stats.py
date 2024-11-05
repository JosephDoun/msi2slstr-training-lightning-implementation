from torch import Tensor
from tqdm import tqdm

from typing import Any


class Average(Tensor):

    def __init__(self, *args, **kwargs):
        self.count = 1

    def __iadd__(self, other: Any) -> Tensor:
        self.count += 1
        return self.add_((other - self) / self.count)

    def __add__(self, other: Any) -> Tensor:
        count = self.count + 1
        other = self.add((other - self) / count)
        other.count = count
        return other

    def __mul__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __imul__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __rmul__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __div__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __idiv__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __rdiv__(self, other):
        raise NotImplementedError()

    def __matmul__(self, other: Any) -> Tensor:
        raise NotImplementedError()


def write_stats(args):
    from config import DATA_CONFIG
    from config import put_yaml_dict

    def load_stats(dataset, name):
        sample = dataset[0]
        avg = Average(sample.mean((-1, -2)))
        var = Average(sample.var((-1, -2)))
        bar = tqdm(dataset)
        for sample in bar:
            avg = avg + sample.mean((-1, -2))
            var = var + sample.var((-1, -2))
        
        stats = DATA_CONFIG.get("stats", {})
        stats.update({name: {"mean": avg.tolist(), "var": var.tolist()}})
        DATA_CONFIG.update(stats)

    if "sen3" in args:
        from datasets import sen3dataset
        load_stats(sen3dataset(210), "sen3")

    if "sen2" in args:
        from datasets import sen2dataset
        load_stats(sen2dataset(5250), "sen2")
    
    # Write.
    put_yaml_dict(DATA_CONFIG, "../config/data.yaml")


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    write_stats(sys.argv)
