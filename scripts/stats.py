import sys
sys.path.append(".")

from datamodules.utils import Average
from tqdm import tqdm


def write_stats(args):
    from config import DATA_CONFIG
    from config import put_yaml_dict

    def load_stats(dataset, name):
        sample = dataset[0]
        
        avg = Average(sample.mean((-1, -2)))
        var = Average(sample.std((-1, -2)))
        
        bar = tqdm(dataset)

        for sample in bar:
            avg = avg + sample.mean((-1, -2))
            var = var + sample.std((-1, -2))
        
        stats = DATA_CONFIG.get("stats", {})
        stats.update({name: {"mean": avg.tolist(), "var": var.tolist()}})
        
        DATA_CONFIG.update(stats)

    if "sen3" in args:
        from datamodules.datasets import sen3dataset
        load_stats(sen3dataset(210), "sen3")

    if "sen2" in args:
        from datamodules.datasets import sen2dataset
        load_stats(sen2dataset(5250), "sen2")
    
    # Write.
    put_yaml_dict(DATA_CONFIG, "../config/data.yaml")


if __name__ == "__main__":
    write_stats(sys.argv)
