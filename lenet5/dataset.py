from pathlib import Path
import numpy as np
from PIL import Image

class DataSet:
    def __init__(self, features: np.array, labels: np.array, metadata: np.array):
        self.features = features
        self.labels = labels
        self.metadata = metadata

def load_raw_data(path: Path) -> DataSet:
    features_list = []
    labels_list = []
    metadata_list  = []
    for path in path.glob('*.bmp'):
        if not path.is_file(): continue
        name = path.stem
        digit = int(name[0])
        assert digit >= 0 and digit <= 9, "Label should be a digit"
        image = Image.open(path)
        assert image.mode == '1', "Image mode should be 1 (black-and-white)"
        assert image.width == 13, "Image width should be 13"
        assert image.height == 16, "Image height should be 16"
        a = 1-np.asarray(image, dtype='B')
        features_list.append(a)
        labels_list.append(digit)
        metadata_list.append({ "orig_file": path.name })
    features = np.stack(features_list)
    labels = np.array(labels_list)
    metadata = np.array(metadata_list)
    return DataSet(features, labels, metadata)

def save_dataset(path: Path, dataset: DataSet):
    np.savez_compressed(path, features=dataset.features, labels=dataset.labels, metadata=dataset.metadata)

def main():
    root = Path(__file__).parent.parent
    raw = root / 'data' / 'raw'
    processed = root / 'data' / 'processed'
    source = processed / 'source.npz'

    print(f"Ensuring folder {processed} ...", end="")
    processed.mkdir(parents=True, exist_ok=True)
    print(" Done")

    print(f"Loading raw data from {raw} ...", end="")
    dataset = load_raw_data(raw)
    print(" Done")

    print(f"Saving {source} ...", end="")
    save_dataset(source, dataset)
    print(" Done")

    print("Data preprocessing finished")


if __name__ == '__main__':
    main()