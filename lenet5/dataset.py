from pathlib import Path
import numpy as np
import torch
from PIL import Image

preprocessing_seed = 600789589

class DataSet:
    def __init__(self, features: np.array | torch.Tensor,
                 labels: np.array | torch.Tensor,
                 metadata: list[dict[str, any]]):
        self.features = features
        self.labels = labels
        self.metadata = metadata
    
    def torch32(self):
        return DataSet(torch.from_numpy(self.features).float()[:,None,:,:],
                       torch.from_numpy(self.labels).byte(),
                       self.metadata)

def load_raw_data(path: Path) -> DataSet:
    features_list = []
    labels_list = []
    metadata_list = []
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
        metadata_list.append({ 'orig_file': path.name })
    features = np.stack(features_list)
    labels = np.array(labels_list, dtype='i4')
    return DataSet(features, labels, metadata_list)

def save_dataset(path: Path, dataset: DataSet):
    np.savez_compressed(path, features=dataset.features, labels=dataset.labels, metadata=dataset.metadata)

def validate_source_dataset(dataset: DataSet):
    assert dataset.features.shape == (120, 16, 13)
    assert dataset.features.dtype == 'B'
    assert (np.unique(dataset.features) == np.array([0, 1], dtype='B')).all()

    assert dataset.labels.shape == (120,)
    assert dataset.labels.dtype == 'i4'
    (labels, counts) = np.unique(dataset.labels, return_counts=True)
    assert (labels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4')).all()
    assert (counts == np.array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12], dtype='i4')).all()

    assert type(dataset.metadata) is list
    assert len(dataset.metadata) == 120
    assert all(meta['orig_file'].lower().endswith('.bmp') for meta in dataset.metadata)

def augment_dataset(dataset: DataSet) -> DataSet:
    features = np.ndarray((480, 16, 16), dtype='B')
    labels = np.ndarray((480,), dtype='i4')
    metadata = [None] * 480
    for i in range(120):
        for j in range(4):
            features[i*4+j, :, :] = np.pad(dataset.features[i], ((0,0),(j,3-j)), mode='constant', constant_values=0)
            labels[i*4+j] = dataset.labels[i]
            metadata[i*4+j] = dataset.metadata[i].copy()
            metadata[i*4+j]['padding'] = j
    return DataSet(features, labels, metadata)

def validate_augmented_dataset(dataset: DataSet):
    assert dataset.features.shape == (480, 16, 16)
    assert dataset.features.dtype == 'B'
    assert (np.unique(dataset.features) == np.array([0, 1], dtype='B')).all()

    assert dataset.labels.shape == (480,)
    assert dataset.labels.dtype == 'i4'
    (labels, counts) = np.unique(dataset.labels, return_counts=True)
    assert (labels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4')).all()
    assert (counts == np.array([48, 48, 48, 48, 48, 48, 48, 48, 48, 48], dtype='i4')).all()

    assert type(dataset.metadata) is list
    assert len(dataset.metadata) == 480
    assert all(meta['orig_file'].lower().endswith('.bmp') for meta in dataset.metadata)
    assert all('padding' in meta and meta['padding'] in [0,1,2,3] for meta in dataset.metadata)

def split_augmented_dataset(dataset: DataSet) -> DataSet:
    per_digit = { digit: np.nonzero(dataset.labels == digit)[0] for digit in range(10) }
    for digit in range(10):
        assert len(per_digit[digit]) == 48, "Each digit should have 48 samples"
        np.random.shuffle(per_digit[digit])
    
    train_per_digit = { digit: per_digit[digit][:32] for digit in range(10) }
    test_per_digit = { digit: per_digit[digit][32:] for digit in range(10) }
        
    train = []
    for digit in range(10):
        train.extend(train_per_digit[digit])
    train = [train[i] for i in np.random.permutation(len(train))]

    test = []
    for digit in range(10):
        test.extend(test_per_digit[digit])
    test = [test[i] for i in np.random.permutation(len(test))]

    return (DataSet(dataset.features[train], dataset.labels[train], [dataset.metadata[i] for i in train]),
            DataSet(dataset.features[test], dataset.labels[test], [dataset.metadata[i] for i in test]))

def validate_train_dataset(dataset: DataSet):
    assert dataset.features.shape == (320, 16, 16)
    assert dataset.features.dtype == 'B'
    assert (np.unique(dataset.features) == np.array([0, 1], dtype='B')).all()

    assert dataset.labels.shape == (320,)
    assert dataset.labels.dtype == 'i4'
    (labels, counts) = np.unique(dataset.labels, return_counts=True)
    assert (labels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4')).all()
    assert (counts == np.array([32, 32, 32, 32, 32, 32, 32, 32, 32, 32], dtype='i4')).all()

    assert type(dataset.metadata) is list
    assert len(dataset.metadata) == 320
    assert all(meta['orig_file'].lower().endswith('.bmp') for meta in dataset.metadata)
    assert all('padding' in meta and meta['padding'] in [0,1,2,3] for meta in dataset.metadata)

def validate_test_dataset(dataset: DataSet):
    assert dataset.features.shape == (160, 16, 16)
    assert dataset.features.dtype == 'B'
    assert (np.unique(dataset.features) == np.array([0, 1], dtype='B')).all()

    assert dataset.labels.shape == (160,)
    assert dataset.labels.dtype == 'i4'
    (labels, counts) = np.unique(dataset.labels, return_counts=True)
    assert (labels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4')).all()
    assert (counts == np.array([16, 16, 16, 16, 16, 16, 16, 16, 16, 16], dtype='i4')).all()

    assert type(dataset.metadata) is list
    assert len(dataset.metadata) == 160
    assert all(meta['orig_file'].lower().endswith('.bmp') for meta in dataset.metadata)
    assert all('padding' in meta and meta['padding'] in [0,1,2,3] for meta in dataset.metadata)

def load_train_dataset():
    root = Path(__file__).parent.parent
    processed = (root / 'data' / 'processed').resolve()
    train = processed / 'train.npz'
    with np.load(train, allow_pickle=True) as data:
        return DataSet(data['features'], data['labels'], data['metadata'].tolist())

def load_test_dataset():
    root = Path(__file__).parent.parent
    processed = (root / 'data' / 'processed').resolve()
    test = processed / 'test.npz'
    with np.load(test, allow_pickle=True) as data:
        return DataSet(data['features'], data['labels'], data['metadata'].tolist())

def main():
    np.random.seed(600789589)

    root = Path(__file__).parent.parent
    raw = (root / 'data' / 'raw').resolve()
    processed = (root / 'data' / 'processed').resolve()
    source = processed / 'source.npz'
    augmented = processed / 'augmented.npz'
    train = processed / 'train.npz'
    test = processed / 'test.npz'

    print(f"Ensuring folder {processed} ... ", flush=True, end="")
    processed.mkdir(parents=True, exist_ok=True)
    print("Done", flush=True)

    print(f"Loading raw data from {raw} ... ", flush=True, end="")
    source_dataset = load_raw_data(raw)
    print("Done", flush=True)

    print(f"Validating source dataset ... ", flush=True, end="")
    validate_source_dataset(source_dataset)
    print("Done", flush=True)

    print(f"Saving {source} ... ", flush=True, end="")
    save_dataset(source, source_dataset)
    print("Done", flush=True)

    print(f"Augmenting dataset ... ", flush=True, end="")
    augmented_dataset = augment_dataset(source_dataset)
    print("Done", flush=True)

    print(f"Validating augmented dataset ... ", flush=True, end="")
    validate_augmented_dataset(augmented_dataset)
    print("Done", flush=True)

    print(f"Saving {augmented} ... ", flush=True, end="")
    save_dataset(augmented, augmented_dataset)
    print("Done", flush=True)

    print(f"Splitting augmented dataset into train and test ... ", flush=True, end="")
    (train_dataset, test_dataset) = split_augmented_dataset(augmented_dataset)
    print("Done", flush=True)  

    print(f"Validating train dataset ... ", flush=True, end="")
    validate_train_dataset(train_dataset)
    print("Done", flush=True)

    print(f"Validating test dataset ... ", flush=True, end="")
    validate_test_dataset(test_dataset)
    print("Done", flush=True)

    print(f"Saving {train} and {test} ... ", flush=True, end="")
    save_dataset(train, train_dataset)
    save_dataset(test, test_dataset)
    print("Done", flush=True)

    print("Data preprocessing finished")

    return { 
        'source_dataset': source_dataset,
        'augmented_dataset': augmented_dataset,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        }


if __name__ == '__main__':
    main()