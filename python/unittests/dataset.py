import pytest
import numpy as np
from scipy import sparse
import pydismec
import pathlib


def make_dummy_data():
    feature_matrix = np.array([
        [0, 0, 0, 1.2, 0, 0, 0, 0, -0.5, 0],
        [0, 5.4, 0, 0, 0, 2.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1.0, -1.0, 1.0, 0, 0],
        [0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    positives = [[], [0, 3], [0], [], [], [3], [3], [], [], [], [1]]
    return pydismec.DataSet(sparse_features=feature_matrix, positives=positives), feature_matrix, positives


def check_dataset(dataset, feature_matrix, positives):
    assert dataset.num_features == feature_matrix.shape[1]
    assert dataset.num_examples == feature_matrix.shape[0]
    assert dataset.num_labels == len(positives)

    def binlbl(x):
        array = np.zeros(4) - 1
        for k in x:
            array[k] = 1
        return array

    for i, p in enumerate(positives):
        assert dataset.num_positives(i) == len(p)
        assert dataset.get_labels(i) == pytest.approx(binlbl(p))

    if isinstance(dataset.get_features(), sparse.csr_matrix):
        assert dataset.get_features().toarray() == pytest.approx(feature_matrix)
    else:
        assert dataset.get_features() == pytest.approx(feature_matrix)


def test_dataset():
    dataset, feature_matrix, positives = make_dummy_data()
    check_dataset(dataset, feature_matrix, positives)

    assert isinstance(dataset.get_features(), sparse.csr_matrix)

    # check that we can set the new features both as dense and sparse matrix
    feature_matrix[2, 2] = 5.0
    dataset.set_features(dense_features=feature_matrix)
    assert isinstance(dataset.get_features(), np.ndarray)
    assert dataset.get_features() == pytest.approx(feature_matrix)

    feature_matrix[2, 1] = -10
    dataset.set_features(sparse_features=sparse.csr_matrix(feature_matrix))
    assert isinstance(dataset.get_features(), sparse.csr_matrix)
    assert dataset.get_features().toarray() == pytest.approx(feature_matrix)


def test_dataset_io(tmp_path):
    dataset, feature_matrix, positives = make_dummy_data()
    d = tmp_path / "xmc.txt"
    pydismec.save_xmc(pathlib.Path(d), dataset)
    loaded = pydismec.load_xmc(d)
    check_dataset(loaded, feature_matrix, positives)

    df = tmp_path / "slice-features.npy"
    np.save(df, feature_matrix)
    dl = tmp_path / "slice-labels.txt"
    dl.write_text("4 11\n1:1 2:1\n10:1\n \n1:1 5:1 6:1\n")
    loaded = pydismec.load_slice(features=df, labels=dl)
    check_dataset(loaded, feature_matrix, positives)

    with pytest.raises(RuntimeError):
        pydismec.save_xmc("/tmp/test", loaded)
