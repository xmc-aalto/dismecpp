import numpy as np
import pytest
from pathlib import Path
import subprocess
import os
import json

# figure out where to get the training binary
BIN_DIR = Path(os.getenv("DISMEC_BIN_DIR", "../cmake-build-release/bin"))
train_exe = BIN_DIR / "train"

assert train_exe.exists()


def compare_files(a, b):
    a = Path(a)
    b = Path(b)
    assert a.exists()
    assert b.exists()

    a_text = a.read_text()
    b_text = b.read_text()
    assert len(a_text) == len(b_text)

    check = a_text == b_text
    assert check, f"{a} != {b}"


def compare_model_meta(a, b):
    a = Path(a)
    b = Path(b)
    assert a.exists()
    assert b.exists()

    a_data = json.loads(a.read_text())
    b_data = json.loads(b.read_text())

    a_data["date"] = "$DATE"
    b_data["date"] = "$DATE"

    assert a_data == b_data


def checkout_reference(path: Path):
    clone = ["git", "clone", "https://version.aalto.fi/gitlab/xmc/dicmecpp-test-data.git", path, "--depth", "1"]
    subprocess.check_call(clone)


def test_dense_result_run(tmp_path: Path):
    """
    Tests that a simple run configuration, with dense model output file,
    produces the same result as the given reference.
    """
    model_name = "eurlex-simple.model"
    program = [train_exe, "tfidf-eurlex-train.txt", tmp_path / model_name, "--augment-for-bias",
               "--epsilon", "0.01", "--num-labels", "128", "-q"]
    output = subprocess.check_output(program)
    ref_path = tmp_path / "data"
    checkout_reference(ref_path)
    ref_path = ref_path / "reference-weights"

    compare_model_meta(tmp_path / model_name, ref_path / model_name)

    compare_files(tmp_path / (model_name + ".weights-0-127"),
                  ref_path / (model_name + ".weights-0-127"))


def test_sparse_result_run(tmp_path: Path):
    """
    Tests that a simple run configuration, with sparse model output file,
    produces the same result as the given reference.
    """
    model_name = "eurlex-sparse.model"
    program = [train_exe, "tfidf-eurlex-train.txt", tmp_path / model_name,
               "--augment-for-bias", "--epsilon", "0.01", "--save-sparse-txt", "--weight-culling", "0.02",
               "--num-labels", "128", "-q"]
    output = subprocess.check_output(program)

    ref_path = tmp_path / "data"
    checkout_reference(ref_path)
    ref_path = ref_path / "reference-weights"

    compare_model_meta(tmp_path / model_name, ref_path / model_name)

    compare_files(tmp_path / (model_name + ".weights-0-127"),
                  ref_path / (model_name + ".weights-0-127"))


def test_npy_result_run(tmp_path: Path):
    """
    Tests that a simple run configuration, with dense npy model output file,
    produces the same result as the given reference. This run has a slightly different epsilon
    value, and does not augment with a bias vector.
    """
    # TODO weight-culling is currently not supported, even though it would make sense!
    model_name = "eurlex-npy.model"
    program = [train_exe, "tfidf-eurlex-train.txt", tmp_path / model_name,
               "--epsilon", "0.005", "--save-dense-npy", #"--weight-culling", "0.02",
               "--num-labels", "100", "-q"]
    output = subprocess.check_output(program)

    ref_path = tmp_path / "data"
    checkout_reference(ref_path)
    ref_path = ref_path / "reference-weights"

    compare_model_meta(tmp_path / model_name, ref_path / model_name)

    tst_data = np.load(tmp_path / (model_name + ".weights-0-99"))
    ref_data = np.load(ref_path / (model_name + ".weights-0-99"))
    assert pytest.approx(tst_data) == ref_data
