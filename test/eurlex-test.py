import pytest
from pathlib import Path
import subprocess
import os

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


def test_simple_run(tmp_path: Path):
    model_name = "eurlex-simple.model"
    program = [train_exe, "tfidf-eurlex-train.txt", tmp_path / model_name, "--augment-for-bias",
               "--epsilon", "0.01", "--num-labels", "256", "-q"]
    output = subprocess.check_output(program)
    compare_files(tmp_path / (model_name + ".weights-0-255"),
                  "reference-weights/eurlex-simple.model.weights-0-255")


def test_sparse_result_run(tmp_path: Path):
    model_name = "eurlex-sparse.model"
    program = [train_exe, "tfidf-eurlex-train.txt", tmp_path / model_name,
               "--augment-for-bias", "--epsilon", "0.01", "--save-sparse-txt", "--weight-culling", "0.02",
               "--num-labels", "256", "-q"]
    output = subprocess.check_output(program)
    compare_files(tmp_path / (model_name + ".weights-0-255"),
                  "reference-weights/eurlex-sparse.model.weights-0-255")
