from pathlib import Path

from dismec.propensity import prepare_propensities
import numpy as np
import argparse


def prepare_weights(train_data_path, test_data_path, train_weights_path, test_weights_path, *,
                    a, b, mode):
    train_prop, test_prop = prepare_propensities(train_data_path, test_data_path, a=a, b=b, mode=mode)
    np.savetxt(train_weights_path, 1.0 / train_prop)
    np.savetxt(test_weights_path, 1.0 / test_prop)
    return train_prop, test_prop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None, type=str, required=True,
                        help="The directory of the dataset which contains the train and test data files")
    parser.add_argument("--train-data", default="train.txt", type=str, required=False)
    parser.add_argument("--test-data", default="test.txt", type=str, required=False)
    parser.add_argument("--weight-files", default="weights-{}-{}.txt", type=str, required=False,
                        help="Pattern for the weights file. Needs to contain {}, which will be filled in with `train` or `test`")
    parser.add_argument("-A", default=0.55, type=float)
    parser.add_argument("-B", default=1.5, type=float)
    parser.add_argument("--mode", default="legacy", type=str)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    train_file = Path(dataset_dir) / args.train_data
    test_file = Path(dataset_dir) / args.test_data
    wf_pattern = args.weight_files
    a = args.A
    b = args.B
    mode = args.mode

    train_prop, test_prop = \
        prepare_weights(train_file, test_file, wf_pattern.format("train", "pos"), wf_pattern.format("test", "pos"),
                        a=a, b=b, mode=mode)

    # dummy files for negative weights
    np.savetxt(wf_pattern.format("train", "neg"), np.ones_like(train_prop))
    np.savetxt(wf_pattern.format("test", "neg"), np.ones_like(test_prop))


if __name__ == '__main__':
    main()
