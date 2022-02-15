"""
    Converting BoW to TF-IDF and normalizing the features

"""

from dismec.preprocess import tfif_calculator
from pathlib import Path
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bow-train-path", default=None, type=str, required=True,
                        help="path to training data with BoW features")
    parser.add_argument("--bow-test-path", default=None, type=str, required=True,
                        help="directory for saving TF-IDF features of training data")
    parser.add_argument("--tfidf-train-path", default=None, type=str, required=False,
                        help="path to test data with BoW features")
    parser.add_argument("--tfidf-test-path", default=None, type=str, required=False,
                        help="directory for saving TF-IDF features of test data")
    
    args = parser.parse_args()
    
    bow_train_path = Path(args.bow_train_path)
    tfidf_train_path = args.tfidf_train_path or bow_train_path.with_name("tfidf-" + bow_train_path.name)

    bow_test_path = Path(args.bow_test_path)
    tfidf_test_path = args.tfidf_test_path or bow_test_path.with_name("tfidf-" + bow_test_path.name)

    print("Calculating TF-IDF for training data")
    tfif_calculator(bow_train_path, tfidf_train_path)

    print("Calculating TF-IDF for test data")
    tfif_calculator(bow_test_path, tfidf_test_path)


if __name__ == '__main__':
    main()

