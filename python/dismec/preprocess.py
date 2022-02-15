import numpy as np
from tqdm import tqdm
import time

try:
    import pydismec
    from scipy import sparse, linalg
    _HAS_PYDISMEC = True
except ImportError:
    print("Could not import pydismec, falling back to (slow) python implementation")
    _HAS_PYDISMEC = False


def _tfidf_calc_cpp(in_file: str, out_file: str):
    data = pydismec.load_xmc(in_file)
    features = data.get_features()

    assert sparse.isspmatrix_csr(features)

    start = time.time()
    print("Computing idf")
    feature_ids = features.indices
    features_frequencies = np.bincount(feature_ids, minlength=features.shape[1])
    with np.errstate(divide='ignore'):
        idf = np.log(data.num_examples / features_frequencies)

    print("Computing tf")
    features.data = np.round(features.data)
    features.eliminate_zeros()
    features.data = 1.0 + np.log(features.data)

    print("Multiply tf and idf")
    multipliers = idf[features.indices]
    features.data *= multipliers

    print("Normalize feature vectors")
    norm_calc = sparse.csr_matrix(features).power(2)
    norm = np.sqrt(np.asarray(np.sum(norm_calc, axis=1)))
    for i, (b, e) in enumerate(zip(features.indptr[:-1], features.indptr[1:])):
        features.data[b:e] /= norm[i]

    print(f"Calculations took {time.time() - start:.2} seconds")

    print("Update dataset")
    start = time.time()
    data.set_features(features)
    pydismec.save_xmc(out_file, data, 4)
    print(f"Saving took {time.time() - start:.2} seconds")


def _get_frequencies_py(in_file: str):
    in_file = open(in_file, "r")

    header = in_file.readline().rstrip('\n').split(" ")
    num_docs = int(header[0])
    num_features = int(header[1])

    features_frequency = np.zeros(num_features)
    for _ in tqdm(range(num_docs), desc='Computing frequencies'):
        sample = in_file.readline().split(" ", 1)
        features = [int(item.split(":")[0]) for item in sample[1].split(" ")]
        features_frequency[features] += 1
    return features_frequency


def _parse_features(features):
    all_features = features.split(" ")
    features_key = np.zeros(len(all_features), dtype=np.int)
    features_value = np.zeros(len(all_features), dtype=np.float)

    for i, feature in enumerate(all_features):
        split_feature = feature.split(":")
        features_key[i] = split_feature[0]
        features_value[i] = round(float(split_feature[1]))
    return features_key, features_value


def _parse_features(features):
    features_key = np.array([], dtype=np.int)
    features_value = np.array([], dtype=np.float)
    for feature in features.split(" "):
        split_feature = feature.split(":")
        features_key = np.append(features_key, int(split_feature[0]))
        features_value = np.append(
            features_value, float(round(float(split_feature[1]))))
    return features_key, features_value


def _tfidf_calc_py(in_file, out_file, num_docs: int):
    features_frequency = _get_frequencies_py(in_file.name)

    # pre-calculate idf array
    with np.errstate(divide='ignore'):
        idf = np.log(num_docs / features_frequency)

    for _ in tqdm(range(num_docs), desc='Computing TF-IDF and writing new features'):
        #  splitting features keys and values in each instance
        sample = in_file.readline().split(" ", 1)
        features_key, features_value = _parse_features(sample[1])

        # computing new weights based on tfidf and normalizing the weights
        pos_features = features_value >= 1
        features_value = features_value[pos_features]
        features_key = features_key[pos_features]
        tf = 1.0 + np.log(features_value)
        weights = tf * idf[features_key]
        weights /= np.linalg.norm(weights)

        # write new features to file
        # note that using a generator expression (instead of a temporary list) provides a significant speedup
        temp_sample = " ".join(f"{item[0]}:{item[1]:.4f}" for item in zip(features_key, weights))
        temp_sample = f"{sample[0]} {temp_sample}\n"

        out_file.write(temp_sample)


if _HAS_PYDISMEC:
    def tfif_calculator(bow_path, tfidf_path):
        return _tfidf_calc_cpp(str(bow_path), str(tfidf_path))
else:
    def tfif_calculator(bow_path, tfidf_path):
        with open(tfidf_path, "w") as out_file, open(bow_path, "r") as in_file:
            header = in_file.readline()
            num_docs = int(header.split(" ")[0])
            out_file.write(header)
            _tfidf_calc_py(in_file, out_file, num_docs)
