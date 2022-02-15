from pathlib import Path
import numpy as np


def label_frequencies(source_file):
    source_file = Path(source_file)
    with open(source_file) as file:
        header = file.readline()
        instances, features, labels = map(int, header.split())
        label_counts = np.zeros((labels,), np.int32)
        for line in file:
            # skip empty lines
            if line[0] == " ":
                continue
            label_str = line.split()[0]
            label_vec = np.fromiter(map(int, label_str.split(",")), dtype=int)
            np.add.at(label_counts, label_vec, 1)
    return instances, label_counts


def jain_propensity_model(counts: np.ndarray, n: int, *, a: float, b: float):
    c = (np.log(n) - 1) * (b + 1)**a
    eterm = c * np.exp(-a * np.log(counts + b))
    return 1.0 / (1.0 + eterm)


def estimate_by_frequency(positives: np.ndarray, total: int):
    return positives / float(total)


def estimate_beta(positives: np.ndarray, total: int):
    return (positives + 1) / float(total + 1)


def adapt_prop(p: np.ndarray, pi_tilde: np.ndarray, pi_p: np.ndarray):
    num = pi_p * (1 - pi_tilde)
    den = p * (pi_p - pi_tilde) + pi_tilde * (1.0 - pi_p)
    return num / den * p


def adjust_propensities(train_instances: int, test_instances: int, train_counts: np.ndarray, test_counts: np.ndarray, *,
                        estimator: callable, a: float, b: float):
    total_instances = train_instances + test_instances
    total_counts = train_counts + test_counts
    base_marginal = estimator(total_counts, total_instances)
    base_propensities = jain_propensity_model(total_counts, total_instances, a=a, b=b)

    train_marginal = estimator(train_counts, train_instances)
    test_marginal = estimator(test_counts, test_instances)

    train_p = adapt_prop(base_propensities, base_marginal, train_marginal)
    test_p = adapt_prop(base_propensities, base_marginal, test_marginal)
    return train_p, test_p


def prepare_propensities(train_data, test_data, *, a: float, b: float, mode: str):
    mode = mode.lower()
    n_test, freq_test = label_frequencies(test_data)
    n_train, freq_train = label_frequencies(train_data)

    if mode in ["individual", "legacy"]:
        # calculate for each subset individually
        naive_train_propensities = jain_propensity_model(freq_train, n_train, a=a, b=b)
        naive_test_propensities = jain_propensity_model(freq_test, n_train, a=a, b=b)
        return naive_train_propensities, naive_test_propensities
    elif mode == "joint":
        n_total = n_test + n_train
        freq_total = freq_test + freq_train
        joint_prop = jain_propensity_model(freq_total, n_total, a=a, b=b)
        return joint_prop, np.copy(joint_prop)
    else:

        if mode == "frequency":
            estimator = estimate_by_frequency
        elif mode == "beta":
            estimator = estimate_beta
        else:
            raise ValueError()
        train_p, test_p = adjust_propensities(n_train, n_test, freq_train, freq_test,
                                              a=0.55, b=1.5, estimator=estimator)
        return train_p, test_p
