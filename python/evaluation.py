"""
    Computing Precision, nDCG, PSP, and PSnDCG

"""

import numpy as np

from dismec.postprocess import read_all, precision, ndcg, psp, psndcg
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred-path", default=None, type=str, required=True,
                        help="Path to the prediction file of DiSMEC")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)

    args = parser.parse_args()

    pred_path = args.pred_path
    data_file = args.data
    weight_file = args.weights

    true_mat, pred_mat, k = read_all(data_file, pred_path)
    weights = np.loadtxt(weight_file)

    p_k = precision(pred_mat, true_mat, k)
    print(
        F"P@1:      {p_k[0]:.2f},    P@3:      {p_k[2]:.2f},    P@5:      {p_k[4]:.2f}")

    ndcg_k = ndcg(pred_mat, true_mat, k)
    print(
        F"nDCG@1:   {ndcg_k[0]:.2f},    nDCG@3:   {ndcg_k[2]:.2f},    nDCG@5:   {ndcg_k[4]:.2f}")

    psp_k = psp(pred_mat, true_mat, weights, k)
    print(
        F"PSP@1:    {psp_k[0]:.2f},    PSP@3:    {psp_k[2]:.2f},    PSP@5:    {psp_k[4]:.2f}")

    psndcg_k = psndcg(pred_mat, true_mat, weights, k)
    print(
        F"PSnDCG@1: {psndcg_k[0]:.2f},    PSnDCG@3: {psndcg_k[2]:.2f},    PSnDCG@5: {psndcg_k[4]:.2f}")


if __name__ == '__main__':
    main()
