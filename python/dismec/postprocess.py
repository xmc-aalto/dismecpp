import os
import numpy as np



def split_label_score(predicted, num_samples, top):

    top_labels = np.zeros((num_samples, top), dtype=np.int)
    top_scores = np.zeros((num_samples, top))
    for i, sample in enumerate(predicted[1:]):
        labels = np.zeros(top, dtype=np.int)
        scores = np.zeros(top)
        sample_split = sample.split(" ")[:-1] # remove the last white space!
        for j, item in enumerate(sample_split):               
            labels[j] = int(item.split(":")[0])
            scores[j] = float(item.split(":")[1])
        top_labels[i] = labels
        top_scores[i] = scores

    return top_labels, top_scores



def merge_pred(pred_path):

    pred_files = [os.path.join(pred_path,f) for f in os.listdir(pred_path) if f.endswith('.txt')]

    out_path = os.path.join(pred_path, "final_pred.txt")
    if out_path in pred_files:
        print(F"predictions are already merged in {out_path}")
        return

    print("Reading the first output file")
    with open(pred_files[0], 'r') as f:
        predicted_str = f.read().split("\n")
    predicted_str = predicted_str[:-1]

    header = predicted_str[0].split(" ")
    num_samples = int(header[0])
    top = int(header[1])

    del pred_files[0]

    top_labels, top_scores = split_label_score(predicted_str, num_samples, top)

    print("Reading the remaining output files and extracting top-k labels")
    for ind, pred_file in enumerate(pred_files):

        if (ind+1) % 10000 == 0:
            print(F"Processing the {ind+1}th output file")

        with open(pred_file, 'r') as f:
            predicted_str = f.read().split("\n")
        predicted_str = predicted_str[:-1]

        top_labels_cr, top_scores_cr = split_label_score(predicted_str, num_samples, top)

        top_labels = np.concatenate((top_labels, top_labels_cr), axis=1)
        top_scores = np.concatenate((top_scores, top_scores_cr), axis=1)

        top_ind = np.argsort(top_scores)[:,:-top-1:-1]
        top_labels = top_labels[np.tile(np.arange(num_samples)[:, np.newaxis], (1, top)), top_ind]
        top_scores = top_scores[np.tile(np.arange(num_samples)[:, np.newaxis], (1, top)), top_ind]

    final_pred = [[]] * num_samples
    for i, (labels, scores) in enumerate(zip(top_labels, top_scores)):
        str_pred = [F"{str(label)}:{str(score)}" for label, score in zip(labels, scores)]
        final_pred[i] = " ".join(str_pred)

    print(F"Saving top-k predictions to {out_path}")
    with open(out_path, "w") as f:
        f.write(" ".join(header) + "\n")
        f.write("\n".join(final_pred))

    return


def read_all(data_path, out_path):
    f = open(data_path, "r")
    header = f.readline().split(" ")
    num_samples = int(header[0])
    # num_features = int(header[1])
    # num_labels = int(header[2])

    true_mat = []
    zero_samples = []
    for i in range(num_samples):
        sample = f.readline().rstrip('\n')
        labels = sample.split(" ",1)[0]
        # remove samples with no labels
        if labels=="":
            zero_samples.append(i)
            continue
        labels = [int(label) for label in labels.split(",")]
        true_mat.append(labels)
    f.close()

    pred_mat = []
    f = open(out_path, "r")
    header = f.readline().split(" ")
    k = int(header[1])
    for i in range(num_samples):
        if i in zero_samples:
            f.readline()
            continue
        predict_sample = f.readline().rstrip('\n')
        predict_sample = predict_sample[:-1] if predict_sample[-1] == " "  else predict_sample # remove the last white space!
        pred_mat.append([int(item.split(":")[0]) for item in predict_sample.split(" ")])
    f.close()

    return true_mat, pred_mat, k



def precision(pred_mat, true_mat, k):
    assert len(pred_mat) == len(true_mat)
    correct_count = np.zeros(k, dtype=np.int)
    for pred, tr in zip(pred_mat, true_mat):
        tr = np.array(tr)
        match= np.in1d(np.array(pred), tr, assume_unique=True)
        correct_count += np.cumsum(match)

    precision = correct_count * 100.0 / (len(pred_mat) * np.arange(1, k+1))
    return precision



def ndcg(pred_mat, true_mat, k):
    assert len(pred_mat) == len(true_mat)
    correct_count = np.zeros(k)
    for pred, tr in zip(pred_mat, true_mat):
        pred = np.array(pred)
        tr = np.array(tr)
        num = np.in1d(pred, tr, assume_unique=True).astype(float)

        num[num>0] = 1.0/np.log((num>0).nonzero()[0]+2)
        
        den = np.zeros(k)
        den_size = min(tr.size, k)
        den[:den_size] = 1.0 / np.log(np.arange(1, den_size+1)+1)

        correct_count += np.cumsum(num) / np.cumsum(den)

    ndcg = correct_count * 100.0 / len(pred_mat)
    return ndcg



def psp(pred_mat, true_mat, inv_prop, k):
    assert len(pred_mat) == len(true_mat)
    num = np.zeros(k)
    den = np.zeros(k)
    for pred, tr in zip(pred_mat, true_mat):

        tr = np.array(tr)
        pred = np.array(pred)

        match = np.in1d(pred, tr, assume_unique=True).astype(float)
        match[match>0] = inv_prop[pred[match>0]]
        num += np.cumsum(match)

        inv_prop_sample = inv_prop[tr]
        inv_prop_sample = np.sort(inv_prop_sample)[::-1]

        match = np.zeros(k)
        match_size = min(tr.size, k)
        match[:match_size] = inv_prop_sample[:match_size]
        den += np.cumsum(match)

    psp = num * 100 / den
    return psp


def psndcg(pred_mat, true_mat, inv_prop, k):

    den = np.zeros(k)
    num = np.zeros(k)
    for pred, tr in zip(pred_mat, true_mat):

        tr = np.array(tr)
        pred = np.array(pred)

        match = np.in1d(pred, tr, assume_unique=True).astype(float)
        match[match>0] = inv_prop[pred[match>0]] / np.log2((match>0).nonzero()[0]+2)
        num += np.cumsum(match)

        match = np.zeros(k)
        match_size = min(tr.size, k)
        ind_large = np.argsort(inv_prop[tr])[::-1]
        temp_match = inv_prop[tr[ind_large]] / np.log2(np.arange(ind_large.size)+2) 
        match[:match_size] = temp_match[:match_size]
        den += np.cumsum(match)

    psndcg = num * 100.0 / den
    return psndcg