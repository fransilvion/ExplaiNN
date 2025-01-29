import numpy as np
import pandas as pd

from functools import partial
import random
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from explainn.utils.tools import dna_one_hot


def get_data_splits(data, splits=[80, 10, 10], random_seed=1714):
    # Initialize
    data_splits = [None, None, None]
    train = splits[0] / 100.
    validation = splits[1] / 100.
    test = splits[2] / 100.

    if train == 1.:
        data_splits[0] = data
    elif validation == 1.:
        data_splits[1] = data
    elif test == 1.:
        data_splits[2] = data
    else:
        # Initialize
        p = partial(train_test_split, random_state=random_seed)
        if train == 0.:
            test_size = test
            data_splits[1], data_splits[2] = p(data, test_size=test_size)
        elif validation == 0.:
            test_size = test
            data_splits[0], data_splits[2] = p(data, test_size=test_size)
        elif test == 0.:
            test_size = validation
            data_splits[0], data_splits[1] = p(data, test_size=test_size)
        else:
            test_size = validation + test
            data_splits[0], data = p(data, test_size=test_size)
            test_size = test / (validation + test)
            data_splits[1], data_splits[2] = p(data, test_size=test_size)

    return data_splits


def get_seqs_labels_ids(tsv_file, debugging=False, rev_complement=False,
                        input_length="infer from data"):
    # Sequences / labels / ids
    df = pd.read_table(tsv_file, header=None, comment="#")
    ids = df.pop(0).values
    if input_length != "infer from data":
        seqs = [_resize_sequence(s, input_length) for s in df.pop(1).values]
    else:
        seqs = df.pop(1).values
    seqs = _dna_one_hot_many(seqs)
    labels = df.values

    # Reverse complement
    if rev_complement:
        seqs = np.append(seqs, np.array([s[::-1, ::-1] for s in seqs]), axis=0)
        labels = np.append(labels, labels, axis=0)
        ids = np.append(ids, ids, axis=0)

    # Return 1,000 sequences
    if debugging:
        return seqs[:1000], labels[:1000], ids[:1000]

    return seqs, labels, ids


def _resize_sequence(s, l):
    if len(s) < l:
        return s.center(l, "N")
    elif len(s) > l:
        start = (len(s)//2) - (l//2)
        return s[start:start+l]
    else:
        return s


def _dna_one_hot_many(seqs):
    """One hot encodes a list of sequences."""
    return(np.array([dna_one_hot(str(seq)) for seq in seqs]))


def get_data_loader(seqs, labels, batch_size=100, shuffle=False):
    # TensorDatasets
    dataset = TensorDataset(Tensor(seqs), Tensor(labels))

    # Avoid Error: Expected more than 1 value per channel when training
    batch_size = _avoid_expect_more_than_1_value_per_channel(len(dataset),
        batch_size)

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def _avoid_expect_more_than_1_value_per_channel(n, batch_size):
    if n % batch_size == 1:
        return _avoid_expect_more_than_1_value_per_channel(n, batch_size - 1)

    return batch_size



def shuffle_string(s, k=2, random_seed=1714):
    # Shuffle
    l = [s[i-k:i] for i in range(k, len(s)+k, k)]
    random.Random(random_seed).shuffle(l)

    return "".join(l)


def handle_nonstandard(non_standard, s):
    """
    """
    # 1) extract blocks of non-standard/lowercase nucleotides;
    # 2) either shuffle the nucleotides or create string of Ns; and
    # 3) put the nucleotides back
    l = list(s)
    for m in re.finditer(regexp, s):
        if non_standard == "shuffle":
            sublist = l[m.start():m.end()]
            random.shuffle(sublist)
            l[m.start():m.end()] = copy.copy(sublist)
        else:
            l[m.start():m.end()] = "N" * (m.end() - m.start())
    s = "".join(l)