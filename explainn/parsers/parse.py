import copy
import json
import numpy as np
import os
import pandas as pd
import re
import random
from sklearn.preprocessing import OneHotEncoder

from explainn.parsers.tools import get_data_splits


def json2explainn(sequences, non_standard=None, random_seed=1714, 
    splits=[80, 10, 10]):
    """
    Inputs:
        sequences       Dictionary containing sequences
        non_standard    
    """

    # Initialize
    regexp = re.compile(r"[^ACGT]+")
    sequences.pop(0)

    # Ys
    enc = OneHotEncoder()
    arr = np.array(list(range(len(sequences[0]) - 1))).reshape(-1, 1)
    enc.fit(arr)
    ys = enc.transform(arr).toarray().tolist()

    # Get DataFrame
    data = []
    for i in range(len(sequences)):
        for j in range(1, len(sequences[i])):
            # Check the sequence is not empty
            if not sequences[i][j]:
                continue
            
            s = sequences[i][j][1]

            # Skip non-standard/lowercase
            if non_standard == "skip":
                if re.search(regexp, s):
                    continue
            # Shuffle/mask non-standard/lowercase
            elif non_standard is not None:
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
            data.append([sequences[i][j][0], s] + ys[j - 1])
    df = pd.DataFrame(data, columns=list(range(len(data[0]))))
    df = df.groupby(1).max().reset_index()
    df = df.reindex(sorted(df.columns), axis=1)

    # Get data splits
    train, validation, test = get_data_splits(df, splits, random_seed)

    return (train, validation, test)
