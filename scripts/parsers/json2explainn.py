#!/usr/bin/env python

import click
import copy
import json
import numpy as np
import os
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))

from utils import click_validator, get_file_handle, get_data_splits

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

def validate_click_options(context):

    # Check that the data splits add to 100
    v = sum(context.params["splits"])
    if v != 100:
        raise click.BadParameter(f"data splits do not add to 100: {v}.")

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS,
               cls=click_validator(validate_click_options))
@click.argument(
    "json_file",
    type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-n", "--non-standard",
    type=click.Choice(["skip", "shuffle", "mask"]),
    help="Skip, shuffle, or mask (i.e. convert to Ns) non-standard (i.e. non A, C, G, T) DNA, including lowercase nucleotides.",
    show_default=True
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True
)
@click.option(
    "-p", "--prefix",
    help="Output prefix.",
    type=str
)
@click.option(
    "-r", "--random-seed",
    help="Random seed.",
    type=int,
    default=1714,
    show_default=True
)
@click.option(
    "-s", "--splits",
    help="Training, validation and test data splits.",
    nargs=3,
    type=click.IntRange(0, 100),
    default=[80, 10, 10],
    show_default=True
)

def main(**args):
    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Get TSV files for ExplaiNN
    _to_ExplaiNN(args["json_file"], args["non_standard"], args["output_dir"],
                 args["prefix"], args["random_seed"], args["splits"])

def _to_ExplaiNN(json_file, non_standard=None, output_dir="./", prefix=None,
                 random_seed=1714, splits=[80, 10, 10]):

    # Initialize
    regexp = re.compile(r"[^ACGT]+")

    # Load JSON
    handle = get_file_handle(json_file, "rt")
    sequences = json.load(handle)
    handle.close()
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

    # Save TSV files
    if train is not None:
        if prefix is None:
            tsv_file = os.path.join(output_dir, "train.tsv.gz")
        else:
            tsv_file = os.path.join(output_dir, f"{prefix}.train.tsv.gz")
        train.to_csv(tsv_file, sep="\t", header=False, index=False,
                     compression="gzip")
    if validation is not None:
        if prefix is None:
            tsv_file = os.path.join(output_dir, "validation.tsv.gz")
        else:
            tsv_file = os.path.join(output_dir, f"{prefix}.validation.tsv.gz")
        validation.to_csv(tsv_file, sep="\t", header=False, index=False,
                          compression="gzip")
    if test is not None:   
        if prefix is None:
            tsv_file = os.path.join(output_dir, "test.tsv.gz")
        else:
            tsv_file = os.path.join(output_dir, f"{prefix}.test.tsv.gz")
        test.to_csv(tsv_file, sep="\t", header=False, index=False,
                    compression="gzip")

if __name__ == "__main__":
    main()