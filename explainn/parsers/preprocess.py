import re
import copy
import random

from Bio import SeqIO
from Bio.Seq import Seq


def combine_seq_files(fasta_files, dna="lowercase", random_seed=1714,
                      subsample=0, transform=None):
    """
    Combines multiple FASTA files, each being a different class, into 
    a single JSON file

    inputs:

    outputs:

    """
    # TODO: Make this work for other files, not just FASTA files

    # Initialize 
    combined_classes = []
    all_records = {}

    if dna == "lowercase":
        regexp = re.compile(r"[^ACGT]+")
    else:
        regexp = re.compile(r"[^acgt]+")

    # For each FASTA file
    for i in range(len(fasta_files)):
        fasta_file = fasta_files[i]

        if transform:
            # For each sequence in the file
            # TODO: Move this into a function in tools.py, it gets 
            #       repeated in other places
            for record in SeqIO.parse(fasta_file, "fasta"):
                # TODO: Need to figure out what transform does...
                s = str(record.seq)

                # Skip
                if transform == "skip":
                    if re.search(regexp, s):
                        continue
                # Shuffle/Mask
                else:
                    # 1) extract blocks of nucleotides matching regexp;
                    # 2) either shuffle them or create string of Ns;
                    # and 3) put the nucleotides back
                    l = list(s)
                    for m in re.finditer(regexp, s):
                        if transform == "shuffle":
                            sublist = l[m.start():m.end()]
                            random.shuffle(sublist)
                            l[m.start():m.end()] = copy.copy(sublist)
                        else:
                            l[m.start():m.end()] = "N" * (m.end() - m.start())
                    record.seq = Seq("".join(l))
                # Add the sequence to the list
                # TODO
                    
                # Modify list to be in the correct format
                # TODO
                    
        records = SeqIO.parse(fasta_file, "fasta")
        all_records[fasta_file] = list(records)

    # Get the longest number of sequences in all of the files
    max_len = max(len(val) for val in all_records.values())

    # Create the final output dict
    for i in range(max_len):
        classes = [0]
        for j in range(len(fasta_files)):
            try:
                record = all_records[fasta_files[j]][i]
                classes.append([record.id, str(record.seq)])
            except IndexError:
                # If there are no more sequences from this file,
                # append empty lists
                classes.append([])
        combined_classes.append(classes)

    # Add the labels as the first item in the list
    combined_classes.insert(0, ["labels"] + (list(fasta_files)))
    
    return combined_classes