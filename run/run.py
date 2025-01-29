#!/usr/bin/env python
import os
import click
import json

from explainn.train.train import train_explainn
from explainn.utils.tools import pearson_loss
from explainn.models.networks import ExplaiNN
from explainn.parsers.preprocess import combine_seq_files
from explainn.parsers.parse import json2explainn

from train import run_train
from test import test_model
from interpret import interpret_results
from utils import save_data_splits

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}
@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "config_file",
    type=click.Path(exists=True, resolve_path=True),
)
def main(**args):
    # Read config file
    with open(args["config_file"]) as f:
        config = json.load(f)

    # TODO: Check that output dir exists

    # Preprocess the data
    classes = combine_seq_files(config["data"]["input_files"])
    splits = json2explainn(classes)

    # Write outputs
    save_data_splits(config["data"]["output_dir"],
        splits[0],
        splits[1],
        splits[2],
        config["data"]["prefix"],
    )
    # Update config file with output location?

    if config["options"]["store_intermediates"]:
        handle = open(os.path.join(config["data"]["output_dir"], "combined_data.json"), "wt")
        json.dump(classes, handle, indent=4, sort_keys=True)
        handle.close()

    # Train the model
    run_train(config)

    # Test the model
    test_model(config)

    # Interpret the results
    interpret_results(config)

    # Finetune the model
    # TODO: Specify this with config/arguments

    # Further interpretation
    # TODO: Specify this with config/arguments


if __name__=='__main__':
    main()