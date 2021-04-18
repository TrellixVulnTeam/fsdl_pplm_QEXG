import argparse
import os
import pandas as pd
import sqlite3

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PPLM.run_pplm import train_discriminator


def run_all(data_dir):


    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="directory to write data to")

    args = parser.parse_args()
    print(f"Args: {vars(args)}")

    # data_dir = 'data'
    # resource_dir = 'resources'
    run_all(**(vars(args)))
