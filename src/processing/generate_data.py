"""

 Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 SPDX-License-Identifier: MIT-0

 Permission is hereby granted, free of charge, to any person obtaining a copy of this
 software and associated documentation files (the "Software"), to deal in the Software
 without restriction, including without limitation the rights to use, copy, modify,
 merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import os
os.system("pip install gluonts==0.11.12")

import torch
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from gluonts.nursery.spliced_binned_pareto.data_functions import create_ds_asymmetric

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BASE_DIR = "/opt/ml/processing"

np.random.seed(0)

def main(args):
    logger.info(f"args: {args}")
    logger.info(f"base_dir: {BASE_DIR}")

    t_dof = [10, 10]
    noise_mult = [0.25, 0.25]
    xi = [1 / 50.0, 1 / 25.0]

    logger.info(f"t_dof: {t_dof}")
    logger.info(f"noise_mult: {noise_mult}")
    logger.info(f"xi: {xi}")

    logger.info(f"Generating data...")
    train_ts_tensor = create_ds_asymmetric(args["train_size"], t_dof, noise_mult, xi)
    val_ts_tensor = create_ds_asymmetric(args["validation_size"], t_dof, noise_mult, xi)
    test_ts_tensor = create_ds_asymmetric(args["test_size"], t_dof, noise_mult, xi)

    logger.info(f"Saving data")
    torch.save(train_ts_tensor, f"{BASE_DIR}/train/train_ts_tensor.pt")
    torch.save(val_ts_tensor, f"{BASE_DIR}/validation/val_ts_tensor.pt")
    torch.save(test_ts_tensor, f"{BASE_DIR}/test/test_ts_tensor.pt")

    logger.info(f"Generating and saving plots")
    plt.figure(figsize=(15, 5))
    plt.plot(train_ts_tensor.cpu().flatten())
    plt.title("Training dataset")
    plt.savefig(
        f"{BASE_DIR}/plots/train_data.png",
        bbox_inches='tight'
    )

    plt.figure(figsize=(15, 5))
    plt.plot(val_ts_tensor.cpu().flatten())
    plt.title("Validation dataset")
    plt.savefig(
        f"{BASE_DIR}/plots/validation_data.png",
        bbox_inches='tight'
    )

    plt.figure(figsize=(15, 5))
    plt.plot(test_ts_tensor.cpu().flatten())
    plt.title("Test dataset")
    plt.savefig(
        f"{BASE_DIR}/plots/test_data.png",
        bbox_inches='tight'
    )

    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description="Data Generation")

    parser.add_argument(
        "--train_size", type=int, default=5_000, help="Training set rows"
    )
    parser.add_argument(
        "--validation_size", type=int, default=1_000, help="Validation set rows"
    )
    parser.add_argument(
        "--test_size", type=int, default=1_000, help="Test set rows"
    )

    return parser.parse_args().__dict__


if __name__ == "__main__":
    _args = parse_args()
    main(_args)


