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
import argparse
import logging
import json

import numpy as np
import torch
from torch import optim
from tqdm import trange
from gluonts.nursery.spliced_binned_pareto.distr_tcn import DistributionalTCN
from gluonts.nursery.spliced_binned_pareto.training_functions import eval_on_series
from gluonts.nursery.spliced_binned_pareto.gaussian_model import GaussianModel

from utils import _get_device_information, _save_model
import endpoint_serving

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

np.random.seed(0)
torch.manual_seed(0)


def main(args):
    logger.info(f"args: {args}")
    epochs = args['epochs']
    tcn_layers = args['tcn_layers']
    context_length = args['context_length']
    lead_time = args['lead_time']
    learning_rate = args['learning_rate']

    device = _get_device_information()

    logger.info("Loading data")
    train_ts_tensor = torch.load("/opt/ml/input/data/train/train_ts_tensor.pt")
    val_ts_tensor = torch.load("/opt/ml/input/data/validation/val_ts_tensor.pt")

    train_ts_tensor = train_ts_tensor.to(device)
    val_ts_tensor = val_ts_tensor.to(device)

    args['bins_upper_bound'] = train_ts_tensor.max()
    args['bins_lower_bound'] = train_ts_tensor.min()

    logger.info("Creating Gaussian model")

    gaussian_distr = GaussianModel(
        mu=torch.tensor(0.0),
        sigma=torch.tensor(1.0),
        device=device
    )
    gaussian_distr.to_device(device)
    output_channels = 2

    logger.info(f"bins_upper_bound: {args['bins_upper_bound']}")
    logger.info(f"bins_lower_bound: {args['bins_lower_bound']}")
    logger.info(f"output_channels: {output_channels}")

    distr_tcn = DistributionalTCN(
        in_channels=1,  # channels in the time series (univariate)
        out_channels=output_channels,  # channels in the time series (num parameters)
        kernel_size=3,
        channels=3,
        layers=tcn_layers,  # number of TCN blocks
        output_distr=gaussian_distr,
    )
    distr_tcn.to(device)
    distr_tcn = distr_tcn.float()

    optimizer = optim.Adam(
        params=distr_tcn.parameters(),
        lr=learning_rate
    )

    ts_len = train_ts_tensor.shape[2]
    val_ts_len = val_ts_tensor.shape[2]
    epoch_mod = 5

    train_losses = []
    val_losses = []
    predictions_list = []

    logger.info(f"Training model...")
    t = trange(epochs, desc=f"[gaussian]", leave=True)
    for epoch in t:
        log_loss_train = eval_on_series(
            distr_tcn,
            optimizer,
            train_ts_tensor,
            ts_len,
            context_length,
            is_train=True,
            return_predictions=False,
            lead_time=lead_time,
        )
        epoch_train_loss = np.mean(log_loss_train)
        train_losses.append(epoch_train_loss)
        logger.info(f"Train Loss: {epoch_train_loss:.3f}")

        if epoch % epoch_mod == 0:
            log_loss_val, epoch_predictions = eval_on_series(
                distr_tcn,
                optimizer,
                val_ts_tensor,
                val_ts_len,
                context_length,
                is_train=False,
                return_predictions=True,
                lead_time=lead_time,
            )
            predictions_list.append(epoch_predictions)
            epoch_val_loss = np.mean(log_loss_val)
            val_losses.append(epoch_val_loss)
            logger.info(f"Validation Loss: {epoch_val_loss:.3f}")
            t.refresh()

    _save_model(distr_tcn, args)
    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description="Distributional TCN with Binned distribution")
    parser.add_argument(
        "--learning-rate", type=float, default=0.0002, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--tcn-layers", type=int, default=4, help="Number of TCN Layers"
    )
    parser.add_argument(
        "--context-length", type=int, default=100, help="Number of timesteps used as input to model"
    )
    parser.add_argument(
        "--lead-time", type=int, default=1, help="Forecast horizon"
    )
    parser.add_argument(
        "--nbins", type=int, default=100, help="Number of bins for the SBP and Binned distribution"
    )
    parser.add_argument(
        "--percentile-tail", type=float, default=0.05, help="Percent of the distribution to be considered as a tail"
    )

    return parser.parse_args().__dict__


def model_fn():
    logger.info("calling model_fn")
    device = _get_device_information()

    config_dir = "/opt/ml/model"

    logger.info(f"config_dir contents: {os.listdir(config_dir)}")
    with open(os.path.join(config_dir, 'config.json'), 'r') as fp:
        params = json.load(fp)

    params['bins_upper_bound'] = torch.tensor(params['bins_upper_bound'], dtype=torch.double)
    params['bins_lower_bound'] = torch.tensor(params['bins_lower_bound'], dtype=torch.double)

    logger.info(f"model_dir: {config_dir}")
    logger.info(f"model_dir contents: {os.listdir(config_dir)}")

    logger.info("Creating gaussian distribution")
    gaussian_distr = GaussianModel(
        mu=torch.tensor(0.0),
        sigma=torch.tensor(1.0),
        device=device
    )
    gaussian_distr.to_device(device)
    output_distribution = gaussian_distr
    output_channels = 2

    logger.info("Creating Distributional TCN")
    distr_tcn = DistributionalTCN(
        in_channels=1,  # channels in the time series
        out_channels=output_channels,  # channels in the time series (num parameters)
        kernel_size=3,
        channels=3,  # channels inside the TCN, keep simplicity, expand for better performance
        layers=4,  # number of TCN blocks
        output_distr=output_distribution,
    )

    logger.info(f"Loading model from: {config_dir}")
    distr_tcn.load_state_dict(
        torch.load(
            os.path.join(config_dir, f"model.pth"),
            map_location=device
        )
    )

    distr_tcn.to(device)
    distr_tcn = distr_tcn.float()
    logger.info('DONE!')

    return distr_tcn, params


def input_fn(request_body, request_content_type):
    return endpoint_serving.input_fn(request_body, request_content_type)


def predict_fn(input_object, model_artifacts):
    return endpoint_serving.predict_fn(input_object, model_artifacts)


def output_fn(predictions, content_type):
    return endpoint_serving.output_fn(predictions, content_type)


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
