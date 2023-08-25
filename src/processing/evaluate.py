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

import argparse
import logging
import json
import pathlib
import time
import tarfile

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from gluonts.nursery.spliced_binned_pareto.gaussian_model import GaussianModel
from gluonts.nursery.spliced_binned_pareto.spliced_binned_pareto import SplicedBinnedPareto, Binned
from gluonts.nursery.spliced_binned_pareto.distr_tcn import DistributionalTCN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BASE_DIR = "/opt/ml/processing"


def _quantile_to_str(q):
    """
    Quick function to cast quantile decimal to q-prefixed string
    """
    return "q-" + str(np.round(q, 3))


def _get_device_information() -> torch.device:
    cuda_id = "0"
    if torch.cuda.is_available():
        dev = f"cuda:{cuda_id}"
    else:
        dev = "cpu"
    device = torch.device(dev)
    logger.info(f"Device is: {device}")

    return device


def main(args):
    logger.info(f"args: {args}")
    context_length = args['context_length']
    lead_time = args['lead_time']
    percentile_tail = args['percentile_tail']
    nbins = args['nbins']

    models = ["sbp", "gaussian"]

    device = _get_device_information()

    dict_storage = dict()

    train_ts_tensor = torch.load(
        "/opt/ml/processing/input/train/train_ts_tensor.pt",
        map_location='cpu'
    )

    test_ts_tensor = torch.load(
        "/opt/ml/processing/input/test/test_ts_tensor.pt",
        map_location='cpu'
    )

    bins_upper_bound = train_ts_tensor.max()
    bins_lower_bound = train_ts_tensor.min()

    for model in models:
        logger.info(f"Processing outputs of model: {model}")
        model_path = f"/opt/ml/processing/model/{model}/model.tar.gz"

        logger.info(f"Extracting files in: {model_path} to path: ./{model}")
        tar = tarfile.open(model_path)
        tar.extract('model.pth', path=f"./{model}")
        tar.extract('config.json', path=f"./{model}")
        tar.close()

        import os
        for root, dirs, files in os.walk(f"./{model}", topdown=False):
            for name in files:
                logger.info(os.path.join(root, name))
            for name in dirs:
                logger.info(os.path.join(root, name))

        if model == "sbp":
            logger.info("Creating splicedbinnedpareto distribution")
            spliced_binned_pareto_distr = SplicedBinnedPareto(
                bins_lower_bound=bins_lower_bound,
                bins_upper_bound=bins_upper_bound,
                nbins=nbins,
                percentile_gen_pareto=torch.tensor(percentile_tail),
                validate_args=None,
            )
            spliced_binned_pareto_distr.to_device(device)
            output_distribution = spliced_binned_pareto_distr
            output_channels = nbins + 4

        elif model == "binned":
            logger.info("Creating binned distribution")
            binned_distr = Binned(
                bins_lower_bound=bins_lower_bound,
                bins_upper_bound=bins_upper_bound,
                nbins=nbins,
                validate_args=None,
            )
            binned_distr.to_device(device)
            output_distribution = binned_distr
            output_channels = nbins

        elif model == "gaussian":
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
            in_channels=1,
            out_channels=output_channels,
            kernel_size=3,
            channels=3,
            layers=4,
            output_distr=output_distribution,
        )
        distr_tcn.to(device)
        distr_tcn = distr_tcn.float()

        distr_tcn.load_state_dict(
            torch.load(
                f"./{model}/model.pth",
                map_location=device
            )
        )

        dict_storage[model] = {
            'distr_tcn': distr_tcn,
            'title_method': model,
        }

        del output_distribution

    lower_tail_end = percentile_tail
    upper_tail_start = 1 - percentile_tail

    likelihoods_of_interest = np.linspace(0.001, lower_tail_end, 25)
    quantile_levels = torch.tensor(
        np.unique(
            np.round(
                np.concatenate(
                    (
                        likelihoods_of_interest,
                        np.linspace(lower_tail_end, upper_tail_start, 81),
                        1 - likelihoods_of_interest,
                    )
                ),
                3,
            )
        )
    )

    logger.info("Quantile Strings")
    quantile_strs = list(map(_quantile_to_str, quantile_levels.numpy()))
    quantile_levels = quantile_levels.to(torch.device(device))
    ts_out_tensor = test_ts_tensor.float()
    ts_len = ts_out_tensor.shape[2]
    data_out = dict(
        time=np.arange(ts_out_tensor.shape[-1] + lead_time),
        ts=np.concatenate(
            (ts_out_tensor.cpu().squeeze(), np.array([np.nan] * lead_time))
        ),
    )

    logger.info("Loop through dict_storage")
    for method_str in list(dict_storage.keys()):
        logger.info(f"Looping through {method_str}")
        distr_tcn = dict_storage[method_str]["distr_tcn"]
        title_method = dict_storage[method_str]["title_method"]

        data_out[method_str] = dict()
        for q_str in quantile_strs:
            data_out[method_str][q_str] = [np.nan] * (context_length + 2)

        # Loop through the time series
        t = trange(
            ts_len - context_length - lead_time, desc=title_method, leave=True
        )
        start = time.time()

        for i in t:

            ts_chunk = ts_out_tensor[:, :, i: i + context_length]
            distr_output = distr_tcn(ts_chunk)

            quantile_values = distr_output.icdf(quantile_levels)
            for qs, qv in zip(quantile_strs, quantile_values):
                data_out[method_str][qs].append(qv.item())

            if i == t.total - 1:
                t.set_description(
                    f"[{title_method}] runtime {int(time.time() - start)}s"
                )
                t.refresh()

        calibration_pairs = []
        logger.info("Looping through quantiles")
        for qs, ql in zip(quantile_strs, quantile_levels.cpu().numpy()):
            proportion_observations = np.array(
                list(
                    map(
                        lambda x: x[0] < x[1],
                        zip(data_out["ts"], data_out[method_str][qs]),
                    )
                )
            ).sum() / np.sum(np.isfinite(np.array(data_out[method_str][qs])))
            calibration_pairs.append([ql, proportion_observations])
        calibration_pairs = np.array(calibration_pairs)
        data_out[method_str]["calibration"] = calibration_pairs
    fig = plt.figure(figsize=[15, 5], constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    rmse_table = pd.DataFrame(
        [],
        columns=["lower_tail", "base", "upper_tail", "full_distribution"],
        index=models,
    )

    # Lower tail
    start = 0.0
    end = lower_tail_end
    indices = quantile_levels.cpu().numpy() > start
    indices *= quantile_levels.cpu().numpy() < end

    f_ax1 = fig.add_subplot(spec[0, 0])
    alpha = 0.5
    plt.plot(
        np.linspace(start, end),
        np.linspace(start, end),
        color="gray",
        alpha=alpha,
        label=None,
    )

    logger.info(f"looping through lower tail calibration pairs")
    for method_str in list(dict_storage.keys()):
        calibration_pairs = data_out[method_str]["calibration"]
        rmse = np.sqrt(np.mean(np.square(np.diff(calibration_pairs[indices, :]))))
        title_method = dict_storage[method_str]["title_method"]
        rmse_table.loc[title_method, "lower_tail"] = rmse
        plt.scatter(
            calibration_pairs[indices, 0],
            calibration_pairs[indices, 1],
            label=f"{title_method} {np.round(rmse, 3)}",
        )
    plt.legend(title="RMSE")
    plt.xlabel(f"CDF of fitted distribution")
    plt.ylabel("Empirical CDF")
    plt.title("Lower tail PP-plot")

    # Base distribution
    start = lower_tail_end
    end = upper_tail_start
    indices = quantile_levels.cpu().numpy() > start
    indices *= quantile_levels.cpu().numpy() < end

    f_ax1 = fig.add_subplot(spec[0, 1])
    alpha = 0.5
    plt.plot(
        np.linspace(start, end),
        np.linspace(start, end),
        color="gray",
        alpha=alpha,
        label=None,
    )

    logger.info(f"looping through base calibration pairs")
    for method_str in list(dict_storage.keys()):
        calibration_pairs = data_out[method_str]["calibration"]
        rmse = np.sqrt(np.mean(np.square(np.diff(calibration_pairs[indices, :]))))
        title_method = dict_storage[method_str]["title_method"]
        rmse_table.loc[title_method, "base"] = rmse
        plt.scatter(
            calibration_pairs[indices, 0],
            calibration_pairs[indices, 1],
            alpha=alpha,
            label=f"{title_method} {np.round(rmse, 2)}",
        )
    plt.legend(title="RMSE")
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.plot(
        [xlim[0], lower_tail_end, lower_tail_end],
        [lower_tail_end, lower_tail_end, ylim[0]],
        color="black",
        label=None,
    )
    plt.text(0.02, lower_tail_end, "Lower\n tail", ha="center", va="bottom")
    plt.plot(
        [upper_tail_start, upper_tail_start, xlim[1]],
        [ylim[1], upper_tail_start, upper_tail_start],
        color="black",
        label=None,
    )
    plt.text(
        (xlim[1] - upper_tail_start) / 2 + upper_tail_start,
        upper_tail_start - 0.03,
        "Upper\n tail",
        ha="center",
        va="top",
    )
    plt.xlabel(f"CDF of fitted distribution")
    plt.title("Base PP-plot")

    # Upper tail
    start = upper_tail_start
    end = 1
    indices = quantile_levels.cpu().numpy() > start
    indices *= quantile_levels.cpu().numpy() < end

    f_ax2 = fig.add_subplot(spec[0, 2])
    plt.plot(
        np.linspace(start, end),
        np.linspace(start, end),
        color="gray",
        alpha=alpha,
        label=None,
    )

    logger.info("loop through upper tail calibration pairs")
    for method_str in list(dict_storage.keys()):
        calibration_pairs = data_out[method_str]["calibration"]
        rmse = np.sqrt(np.mean(np.square(np.diff(calibration_pairs[indices, :]))))
        title_method = dict_storage[method_str]["title_method"]
        rmse_table.loc[title_method, "upper_tail"] = rmse
        plt.scatter(
            calibration_pairs[indices, 0],
            calibration_pairs[indices, 1],
            label=f"{title_method} {np.round(rmse, 3)}",
        )
    plt.legend(title="RMSE")
    plt.xlabel(f"CDF of fitted distribution")
    plt.title("Upper tail PP-plot")
    plt.savefig(
        f"{BASE_DIR}/plots/evaluation_plot.png",
        bbox_inches='tight'
    )

    for method_str in list(dict_storage.keys()):
        calibration_pairs = data_out[method_str]["calibration"]
        rmse = np.mean(np.abs(np.diff(calibration_pairs)))
        title_method = dict_storage[method_str]["title_method"]
        rmse_table.loc[title_method, "full_distribution"] = rmse

    logger.info("Saving eval")
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(rmse_table.T.to_dict()))

    logger.info(rmse_table.to_markdown())


def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--context-length", type=int, default=100, help="Number of timesteps used as input to model"
    )
    parser.add_argument(
        "--lead-time", type=int, default=1, help="Forecast horizon"
    )
    parser.add_argument(
        "--percentile-tail", type=float, default=0.05, help="Percent of the distribution to be considered as a tail"
    )
    parser.add_argument(
        "--nbins", type=int, default=100, help="Number of bins for the SBP and Binned distribution"
    )

    return parser.parse_args().__dict__


if __name__ == "__main__":
    args = parse_args()
    main(args)
