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
import json
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from gluonts.nursery.spliced_binned_pareto.distr_tcn import DistributionalTCN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _get_device_information() -> torch.device:
    cuda_id = "0"
    if torch.cuda.is_available():
        dev = f"cuda:{cuda_id}"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


def _save_model(model: DistributionalTCN, params: dict) -> None:
    params['bins_upper_bound'] = params['bins_upper_bound'].item()
    params['bins_lower_bound'] = params['bins_lower_bound'].item()

    model_path = os.path.join(os.environ["SM_MODEL_DIR"])

    logger.info(f"Saving the model to path: {model_path}")
    Path(model_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(model_path, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(params, f)

    logger.info("Model saved!")


def plot_sbp_distribution(prediction_df, historical_list, empty_list, last_value, connect_length, line_width):

    def _connect_points(plot_df, column_name, length=connect_length, last_value=last_value):
        return np.ravel(np.linspace(last_value, plot_df.loc[0, column_name], num=length)).tolist()

    plot_df = pd.DataFrame(
        {
            "historical": historical_list + [None] * connect_length,
            "1st_percentile": empty_list + last_value + _connect_points(prediction_df, 'low_lower'),
            "5th_percentile": empty_list + last_value + _connect_points(prediction_df, 'lower'),
            "median": empty_list + last_value + _connect_points(prediction_df, 'median'),
            "95th_percentile": empty_list + last_value + _connect_points(prediction_df, 'upper'),
            "99th_percentile": empty_list + last_value + _connect_points(prediction_df, 'up_upper'),
        }
    )

    prediction_df.rename(
        columns={
            'low_lower': '1st_percentile',
            'lower': '5th_percentile',
            'upper': '95th_percentile',
            'up_upper': '99th_percentile',
        },
        inplace=True
    )

    for i in range(line_width):
        plot_df = pd.concat([plot_df, prediction_df], axis=0)

    plot_df = plot_df.reset_index(drop=True)

    line_width_x = list(range(len(plot_df) - line_width, len(plot_df)))
    con_line_width_x = list(range(len(plot_df) - connect_length - line_width, len(plot_df) - line_width + 1))
    color_dict = {
        '1st_percentile': '#FF1919',
        '99th_percentile': '#FF1919',
        '5th_percentile': '#248f24',
        '95th_percentile': '#248f24',
        'median': '#808080',
    }
    fill_color_inner = '#FFCCCC'
    fill_color_outer = '#D6F5D6'

    plot_df.plot(
        color=[color_dict.get(col, '000000') for col in plot_df.columns],
        figsize=(16, 8)
    )

    plt.fill_between(
        x=line_width_x,
        y1=plot_df['1st_percentile'][-line_width:],
        y2=plot_df['99th_percentile'][-line_width:],
        color=fill_color_inner,
        alpha=0.5
    )

    plt.fill_between(
        x=con_line_width_x,
        y1=_connect_points(prediction_df, column_name='1st_percentile', length=connect_length+1),
        y2=_connect_points(prediction_df, column_name='99th_percentile', length=connect_length+1),
        color=fill_color_inner,
        alpha=0.5
    )

    plt.fill_between(
        x=line_width_x,
        y1=plot_df['5th_percentile'][-line_width:],
        y2=plot_df['95th_percentile'][-line_width:],
        color=fill_color_outer,
        alpha=0.9
    )

    plt.fill_between(
        x=con_line_width_x,
        y1=_connect_points(prediction_df, column_name='5th_percentile', length=connect_length+1),
        y2=_connect_points(prediction_df, column_name='95th_percentile', length=connect_length+1),
        color=fill_color_outer,
        alpha=0.9
    )

    plt.scatter(len(historical_list), historical_list[-1], facecolors='none', edgecolors='black', s=50)
    plt.title("Single Step Forecast")

