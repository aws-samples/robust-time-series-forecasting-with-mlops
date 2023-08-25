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


import logging
import json
import torch

from utils import _get_device_information

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def input_fn(request_body, request_content_type):
    logger.info("Running input_fn")
    device = _get_device_information()
    data = json.loads(request_body)["inputs"]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data


def predict_fn(input_object, model_artifacts):
    from gluonts.nursery.spliced_binned_pareto.training_functions import (
        eval_on_series,
    )
    distr_tcn = model_artifacts[0]
    params = model_artifacts[1]

    logger.info("Running predict_fn")
    val_ts_tensor = input_object

    from torch import optim
    optimizer = optim.Adam(params=distr_tcn.parameters(), lr=params['learning_rate'])
    val_ts_len = val_ts_tensor.shape[2]
    context_length = 100
    lead_time = 1
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
    return epoch_predictions


def output_fn(predictions, content_type):
    logger.info("Running output_fn")
    forecast_dict = dict()
    for key in predictions.keys():
        list_ = []
        for i in predictions[key]:
            list_.append(float(i[0].detach().numpy()))
        forecast_dict[key] = list_

    return forecast_dict
