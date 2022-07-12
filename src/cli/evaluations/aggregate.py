# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
from pathlib import Path
import click
import os
import pandas as pd
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from cli.evaluations._main import evaluations

BASELINES = ["arima", "ets", "prophet", "mqcnn"]

METRICS = ["mase", "smape", "nrmse", "nd", "ncrps"]

DATASETS = ["m3_yearly", "m3_quarterly", "m3_monthly", "m3_other", "m4_quarterly", "m4_monthly", 
    "m4_weekly", "m4_daily", "m4_hourly", "m4_yearly", "tourism_quarterly", "tourism_monthly", 
    "dominick", "weather", "hospital", "covid_deaths", "electricity", "kdd_2018", "nn5", "rossmann", "solar", "taxi", "wiki"]

    
def summarize(evaluations_path: Path, experiment: str, metric:str) -> pd.DataFrame:
    results = []
    source = Path(evaluations_path)
    source = Path.joinpath(source, experiment)
    if Path.joinpath(source, experiment + '.csv').exists():
        print("load from csv file:", Path.joinpath(source, experiment + '.csv'))
        res_df = pd.read_csv(Path.joinpath(source, experiment + '.csv'))
        index_models = list(set(res_df.model.values.tolist()))
    else:
        models = os.listdir(source)
        autogluon_models = set()
        for model in models:
            model_dir = Path.joinpath(source, model)
            if Path.is_file(model_dir):
                continue
            datasets = os.listdir(model_dir)
            for ds in datasets:
                # TODO collect dataset we need, try collect all dataset but just print we need
                if ds not in DATASETS:
                    continue
                ds_dir = Path.joinpath(model_dir, ds)
                hyperparameters = os.listdir(ds_dir)
                for hp in hyperparameters:
                    hp_dir = Path.joinpath(ds_dir, hp)
                    config = json.load(open(Path.joinpath(hp_dir, 'config.json'), 'r'))
                    performance = json.load(open(Path.joinpath(hp_dir, 'performance.json'), 'r'))
                    n = len(performance['performances'])
                    res = {}
                    if model == 'autogluon':
                        autogluom_model = model + '-' + config['hyperparameters']['presets'] + '-' + str(config['hyperparameters']['run_time'])
                        autogluon_models.add(autogluom_model)
                        res['model'] = autogluom_model
                    else:
                        res['model'] = model
                    res['dataset'] = ds
                    res.update(performance['performances'][-1]['testing'])
                    val_loss = performance['performances'][n-1]['evaluation']['val_loss'] if 'evaluation' in performance['performances'][n-1] else -1
                    res['val_loss'] = val_loss
                    res['seed'] = config['seed']
                    res['hps'] = hp
                    results.append(res)

        index_models = BASELINES + list(autogluon_models)
        res_df = pd.DataFrame(results)

    res_df = res_df.loc[res_df.groupby(['dataset', 'model', 'seed']).val_loss.idxmin()]
    return res_df.pivot_table(index='dataset', columns='model', values=metric).reindex(index_models, axis=1)


@evaluations.command(
    short_help="Collect results for baseline and autogluon runs, and report all metrics."
)
@click.option(
    "--evaluations_path",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATIONS_PATH,
    help="The directory where TSBench evaluations are stored.",
)
@click.option(
    "--experiment",
    type=click.Path(),
    required=True,
    help="The experiment name which you want to visualization.",
)
@click.option(
    "--baselines",
    type=click.Path(),
    default="baselines",
    help="Name of the experiment where the baseline results are stored.",
)
@click.option(
    "--rounding",
    default=4,
    type=int,
    help="Number of decimal places to report in the table",
)
def aggregate(evaluations_path: Path, experiment: str, baselines: str, rounding: int):
    results_per_metric = []
    for metric in METRICS:
        # Collect baseline results
        df_baselines = summarize(evaluations_path, baselines, metric)
        df_autogluon = summarize(evaluations_path, experiment, metric)
        df_combined = df_baselines.join(df_autogluon)
        # Add the current metric name as a top-level index
        df_with_metric = pd.concat({metric: df_combined}, names=["metric"])
        results_per_metric.append(df_with_metric)
    # Only report datasets (rows) for which we have autogluon results 
    results_all = pd.concat(results_per_metric, axis=0).dropna()
    results_all = results_all.round(rounding)
    save_path = Path.joinpath(evaluations_path, experiment, f"{experiment}_table.csv")
    results_all.to_csv(save_path)
    print(results_all)
