# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
# Florian Mai <florian.mai@idiap.ch>

import os
import heapq
from collections import defaultdict
from scipy.stats import distributions, lognorm, reciprocal, norm, uniform
from deepobs.analyzer.shared_utils import _determine_available_metric, _read_all_settings_folders, _load_json, aggregate_runs
import numpy as np


PRIOR_DICT = {
    'lr': distributions.expon
}

def fit_prior(performances, distrib, **params_to_fit):
    if hasattr(distrib, 'fit'):
        return distrib.fit(performances, **params_to_fit)
    else:
        raise ValueError( "distrib has to be a valid scipy.distributions object")


def read_runs(folder):
    val_metric = _determine_available_metric(folder, "valid_accuracies", default_metric="valid_losses")
    settings = _read_all_settings_folders(folder)
    path = [os.path.join(folder, sett) for sett in settings]
    jsons = [aggregate_runs(p) for p in path]
    per_hyp = [(x[val_metric]['all_final_values'][0], x['optimizer_hyperparams']) for x in jsons]
    return per_hyp, val_metric


def read_performance(dirs):
    runs = [read_runs(x)[0] for x in dirs]
    val_metric = read_runs(dirs[0])[1]
    return runs, val_metric


def post_process(runs, metric, retention=0.2, top_performance=True):
    num_k = int(len(runs) * retention)   
    if metric == 'valid_accuracies':
        if not top_performance:
            k_largest = heapq.nlargest(num_k, runs, key=lambda x: x[0])
        else:
            best_run = max(runs, key=lambda x: x[0])
    #         k_largest = runs[runs>= (1-retention)*np.max(runs)]
            k_largest = [x for x in runs if x[0] >= (1 - retention)*best_run[0]]
    else:
        if not top_performance:
            k_largest = heapq.nsmallest(num_k, runs, key=lambda x: x[0])
        else:
            best_run = min(runs, key=lambda x: x[0])
            k_largest = [x for x in runs if x[0] <= (1+ retention)* best_run[0]]
    #         k_largest = runs[runs <= (1+retention)*np.min(runs)]
    #     print(k_largest)
    return k_largest


class PriorComputer:
    def __init__(self, retention=0.2):
        self.retention = retention
        self.runs = []
    
    def add_runs_from_folder(self, folder, minimize=True):
        temp_runs, metric = read_performance(folder)
        temp_runs = post_process(temp_runs[0], metric, self.retention)
        self.runs.extend(temp_runs)
    
    def compute_prior(self, distrib, log = False):
        # if not isinstance(distrib, (distributions.rv_discrete, distributions.rv_continuous)):
        #     raise ValueError('Make the distribution from scipy distributions object')
        param_dict = defaultdict(list)
        # print(self.runs)
        for x in self.runs:
            for param in x[1]:
                p = x[1][param]
                if log:
                    p = np.log10(p)
                param_dict[param].append(p)
        self.param_dict = param_dict
        return [fit_prior(self.param_dict[x], distrib[x]) for x in self.param_dict]


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import distributions
    nums = np.random.randn(30000)
    nums -= nums.min()
    nums += 2.1
    print(nums.min())
    print(fit_prior(nums, distributions.lognorm))
