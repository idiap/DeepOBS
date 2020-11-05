#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from .shared_utils import _determine_available_metric, create_setting_analyzer_ranking
from sklearn.utils import resample


def get_hyperparameter_optimization_performance(optimizer_all_runs_paths, num_shuffle = 0, return_variance = False, obj = "max"):
    """Summarizes the performance of the optimizer.

    Args:
        optimizer_all_runs_paths (list of str): A list of all the runs (seeds) for
        hyperparameter optimization of a single optimizer and test problem.
        num_shuffle (int): Number of times to shuffle the runs to simulate multiple seeds.
        If num_shuffle < 0, compute the expected validation performance as in "Show Your Work".
    Returns:
        list of triple : A list of (test_accs, val_accs, wall_clock_times), where each
        triple denotes a run with a different seed.
        """
    
    all_runs = []
    for p in optimizer_all_runs_paths:
        all_runs.append(_read_single_run(p, return_raw_values = num_shuffle < 0))
    if num_shuffle >= 0:
        for _ in range(num_shuffle):
            p = optimizer_all_runs_paths[0]
            all_runs.append(_read_single_run(p, do_shuffle=True))
        #sys.exit()
    elif num_shuffle < 0:
        # compute expected validation performance
        val_performances = all_runs[0][0]
        exp_val = [_expected_validation_performance(val_performances, n, obj) for n in range(1, len(val_performances) + 1)]
        if not return_variance:
            exp_val = [e[0] for e in exp_val]
        all_runs[0] = (exp_val,) + all_runs[0][1:]
    return all_runs


def _expected_validation_performance(val_performances, n, obj):
    """
    Compute expected validation performance as in "Show Your Work" (but use test performance instead of val performance).
    """
    max_budget = len(val_performances)

    val_performances = np.array(val_performances)
    if obj != "max":
        val_performances = val_performances * -1.
    unique_values = np.unique(val_performances, return_counts = False)    

    cdf_smallereq = np.array([(val_performances <= unique_values[i]).sum() / max_budget for i in range(len(unique_values))])
    cdf_smaller = np.array([(val_performances < unique_values[i]).sum() / max_budget for i in range(len(unique_values))])
    pdfs = np.power(cdf_smallereq, n) - np.power(cdf_smaller, n)
    #print(unique_values)
    #print(cdf_smallereq)
    #print(cdf_smaller)
    #print(pdfs)

    exp_n = (unique_values * pdfs).sum()
    if obj != "max":
        exp_n = exp_n * -1.


    # compute variance as Var(X) = E[X^2] - E^2[X]
    exp_n_squared = (np.power(unique_values, 2) * pdfs).sum()
    exp_squared_n = np.power(exp_n, 2)
    var = exp_n_squared - exp_squared_n
    #print(n, ":", exp_n)
    return exp_n, var

def print_tunability_to_latex(results_as_dict, 
                              included_columns = ["One Shot Performance",
                                                  "Cumulative Performance Early", 
                                                  "Cumulative Performance Uniform", 
                                                  "Cumulative Performance Late",
                                                   "Peak Performance"], 
                              round_to_decimals = 1,
                              score_type = "acc"):
    name_column = list(results_as_dict.keys())
    results_by_column = {c : [] for c in included_columns}
    for opt, opt_res in results_as_dict.items():
        for metric, res in opt_res.items():
            if metric in included_columns:
                results_by_column[metric].append(res)
                
    # round results
    for m, v in results_by_column.items():
        results_by_column[m] = np.round(np.array(v) * (100 if (score_type == "acc" and m != "Avg WCT") or m == "Sharpness" else 1), round_to_decimals)
            
    data = {'Optimizer' : name_column}
    data.update(results_by_column)
    res_df = pd.DataFrame(data)
    
    rename_columns = {"One Shot Performance" : "OSP",
                      "Peak Performance" : "PP",
                    "Cumulative Performance Early" : "CPE",
                   "Cumulative Performance Late" : "CPL",
                   "Cumulative Performance Uniform" : "CPU"}
    res_df = res_df.rename(columns = rename_columns)
    included_columns = [rename_columns[c] if c in rename_columns else c for c in included_columns]
    print(res_df.sort_values(by=['Optimizer']).to_latex(columns = ["Optimizer"] + included_columns, index=False))
    
def _transform_logs_to_x_axis(all_opts_all_logs, x_axis, n_buckets, obj):
    """
    Transforms the logs, which contain the test- and validation performances as well as 
    wall-clock-times for each trial, into logs that contain the test, and validation performances
    based on wall-clock-times.
    
    To this end, the minimum wall-clock-time across all optimizers is computed, which
    determines the maximum budget to consider. The maximum budget is then bucketed into
    'n_buckets'. For each run, we then determine for each bucket the maximum performance
    that an optimizer has achieved in the run up to that time.
    """
    
    if x_axis == "trials":
        return all_opts_all_logs
    else:
        
        # first determine the shortest WCT of all optimizers
        total_wcs = []
        for opt in all_opts_all_logs:
            
            # since each entry constitues a different shuffle of the same seed, the
            # total maximum runtime is the same, so we can just look at the first only.
            ts, vs, wcs = opt[0]
            total_wcs.append(wcs[-1]) # total wcs is in the end
            
        min_wct = min(total_wcs)
        if obj == "min":
            all_ts = np.concatenate([np.array(opt[0][0]) for opt in all_opts_all_logs])
            max_ts = all_ts.max()
        bucket_size = min_wct / n_buckets
        
        buckets = np.arange(start = bucket_size, stop = min_wct + bucket_size, step = bucket_size)
        
        new_all_opts_all_logs = []
        for opt in all_opts_all_logs:
            new_runs = []
            for run in opt:
                ts, vs, wct = run
                new_ts, new_vs = [], []
                trial_counter = 0
                for bucket_counter in range(n_buckets):
                    
                    while(trial_counter < len(wct) and wct[trial_counter] <= buckets[bucket_counter]):
                        trial_counter = trial_counter + 1
                        
                    if trial_counter == 0:
                        if obj == "max":
                            best_ts = 0
                            best_vs = 0
                        else:
                            best_ts = max_ts
                            best_vs = max_ts
                    else:
                        best_ts = ts[trial_counter - 1]
                        best_vs = vs[trial_counter - 1]
                    
                    new_ts.append(best_ts)
                    new_vs.append(best_vs)
                    new_vs.append(buckets[bucket_counter])
                new_runs.append((new_ts, new_vs, buckets))
            new_all_opts_all_logs.append(new_runs)
            
        return new_all_opts_all_logs

def _compute_average_wct(all_opts_all_logs):
    wcts = []
    for opt in all_opts_all_logs:
        _, _, wct = opt[0]
        wcts.append(wct[-1] / len(wct))
    print(wcts)
    return wcts
    

def compute_tunability_metrics(all_optimizers_all_runs_path, optimizer_labels, obj = "max", num_shuffle = 0, rankings = [], x_axis = "wct"):
    """
    If num_shuffle < 0, we compute the expected validation performance from the single example provided.
    """
    all_opts_all_logs = []
    for name, o in zip(optimizer_labels,all_optimizers_all_runs_path):
        all_logs = get_hyperparameter_optimization_performance(o, num_shuffle = num_shuffle, obj = obj)
        all_opts_all_logs.append(all_logs)
        
    tunability_num_eval = _compute_alpha_tunability(all_opts_all_logs, obj = obj, tunability_x_axis="num_evals")
    tunability_wct = _compute_alpha_tunability(all_opts_all_logs, obj = obj, tunability_x_axis="wct")
    
    # compute first alpha lvl - second alpha lvl
    def compute_sharpness(lvls):
        return lvls[-1] - lvls[-2]
    sharpness = [compute_sharpness(local_lvls) for local_lvls in tunability_num_eval[0]]
    
    # compute average wct per run
    avg_wct = _compute_average_wct(all_opts_all_logs)
    
    num_evals = len(all_opts_all_logs[0][0][0])
    #num_evals = int(len(all_opts_all_logs[0][0][0]) / 20.)
    
    # if we want to use epochs or WCT on the x-axis, we need to transform the logs
    all_opts_all_logs = _transform_logs_to_x_axis(all_opts_all_logs, x_axis, num_evals, obj)
    
    ranking_perfs = []     
    for rank in rankings:
        rank_weights = np.zeros((num_evals))
        rank_weights[rank-1] = 1.
        all_performances = _compute_cumulative_performance(all_opts_all_logs, rank_weights)
        if obj == 'min':
            all_performances = min(all_performances) / all_performances
        else:
            all_performances = all_performances / max(all_performances)
        ranking_perfs.append(all_performances)
        
    # one shot performance and peak performance
    def perf_at(k):
        one_weight = np.zeros((num_evals))
        one_weight[k] = 1.
        return _compute_cumulative_performance(all_opts_all_logs, one_weight)

    one_shot_performance = perf_at(0)    
    peak_performance = perf_at(-1)
    max_budget = len(all_opts_all_logs[0][0][0])
    exp_performance_at = [perf_at(k) for k in range(max_budget)]
    
    # weight all iterations equally
    equal_weights = np.ones((num_evals)) / num_evals
    cum_performance_uniform = _compute_cumulative_performance(all_opts_all_logs, equal_weights)
    
    # weight early iterations higher
    increasing_weights = list(range(1, num_evals + 1))
    decreasing_weights = increasing_weights.copy()
    decreasing_weights.reverse()
    early_weights = np.array(decreasing_weights)
    early_weights = early_weights / early_weights.sum()
    cum_performance_early = _compute_cumulative_performance(all_opts_all_logs, early_weights)
    
    # weight late iterations higher
    late_weights = np.array(increasing_weights)
    late_weights = late_weights / late_weights.sum()
    cum_performance_late = _compute_cumulative_performance(all_opts_all_logs, late_weights)

    # weight by exponential scheme
    kappas = [0.1, 0.01, 0.001, 0.0001]
    def exponential_weighting(kappa):
        kappa_weights = np.arange(0, num_evals)
        kappa_weights = kappa_weights * (-1) * kappa
        kappa_weights = np.exp(kappa_weights)
        kappa_weights = kappa_weights / kappa_weights.sum()
        #print(kappa_weights)
        return kappa_weights
    cum_performance_exp = [_compute_cumulative_performance(all_opts_all_logs, exponential_weighting(k)) for k in kappas]

    # put results into a dict for better readability
    def _create_result_dict(i):
        result_dict = {
            "Local Alpha-Tunability / Evaluations" : tunability_num_eval[0][i],
            "Global Alpha-Tunability / Evaluations" : tunability_num_eval[1][i],
            "Local Alpha-Tunability / Wallclock-time" : tunability_wct[0][i],
            "Global Alpha-Tunability / Wallclock-time" : tunability_wct[1][i],
            "Cumulative Performance Uniform" : cum_performance_uniform[i],
            "Cumulative Performance Early" : cum_performance_early[i],
            "Cumulative Performance Late" : cum_performance_late[i],
            "Sharpness" : sharpness[i],
            "Peak Performance" : peak_performance[i],
            "One Shot Performance" : one_shot_performance[i],
            "Avg WCT" : avg_wct[i]
        }
        result_dict.update({"r="+str(rankings[j]) : ranking_perfs[j][i] for j in range(len(ranking_perfs))})
        result_dict.update({"k={}".format(k) : cum_performance_exp[j][i] for j,k in enumerate(kappas)})
        result_dict.update({"{}".format(k+1) : exp_performance_at[k][i] for k in range(max_budget)})
        return result_dict
    
    performance_dict = {}
    for i, opt_name in enumerate(optimizer_labels):
        performance_dict[opt_name] = _create_result_dict(i)
    
    
    return performance_dict

def _compute_cumulative_performance(test_performances, weights):
    """
    Compute the cumulative performance of all test scores based on the current weighting.
    
    Parameters
    -------------
    test_performances: list of list of list of floats, where the outermost list
                        contains all optimizers, the second list denotes all runs/shuffles,
                        and the last list contains all 'n' measurements.
    weights: numpy array of floats denoting the weights of all 'n' measurements.
    """
    cum_perf_opt = []
    for opt in test_performances:
        cum_perf_run = []
        for run in opt:
            ts, _, _ = run
            
            # TODO: where do we cut off?
    
            # apply weights
            ts = np.array(ts)
            ts = ts * weights
            cum_perf = ts.sum() 
            cum_perf_run.append(cum_perf)
            
        cum_perf_opt.append(sum(cum_perf_run)/len(cum_perf_run))
        
    return cum_perf_opt
    
def _read_single_run(run_path, do_shuffle = False, wct_unit = "epochs", return_raw_values = False): 
    val_metric = _determine_available_metric(run_path, "valid_accuracies", default_metric="valid_losses")
    test_metric = _determine_available_metric(run_path, "test_accuracies", default_metric="test_losses")
    
    # sort by validation metric
    val_acc_by_time = create_setting_analyzer_ranking(run_path, "timestamp", val_metric) # last is most recent
    if do_shuffle:
        val_acc_by_time = resample(val_acc_by_time)
        
    val_accs_temp = [vac.get_best_value(val_metric) for vac in val_acc_by_time]
    test_accs_temp = [tac.get_best_value(test_metric) for tac in val_acc_by_time]
    
    # summing up the epochs / wallclock times
    def sum_values(values):
        intermediate_vals = []
        total_sum = 0.
        for v in values:
            total_sum = total_sum + v
            intermediate_vals.append(total_sum)
        return intermediate_vals
    
    # wallclock time for each run results as difference to previous run
    if wct_unit == "time":
        wallclock_times = [wct.get_wallclock_time() for wct in val_acc_by_time]
        elapsed_time = sum_values(wallclock_times)
    elif wct_unit == "epochs":
        
        num_epochs = [v.get_actual_num_epochs() for v in val_acc_by_time]
        elapsed_time = sum_values(num_epochs)
    
    if "acc" in val_metric:
        test_accs, val_accs = _track_max_value(test_accs_temp, val_accs_temp)
    else:
        test_accs, val_accs = _track_min_value(test_accs_temp, val_accs_temp)

    if return_raw_values:
        return test_accs_temp, val_accs_temp, elapsed_time
    
    return test_accs, val_accs, elapsed_time

def _track_max_value(test_accs_temp, val_accs_temp):
    max_val_acc = 0
    tacc_at_max_vacc = 0
    test_accs, val_accs = [], []
    for _, (tacc, vacc) in enumerate(zip(test_accs_temp, val_accs_temp)):
        if vacc > max_val_acc:
            max_val_acc = vacc
            tacc_at_max_vacc = tacc

        test_accs.append(tacc_at_max_vacc)
        val_accs.append(max_val_acc)
        
    return test_accs, val_accs


def _track_min_value(test_losses_temp, val_losses_temp):
    min_val = float('inf')
    tval_at_min_vval = float('inf')
    tvals, vvals = [], []
    for _, (tval, vval) in enumerate(zip(test_losses_temp, val_losses_temp)):
        if vval < min_val:
            min_val = vval
            tval_at_min_vval = tval

        tvals.append(tval_at_min_vval)
        vvals.append(min_val)
        
    return tvals, vvals


def plot_hyperparam_optimization(all_opts_all_logs, labels="SGD", y_type='test', x_type='wct', score_type = "acc"):
    """
    Plots the test (y = 'test') or val accuracy (y = 'val')
    as a function of either the number of evaluations (x = 'num-evals')
    or the wallclock time (x = 'wct').

    Params:
    all_logs : list of triples of (test_accs, val_accs, wc_time)
    """

    y_max = 0.
    for i, all_logs in enumerate(all_opts_all_logs):
        
        ts, vs, wcs = zip(*all_logs)
        if y_type == 'test':
            y = np.array(ts,  dtype=np.float32)
        elif y_type == 'val':
            y = np.array(vs, dtype=np.float32)
        else:
            ValueError("Unknown type of y:"+str(y))
    
        if x_type == 'wct':
            x = np.array(wcs,  dtype=np.float32).mean(axis=0).reshape((-1))
            x = x.repeat(len(y))
        elif x_type == 'num-evals':
            x = np.array(list(range(len(wcs[0])))) + 1
        else:
            ValueError("Unknown type of x:"+str(x))
            
        df = pd.concat([pd.DataFrame({'x': x, 'y': y[i, :]}) for i in range(len(y))], axis=0)
        df.y = df.y if score_type != 'acc' else df.y * 100.
        
        ax = sns.lineplot(x='x', y='y', ci='sd', data=df, label=labels[i])
        y_max = y.max() if y.max() > y_max else y_max
        
    ax.set(xlabel='Number of Configurations', ylabel='Test Accuracy' if score_type == "acc" else "Test Loss")
    ax.set_xlim(0.9, 105)
    ax.set_xscale('log')
    y_upper_limit = 100 if score_type == "acc" else y_max
    ax.set_ylim(-5, y_upper_limit)


def plot_box_hyperparam_optim(all_opts_all_logs, labels, y_type='test', x_type='wct', score_type = "acc", y_limits=(10, 100), do_best=True, plot_box=True, do_legend=True):
    y_max = 0.

    all_opt_dict = []
    for i, all_logs in enumerate(all_opts_all_logs):
        ts, vs, wcs = zip(*all_logs)
        if y_type == 'test':
            y = np.array(ts,  dtype=np.float32)
        elif y_type == 'val':
            y = np.array(vs, dtype=np.float32)
        else:
            ValueError("Unknown type of y:"+str(y))
    
        if x_type == 'wct':
            x = np.array(wcs,  dtype=np.float32).mean(axis=0).reshape((-1))
            x = x.repeat(len(y))
        elif x_type == 'num-evals':
            x = np.array(list(range(len(wcs[0])))) + 1
        else:
            ValueError("Unknown type of x:"+str(x))
        # print('')
        
        df = pd.concat([pd.DataFrame({'x': x, 'y': y[i, :]}) for i in range(len(y))], axis=0)
        if do_best:
            df = df[df.x.isin([1, 4, 16, 64])]
        else:
            df = df[df.x.isin([1, 2, 4, 16, 32, 64, 100])]
        df['Category'] = [labels[i]] * df.shape[0]
        all_opt_dict.append(df)

    all_opt_dict = pd.concat(all_opt_dict, ignore_index=True)
    all_opt_dict.y = all_opt_dict.y if score_type != 'acc' else all_opt_dict.y * 100.
    ## Keeping the best in each family
    if do_best:
        # best_all = all_opt_dict[all_opt_dict.x == all_opt_dict.x.max()].groupby('Category').max().reset_index()
        # best_sgd = best_all.ix[best_all.loc[best_all.Category.str.contains('SGD'), 'y'].argmax()].Category
        # best_sgd = 'SGDMW'
        # all_opt_dict= all_opt_dict.loc[all_opt_dict.Category.isin(['AdamLR', 'Adam', 'Adagrad', best_sgd]), :]
        all_opt_dict = all_opt_dict.loc[all_opt_dict.Category.isin(['AdamLR', 'Adam', 'SGDMCWC', 'SGDMW', 'SGDDecay']), :]

    all_opt_groups = all_opt_dict.groupby('x')

    if plot_box:
        from matplotlib.patches import Polygon

        fig, ax = plt.subplots(1, len(all_opt_groups), sharey=True, gridspec_kw={'wspace': 0})#, figsize=(6.4, 4.8))
        # colors = ['xkcd:blue', 'orange', 'xkcd:green', 'red']
        # colors = sns.color_palette('husl', len(all_opt_dict.Category.unique()))
        colors = sns.color_palette('colorblind')
        for i, (label, x) in enumerate(all_opt_groups):
            # t = sns.boxplot(x='Category', y='y', data=x, orient='v', ax=ax[i], showfliers=False, width=0.2)
            optim_names = x.Category.unique()
            if not do_best:
                optim_names = labels
            ## make median
            print(optim_names)
            medianprops = dict(linestyle='-', linewidth=1, color='grey')
            boxprops = dict(linestyle='--', linewidth=0.1, color='white')

            temp = [x[x.Category == o].y.tolist() for o in optim_names]
            placing = 0.3 if do_best else 0.9
            thickness = 0.2 if do_best else 0.7
            bp = ax[i].boxplot(
                temp,
                positions=[placing*j for j in range(len(temp))],
                widths=thickness,
                showfliers=False,
                showcaps=False,
                medianprops=medianprops,
                boxprops=boxprops
            )
            ax[i].set_axisbelow(True)
            ax[i].set_xlabel('Budget {:>3}'.format(label), fontsize=20 if do_best else 9, fontweight='bold')

            ### Filling in the box plts.
            for j in range(len(temp)):
                box = bp['boxes'][j]
                boxX = []
                boxY = []
                for k in range(5):
                    boxX.append(box.get_xdata()[k])
                    boxY.append(box.get_ydata()[k])
                box_coords = np.column_stack([boxX, boxY])
                ax[i].add_patch(Polygon(box_coords, facecolor=colors[j]))
            
        for i, a in enumerate(ax):
            if i > 0:
                a.get_yaxis().get_label().set_visible(False)
                a.get_yaxis().set_ticks_position('none')
                a.spines['left'].set_visible(False)

            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.spines['bottom'].set_visible(False)

            a.get_xaxis().set_ticks_position('none')
            a.get_xaxis().set_ticks([])

            a.set_ylim(y_limits)
            # a.get_yaxis().set_major_locator(MaxNLocator(integer=True))

            a.yaxis.grid(True, which='major', color='.5', linestyle='-', linewidth=1, alpha=0.5)
            
            
        ax[0].set_ylabel('Test Accuracy' if score_type=='acc' else 'Test Loss', fontsize=26 if do_best else 14, fontweight='bold')
        # ax[0].set_ylabel('Budget {:>3}'.format(label), fontsize=24 if do_best else 8)
        if do_legend:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], markerfacecolor=c, color='w', label=o, marker='o') for o, c in zip(optim_names, colors)]
            print(colors)
            ax[-1].legend(handles=handles, ncol=3, bbox_to_anchor=(-1, 1 -0.05), loc='upper center', shadow=True, fancybox=True)

        return fig, ax

    else:
        fig, ax = plt.subplots(1, 1, sharey=True, gridspec_kw={'wspace': 0}, figsize=(5, 4.5))
        sns.set_palette("colorblind")

        all_opt_dict['avg'] = all_opt_dict.groupby(['x', 'Category'])['y'].transform('mean')
        all_opt_dict['var'] = all_opt_dict.groupby(['x', 'Category'])['y'].transform('std')
        uniqs, unique_inds = np.unique(all_opt_dict.x, return_index=True)
        all_opt_dict['inds'] = all_opt_dict['x'].map(dict(zip(all_opt_dict.x, unique_inds)))
        t = sns.scatterplot(x='inds', y='avg', hue='Category', size='var', data=all_opt_dict, ax=ax, alpha=0.7, legend='full')
        t.xaxis.set_major_locator(plt.FixedLocator(locs=unique_inds))
        t.set_xticklabels(uniqs)



def _compute_tunability_at_lvl(lvls, mts, max_ts, obj, tunability_x_axis = "num_eval", wcts = None):
    lvl_iter = iter(lvls)
    cur_lvl = next(lvl_iter)
    tunability_at_lvl = []
    stop = False
    for i, t in enumerate(mts):
        while (obj == "max" and t >= max_ts * cur_lvl) or (obj == "min" and t <= max_ts * (1 + (1 - cur_lvl))):
            
            if tunability_x_axis == "num_evals":
                tunability_score = float((i + 1)) / len(mts)
            elif tunability_x_axis == "wct":
                tunability_score = float(wcts[i]) / wcts[-1]
            else:
                raise ValueError("Unknown tunability x axis criterion:", tunability_x_axis)
            tunability_at_lvl.append(tunability_score)
            try:
                cur_lvl = next(lvl_iter)
            except StopIteration:
                stop = True
                break
        if stop:
            break

    return tunability_at_lvl

def _compute_alpha_tunability(all_opts_all_logs, lvls = [0.9, 0.95, 1.0], obj = "max", tunability_x_axis="num_evals"):
    # compute the maximum performance of each algorithm
    global_max = 0. if obj == "max" else float('inf')
    local_tunabilities = []
    for opt in all_opts_all_logs:
        ts, _, wcts = zip(*opt)
        mts = np.array(ts).mean(axis = 0)
        wcts = np.array(wcts).mean(axis = 0)
        
        max_ts = eval("mts."+obj)()
        if obj == "max" and max_ts > global_max:
            global_max = max_ts
        elif obj == "min" and max_ts < global_max:
            global_max = max_ts

        # compute local tunability
        tunability_at_lvl = _compute_tunability_at_lvl(lvls, mts, max_ts, obj, 
                                                       tunability_x_axis=tunability_x_axis, 
                                                       wcts=wcts)
        local_tunabilities.append(tunability_at_lvl)

    # compute global tunability
    global_tunabilities = []
    for opt in all_opts_all_logs:
        ts, _, _ = zip(*opt)
        mts = np.array(ts).mean(axis = 0)
        max_ts = global_max
        tunability_at_lvl = _compute_tunability_at_lvl(lvls, mts, max_ts, obj,
                                                       tunability_x_axis=tunability_x_axis, 
                                                       wcts=wcts)
        # fill missing elems with inf
        tunability_at_lvl.extend([float('inf') for i in range(len(lvls) - len(tunability_at_lvl))])
        global_tunabilities.append(tunability_at_lvl)

    return local_tunabilities, global_tunabilities
