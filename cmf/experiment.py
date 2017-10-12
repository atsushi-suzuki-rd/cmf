import numpy as np
import pandas as pd
from sklearn import linear_model
from cmf.cmfpn import CMFPN
from cmf.cnmf import CNMF
from cmf.utils import rmse, generate_data, completion_experiment
from cmf.wrapper import TimeSeriesLinearRegression, MatrixCompletionRegression, RPCA

def hyperparameter_selection_of_cmf(X, interval, grid_section_list, l1_weight_list, verbose=0):
    cmfpn_dict = {}
    for l1_weight in l1_weight_list:
        cmfpn_arg_dict = dict(
            convolution_width=100 // interval,
            n_components=2,
            response_l2_weight=1.0000,
            response_l1_weight=0.0000,
            signal_l1_weight=l1_weight,
            signal_l2_weight=0.0,
            convergence_threshold=0.0000001,
            loop_max=200,
            fit_accelerator_max=0.0,
            transform_accelerator_max=0.0,
            verbose=0,
            initialization='smooth_svd',
            noise_scale=0.01,
            noise_threshold=0.00)
        cmfpn_dict[str(l1_weight)] = CMFPN(**cmfpn_arg_dict)
    cmfpn_loss_dict = completion_experiment(X, section_list=grid_section_list, completer_dict=cmfpn_dict, interval=interval,
                                            verbose=0)
    if verbose >= 2:
        cmfpn_mean_rmse = {weight: result.mean() for weight, result in cmfpn_loss_dict['rmse'].items()}
        print('mean RMSE of CMF on validation data:')
        print(cmfpn_mean_rmse)
        cmfpn_mean_md = {weight: result.mean() for weight, result in cmfpn_loss_dict['md'].items()}
        print('mean absolute deviation of CMF on validation data:')
        print(cmfpn_mean_md)
    cmfpn_l1_weight = float(min(cmfpn_mean_rmse.items(), key=lambda x:x[1])[0])
    return cmfpn_l1_weight


def hyperparameter_selection_of_pca(X, interval, grid_section_list, method_name, verbose=0):
    n_components_list = np.arange(1, 17)

    pca_dict = {str(n_components): RPCA(n_components, method_name=method_name, scale_option='vector') for n_components in
                 n_components_list}

    pca_loss_dict = completion_experiment(X, section_list=grid_section_list, completer_dict=pca_dict,
                                           interval=interval, verbose=0)
    pca_mean_rmse = {weight: result.mean() for weight, result in pca_loss_dict['rmse'].items()}
    pca_mean_md = {weight: result.mean() for weight, result in pca_loss_dict['md'].items()}
    if verbose >= 2:
        print('mean RMSE of {} on validation data:'.format(method_name))
        print(pca_mean_rmse)
        print('mean absolute deviation of {} on validation data:'.format(method_name))
        print(pca_mean_md)
    pca_n_components = int(min(pca_mean_rmse.items(), key=lambda x: x[1])[0])
    return pca_n_components


def hyperparameter_selection(loader, interval=10, data_length=2500, grid_start_list=np.arange(3500, 41000, 5000), l1_weight_list=np.arange(2, 20, 2.0), verbose=0):
    grid_section_list = [(start, start+data_length) for start in grid_start_list]
    X = loader.get_diff_data_for_completion(interval=interval)
    cmfpn_l1_weight = hyperparameter_selection_of_cmf(X, interval, grid_section_list, l1_weight_list, verbose=verbose)
    ppca_n_components = hyperparameter_selection_of_pca(X, interval, grid_section_list, 'ppca', verbose=verbose)
    bpca_n_components = hyperparameter_selection_of_pca(X, interval, grid_section_list, 'bpca', verbose=verbose)
    return cmfpn_l1_weight, ppca_n_components, bpca_n_components


def test(loader, cmfpn_l1_weight, ppca_n_components, bpca_n_components, interval=10, data_length=2500, test_start_list=np.arange(1000, 41000, 5000), verbose=0):
    test_section_list = [(start, start + data_length) for start in test_start_list]
    X = loader.get_diff_data_for_completion(interval=interval)
    min_val = np.min(X, axis=0)
    cmfpn_arg_dict = dict(
        convolution_width=100//interval,
        n_components=2,
        response_l1_weight=0.0000,
        signal_l1_weight=cmfpn_l1_weight,
        signal_l2_weight=0.0,
        convergence_threshold=0.0000001,
        loop_max=100,
        fit_accelerator_max=0.0,
        transform_accelerator_max=0.0,
        verbose=0,
        initialization='smooth_svd',
        noise_scale = 0.01,
        noise_threshold = 0.00)

    cnmf_arg_dict = dict(
        convolution_width=100//interval,
        n_components=2,
        convergence_threshold=0.0000001,
        bias = min_val,
        loop_max=100)

    nmf_arg_dict = dict(
        convolution_width=1,
        n_components=2,
        convergence_threshold=0.0000001,
        bias = min_val,
        loop_max=100)

    completer_dict = {
        'ppca': RPCA(n_components=ppca_n_components, method_name='ppca', scale_option='vector'),
        'bpca': RPCA(n_components=bpca_n_components, method_name='bpca', scale_option='vector'),
        'nmf': CNMF(**nmf_arg_dict),
        'cnmf': CNMF(**cnmf_arg_dict),
        'cmf': CMFPN(**cmfpn_arg_dict),
    }

    loss_dict = completion_experiment(X, section_list=test_section_list, completer_dict=completer_dict, interval=interval, verbose=0)
    mean_rmse = {method: result.mean() for method, result in loss_dict['rmse'].items()}
    mean_md = {method: result.mean() for method, result in loss_dict['md'].items()}
    if verbose >= 2:
        print('mean RMSE on test data:')
        print(mean_rmse)
        print('mean absolute deviation on test data:')
        print(mean_md)
    return mean_rmse, mean_md, completer_dict

# file_path = 'ethylene_CO_down_sampled.txt'
# data = pd.read_csv(file_path, delim_whitespace=True)
# gas_columns = list(data.columns[1:3])
# used_columns = list(data.columns[3:19])
# observable_columns = list(data.columns[5:7])
# hidden_columns = list(set(used_columns) - set(observable_columns))
# gas_column_idxs = [1,2]
# observable_column_idxs = [used_columns.index(column_name) for column_name in observable_columns]
# hidden_column_idxs = [used_columns.index(column_name) for column_name in hidden_columns]
# interval = 200
# raw_down_sampled = data[::interval]
# m = raw_down_sampled[used_columns].mean(axis = 0)
# s = raw_down_sampled[used_columns].std(axis = 0)
# min_val = np.array(raw_down_sampled.min(axis=0))
# down_sampled = pd.DataFrame(raw_down_sampled)
# down_sampled[used_columns] = raw_down_sampled[used_columns].sub(m).div(s)
# train_data_length = 250000
# test_data_length = 250000
# train_start_list = [1750000, 2250000, 2750000, 3250000, 250000, 750000, 1250000]
# test_start_list = [250000, 750000, 1250000, 1750000, 2250000, 2750000, 3250000]
# train_section_list = [(start, start+train_data_length) for start in train_start_list]
# test_section_list = [(start, start+train_data_length) for start in train_start_list]
# # regression_window_length_list = [400, 600, 800, 1000, 1200]
# # l1_weight_list = [0.01, 0.02, 0.03, 0.04, 0.05]
#
#
# loss_function = rmse
#
#
# cmfpn_arg_dict = dict(
#     convolution_width=10000 // interval,
#     n_components=2,
#     activation_l1_weight=l1_weight,
#     activation_l2_weight=0.0,
#     base_max=10.0,
#     convergence_threshold=0.0000001,
#     loop_max=100,
#     fit_accelerator_max=0.0,
#     transfer_accelerator_max=0.0,
#     verbose=0,
#     initialization='smooth_svd')
#
# cnmf_arg_dict = dict(
#     convolution_width=10000 // interval,
#     n_components=2,
#     base_max=10.0,
#     convergence_threshold=0.0000001,
#     bias = min_val,
#     loop_max=100)
#
# nmf_arg_dict = dict(
#     convolution_width=1,
#     n_components=2,
#     base_max=10.0,
#     convergence_threshold=0.0000001,
#     bias = min_val,
#     loop_max=100)
#
# regressor_dict = {
#     'lr': linear_model.LinearRegression(normalize=False),
#     'lr_n': linear_model.LinearRegression(normalize=True),
#     'tlr': TimeSeriesLinearRegression(normalize=False),
#     'tlr_n': TimeSeriesLinearRegression(normalize=True),
#     'svd': MatrixCompletionRegression(RPCA(n_components=2, method_name='svd')),
#     'svd_n': MatrixCompletionRegression(RPCA(n_components=2, method_name='svd', scale_option='vector', center_option='TRUE')),
#     'ppca': MatrixCompletionRegression(RPCA(n_components=2, method_name='ppca')),
#     'ppca_n': MatrixCompletionRegression(RPCA(n_components=2, method_name='ppca', scale_option='vector', center_option='TRUE')),
#     'nmf': MatrixCompletionRegression(CNMF(**nmf_arg_dict)),
#     'cnmf': MatrixCompletionRegression(CNMF(**cnmf_arg_dict)),
#     'cmf': MatrixCompletionRegression(CMFPN(**cmfpn_arg_dict)),
# }
#
#
# loss_dict = regression_experiment(X, Y, train_section_list, test_section_list, regressor_dict)
# print(loss_dict)
