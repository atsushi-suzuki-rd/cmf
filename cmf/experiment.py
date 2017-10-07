import numpy as np
import pandas as pd
from sklearn import linear_model
from cmf.cmfpn import CMFPN
from cmf.cnmf import CNMF
from cmf.utils import rmse, generate_data
from cmf.wrapper import TimeSeriesLinearRegression, MatrixCompletionRegression, RPCA


file_path = 'ethylene_CO_down_sampled.txt'
data = pd.read_csv(file_path, delim_whitespace=True)
gas_columns = list(data.columns[1:3])
used_columns = list(data.columns[3:19])
observable_columns = list(data.columns[5:7])
hidden_columns = list(set(used_columns) - set(observable_columns))
gas_column_idxs = [1,2]
observable_column_idxs = [used_columns.index(column_name) for column_name in observable_columns]
hidden_column_idxs = [used_columns.index(column_name) for column_name in hidden_columns]
interval = 200
raw_down_sampled = data[::interval]
m = raw_down_sampled[used_columns].mean(axis = 0)
s = raw_down_sampled[used_columns].std(axis = 0)
min_val = np.array(raw_down_sampled.min(axis=0))
down_sampled = pd.DataFrame(raw_down_sampled)
down_sampled[used_columns] = raw_down_sampled[used_columns].sub(m).div(s)
train_data_length = 250000
test_data_length = 250000
train_start_list = [1750000, 2250000, 2750000, 3250000, 250000, 750000, 1250000]
test_start_list = [250000, 750000, 1250000, 1750000, 2250000, 2750000, 3250000]
train_section_list = [(start, start+train_data_length) for start in train_start_list]
test_section_list = [(start, start+train_data_length) for start in train_start_list]
# regression_window_length_list = [400, 600, 800, 1000, 1200]
# l1_weight_list = [0.01, 0.02, 0.03, 0.04, 0.05]


loss_function = rmse


cmfpn_arg_dict = dict(
    convolution_width=10000 // interval,
    n_components=2,
    activation_l1_weight=l1_weight,
    activation_l2_weight=0.0,
    base_max=10.0,
    convergence_threshold=0.0000001,
    loop_max=100,
    fit_accelerator_max=0.0,
    transfer_accelerator_max=0.0,
    verbose=0,
    initialization='smooth_svd')

cnmf_arg_dict = dict(
    convolution_width=10000 // interval,
    n_components=2,
    base_max=10.0,
    convergence_threshold=0.0000001,
    bias = min_val,
    loop_max=100)

nmf_arg_dict = dict(
    convolution_width=1,
    n_components=2,
    base_max=10.0,
    convergence_threshold=0.0000001,
    bias = min_val,
    loop_max=100)

regressor_dict = {
    'lr': linear_model.LinearRegression(normalize=False),
    'lr_n': linear_model.LinearRegression(normalize=True),
    'tlr': TimeSeriesLinearRegression(normalize=False),
    'tlr_n': TimeSeriesLinearRegression(normalize=True),
    'svd': MatrixCompletionRegression(RPCA(n_components=2, method_name='svd')),
    'svd_n': MatrixCompletionRegression(RPCA(n_components=2, method_name='svd', scale='vector', center='TRUE')),
    'ppca': MatrixCompletionRegression(RPCA(n_components=2, method_name='ppca')),
    'ppca_n': MatrixCompletionRegression(RPCA(n_components=2, method_name='ppca', scale='vector', center='TRUE')),
    'nmf': MatrixCompletionRegression(CNMF(**nmf_arg_dict)),
    'cnmf': MatrixCompletionRegression(CNMF(**cnmf_arg_dict)),
    'cmf': MatrixCompletionRegression(CMFPN(**cmfpn_arg_dict)),
}


loss_dict = regression_experiment(X, Y, train_section_list, test_section_list, regressor_dict)
print(loss_dict)
