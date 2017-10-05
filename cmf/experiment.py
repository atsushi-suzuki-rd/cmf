import numpy as np
import pandas as pd
from sklearn import linear_model
import pyper
from cmf.cmfpn import CMFPN
from cmf.cnmf import CNMF


class TimeSeriesLinearRegression(object):
    def __init__(self, window_width=5, **kwargs):
        self.window_width=window_width
        self.linear_regressor = linear_model.LinearRegression(**kwargs)

    def fit(self, X, Y, **kwargs):
        XX = self._sliding_duplicate(X)
        self.linear_regressor.fit(XX, Y)

    def predict(self, X, **kwargs):
        XX = self._sliding_duplicate(X)
        return self.linear_regressor.predict(XX)

    def _sliding_duplicate(self, X):
        return np.concatenate(
            [np.pad(np.array(X), self.window_width, mode='constant', axis=0)[self.window_width-i:-(self.window_width-i), :]
            for i in range(-self.window_width, self.window_width)],
            axis=1)


class MatrixCompletionRegression(object):
    def __init__(self, matrix_factorizer):
        self.matrix_factorizer = matrix_factorizer
        self.X_n_features = None
        self.Y_n_features = None

    def fit(self, X, Y, **kwargs):
        self.X_n_features = X.shape[1]
        self.Y_n_features = Y.shape[1]
        XY = np.concatenate([X, Y], axis=1)
        self.matrix_factorizer.fit(XY, **kwargs)

    def predict(self, X, **kwargs):
        n_samples = X.shape[0]
        XO = np.concatenate([X, np.zeros((n_samples, self.Y_n_features))], axis=1)
        F = np.concatenate([np.ones(X.shape), np.zeros((n_samples, self.Y_n_features))])
        Z = self.matrix_factorizer.transform(X, filtre=F)
        XY_hat = self.matrix_factorizer.inverse_transform(Z)
        Y_hat = XY_hat[:, self.X_n_features:]
        return Y_hat


class RPCA(object):
    n_instances = 0
    r = pyper.R(use_pandas='True', use_numpy=True)

    def __init__(self, n_components=2, method_name='ppca', scale='none', center='FALSE'):
        self.identifier = 'py_{}'.format(self.n_instances)
        self.n_instances += 1
        self.pca_str = 'pca_{}'.format(self.identifier)
        self.loadings_str = 'loadings_{}'.format(self.identifier)
        self.n_components = n_components
        self.method_name = method_name
        self.scale = scale
        self.center = center
        self.loadings = None

    def fit(self, X):
        self._fit(X)

    def _fit(self, X):
        self._fit_transform(X)

    def fit_transform(self, X):
        return self._fit_transform(X)

    def _fit_transform(self, X):
        X_str = 'X_{}'.format(self.identifier)
        scores_str = 'scores_{}'.format(self.identifier)
        self.r.assign(X_str, X)
        self.r('{} <- pca({}, method="{}", nPcs={}, scale={}, center={})'.format(
            self.pca_str,
            X_str,
            self.method_name,
            self.n_components,
            self.scale,
            self.center))
        self.r('{} <- {}@loadings'.format(self.loadings_str, self.pca_str))
        self.r('{} <- {}@scores'.format(scores_str, self.pca_str))
        self.loadings = self.r.get('t({})'.format(self.loadings_str))

    def transform(self, X, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        F = filtre
        X_new_str = 'X_new_{}'.format(self.identifier)
        F_str = 'F_{}'.format(self.identifier)
        predict_str = 'predict_{}'.format(self.identifier)
        new_scores_str = 'new_scores_{}'.format(self.identifier)
        self.r.assign(X_new_str, X)
        self.r.assign(F_str, F)
        self.r('{}[{}==NA] <- NA'.format(X_new_str, F_str))
        self.r('{} <- predict({}, {})'.format(predict_str, self.pca_str, X_new_str))
        self.r('{} <- {}$scores'.format(new_scores_str, predict_str))
        new_scores = self.r.get(new_scores_str)
        return new_scores

    def inverse_transform(self, scores):
        return scores @ self.loadings



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
# lr_error_table = np.float('nan') * np.ones(len(train_start_list))
# lrn_error_table = np.float('nan') * np.ones(len(train_start_list))
# mlr_error_table = np.float('nan') * np.ones([len(regression_window_length_list), len(train_start_list)])
# mlrn_error_table = np.float('nan') * np.ones([len(regression_window_length_list), len(train_start_list)])
# crmf_completion_error_table = np.float('nan') * np.ones((len(l1_weight_list), len(train_start_list)))
# cnimf_completion_error_table = np.float('nan') * np.ones((len(train_start_list),))
# cnimf_with_bias_modification_completion_error_table = np.float('nan') * np.ones((len(train_start_list),))
# test_data_mean_square_table = np.float('nan') * np.ones((len(train_start_list),))


def rmse(Y_hat, Y_test):
    return np.sqrt(np.mean((Y_hat - Y_test) ** 2))

loss_function = rmse


def generate_data(data_list, start_iter_list, data_length, interval):
    def truncate(data, start_sec, interval):
        start_point = start_sec // interval
        end_point = start_point + (data_length // interval)
        return data[start_point:end_point]
    for start_sec_tuple in zip(*start_iter_list):
        yield tuple(truncate(data, start_sec, interval) for data, start_sec in zip(data_list, start_sec_tuple))

cmfpn_arg_dict = dict(
    convolution_max=200,
    true_width=10000 // interval,
    true_n_components=2,
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
    convolution_max=200,
    true_width=10000 // interval,
    true_n_components=2,
    base_max=10.0,
    convergence_threshold=0.0000001,
    loop_max=100)

nmf_arg_dict = dict(
    convolution_max=200,
    true_width=1,
    true_n_components=2,
    base_max=10.0,
    convergence_threshold=0.0000001,
    loop_max=100)

min_XY = np.min(XY_train)
XY_train_shifted = XY_train - min_XY
cnimf.fit(XY_train_shifted, None, filtre=F_train)
XO_test_shifted = np.maximum(XO_test - min_XY, np.zeros(XO_test.shape))
cnimf.transfer(XO_test_shifted, transfer_filtre=F_test)
XY_cnimf = cnimf.transfer_approximated + min_XY
cnimf_completion_error = np.mean(((XY_cnimf - XY_test)[:, hidden_column_idxs]) ** 2)
cnimf_bias = (XY_cnimf - XY_test).mean()
cnimf_with_bias_modification_completion_error = np.mean(((XY_cnimf - cnimf_bias - XY_test)[:, hidden_column_idxs]) ** 2)
cnimf_completion_error_table[i_start] = cnimf_completion_error
cnimf_with_bias_modification_completion_error_table[i_start] = cnimf_with_bias_modification_completion_error
test_data_mean_square_table[i_start] = np.mean(XY_test ** 2)

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

loss_dict = {}
for regressor_name in regressor_dict:
    loss_dict[regressor_name] = np.full(len(train_section_list), np.nan)

for i_section, (X_train, Y_train, X_test, Y_test) \
        in enumerate(generate_data(data_list=[X, Y, X, Y],
                                   start_iter_list=[train_section_list,
                                                    train_section_list,
                                                    test_section_list,
                                                    test_section_list])):
    for regressor_name, regressor in regressor_dict.items():
        regressor.fit(X_train, Y_train)
        Y_hat = regressor.predict(X_test)
        loss = loss_function(Y_hat, Y_test)
        loss_dict[regressor_name][i_section] = loss


for i_start in range(len(train_start_list)):
    train_start = train_start_list[i_start] // interval
    test_start = test_start_list[i_start] // interval
    train_end = train_start + (train_data_length // interval)
    test_end = test_start + (test_data_length // interval)
    X_train = np.array(down_sampled[observable_columns][train_start:train_end].diff())[1:]
    Y_train = np.array(down_sampled[hidden_columns][train_start:train_end].diff())[1:]
    XY_train = np.array(down_sampled[used_columns][train_start:train_end].diff())[1:]
    F_train = np.ones(XY_train.shape)
    X_test = np.array(down_sampled[observable_columns][test_start:test_end].diff())[1:]
    Y_test = np.array(down_sampled[hidden_columns][test_start:test_end].diff())[1:]
    XY_test = np.array(down_sampled[used_columns][test_start:test_end].diff())[1:]
    F_test = np.zeros(XY_test.shape)
    F_test[:, observable_column_idxs] = 1
    XO_test = XY_test * F_test
    lr = linear_model.LinearRegression()
    lr.fit(X_train, Y_train)
    Y_lr = lr.predict(X_test)
    lr_error = np.mean((Y_lr - Y_test) ** 2)
    lr_error_table[i_start] = lr_error
    lrn = linear_model.LinearRegression(normalize=True)
    lrn.fit(X_train, Y_train)
    Y_lrn = lrn.predict(X_test)
    lrn_error = np.mean((Y_lrn - Y_test) ** 2)
    lrn_error_table[i_start] = lrn_error
    for i_regression_window_length in range(len(regression_window_length_list)):
        regression_window_length = regression_window_length_list[i_regression_window_length]
        n_window_samples = regression_window_length // interval
        XX_train = np.concatenate([np.roll(np.array(down_sampled[observable_columns].diff()), i, axis=0) for i in
                                   range(-n_window_samples, n_window_samples)], axis=1)[train_start:train_end, :][1:, :]
        XX_test = np.concatenate([np.roll(np.array(down_sampled[observable_columns].diff()), i, axis=0) for i in
                                  range(-n_window_samples, n_window_samples)], axis=1)[test_start:test_end, :][1:, :]
        mlr = linear_model.LinearRegression()
        mlr.fit(XX_train, Y_train)
        Y_mlr = mlr.predict(XX_test)
        mlr_error = np.mean((Y_mlr - Y_test) ** 2)
        mlr_error_table[i_regression_window_length, i_start] = mlr_error
        mlrn = linear_model.LinearRegression(normalize=True)
        mlrn.fit(XX_train, Y_train)
        Y_mlrn = mlrn.predict(XX_test)
        mlrn_error = np.mean((Y_mlrn - Y_test) ** 2)
        mlrn_error_table[i_regression_window_length, i_start] = mlrn_error
    for i_l1_weight in range(len(l1_weight_list)):
        l1_weight = l1_weight_list[i_l1_weight]
        loss_weight = 1.0 / np.ones(XY_train.shape[1])
        base_l2_weight = 1.0 / np.ones(XY_train.shape[1])
        base_l1_weight = 0.0 / np.ones(XY_train.shape[1])
        crmf.fit(XY_train, None, filtre=F_train, loss_weight=loss_weight, base_l1_weight=base_l1_weight,
                 base_l2_weight=base_l2_weight)
        crmf.transfer(XO_test, transfer_filtre=F_test)
        XY_crmf = crmf.transfer_approximated
        crmf_completion_error = np.mean(((XY_crmf - XY_test)[:, hidden_column_idxs]) ** 2)
        crmf_completion_error_table[i_l1_weight, i_start] = crmf_completion_error
