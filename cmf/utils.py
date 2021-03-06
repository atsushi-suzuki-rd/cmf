import numpy as np
import pandas as pd


def rmse(Y_hat, Y_test):
    return np.sqrt(np.mean((Y_hat - Y_test) ** 2))


def md(Y_hat, Y_test):
    return np.mean(np.abs(Y_hat - Y_test))


def standard_normalization(data):
    mean = data.median(axis=0)
    std = data.std(axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


def standard_scaling(data):
    std = data.std(axis=0)
    normalized_data = data / std
    return normalized_data


def median_absolute_deviation_normalization(data):
    median = data.median(axis=0)
    if isinstance(median, np.ndarray):
        mad = np.abs(data - median).mean(axis=0)
    if isinstance(median, pd.DataFrame):
        mad = (data - median).abs().mean(axis=0)
    normalized_data = (data - median) / mad
    return normalized_data


def absolute_deviation_scaling(data):
    if isinstance(data, np.ndarray):
        absolute_deviation = np.abs(data).mean(axis=0)
    if isinstance(data, pd.DataFrame):
        absolute_deviation = data.abs().mean(axis=0)
    normalized_data = data / absolute_deviation
    return normalized_data


class GasDataLoader(object):
    def __init__(self, file_path, time_column_idx=1):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, delim_whitespace=False)
        self.gas_column_idx_list = list(range(time_column_idx+1, time_column_idx+3))
        self.sensor_column_idx_list = list(range(time_column_idx+3, time_column_idx+19))
        self.gas_column_list = list(self.data.columns[self.gas_column_idx_list])
        self.sensor_column_list = list(self.data.columns[self.sensor_column_idx_list])

    def get_diff_data_for_regression(self, hidden_sensor_list, normalization=absolute_deviation_scaling, interval=1):
        hidden_column_idx_list = list(np.array(hidden_sensor_list) + 4)
        observable_column_idx_list = list(set(self.sensor_column_idx_list) - set(hidden_column_idx_list))
        hidden_column_list = list(self.data.columns[hidden_column_idx_list])
        observable_column_list = list(self.data.columns[observable_column_idx_list])
        normalized_data = normalization(self.data[::interval].diff()[1:])
        X = np.array(normalized_data[observable_column_list])
        Y = np.array(normalized_data[hidden_column_list])
        return X, Y

    def get_diff_data_for_completion(self, normalization=absolute_deviation_scaling, interval=1):
        normalized_data = normalization(self.data[::interval].diff()[1:])
        X = np.array(normalized_data[self.sensor_column_idx_list])
        return X

    def get_gas_diff(self, interval=1, section=None):
        if section is None:
            gas_diff = self.data[self.gas_column_list][::interval].diff()[1:]
        else:
            gas_diff = self.data[self.gas_column_list][section[0]:section[1]:interval].diff()[1:]
        return gas_diff


def generate_data(data_list, section_iter_list, interval=1):
    def truncate(data, section_sec, interval=1):
        (start_point, end_point) = (section_sec[0] // interval), (section_sec[1] // interval)
        return data[start_point:end_point]
    for section_sec_tuple in zip(*section_iter_list):
        yield tuple(truncate(data, section_sec, interval) for data, section_sec in zip(data_list, section_sec_tuple))


def regression_experiment(X, Y, train_section_list, test_section_list, regressor_dict, loss_function_dict={'rmse': rmse, 'md': md}, interval=1, verbose=2):
    loss_dict = {}
    for loss_function_name in loss_function_dict:
        loss_dict[loss_function_name] = {}
        for regressor_name in regressor_dict:
            loss_dict[loss_function_name][regressor_name] = np.full(len(train_section_list), np.nan)

    for i_section, (X_train, Y_train, X_test, Y_test) \
            in enumerate(generate_data(data_list=[X, Y, X, Y],
                                       section_iter_list=[train_section_list,
                                                          train_section_list,
                                                          test_section_list,
                                                          test_section_list],
                                       interval=interval)):
        for regressor_name, regressor in regressor_dict.items():
            if verbose >= 2:
                print(i_section, regressor_name)
            regressor.fit(X_train, Y_train)
            Y_hat = regressor.predict(X_test)
            for loss_function_name, loss_function in loss_function_dict.items():
                loss = loss_function(Y_hat, Y_test)
                loss_dict[loss_function_name][regressor_name][i_section] = loss
    return loss_dict


def completion_experiment(X, section_list, completer_dict, loss_function_dict={'rmse': rmse, 'md': md}, missing_ratio=0.5, interval=1, verbose=2):
    loss_dict = {}
    for loss_function_name in loss_function_dict:
        loss_dict[loss_function_name] = {}
        for completer_name in completer_dict:
            loss_dict[loss_function_name][completer_name] = np.full(len(section_list), np.nan)

    for i_section, (X_complete, ) \
            in enumerate(generate_data(data_list=[X],
                                       section_iter_list=[section_list],
                                       interval=interval)):
        for completer_name, completer in completer_dict.items():
            if verbose >= 2:
                print(i_section, completer_name)
            F = np.random.binomial(1, 1.-missing_ratio, X_complete.shape)
            X_missing = X_complete * F
            Z = completer.fit_transform(X_missing, filtre=F)
            X_hat = completer.inverse_transform(Z)
            for loss_function_name, loss_function in loss_function_dict.items():
                loss = loss_function(X_hat * (1 - F), X_complete * (1 - F))
                loss_dict[loss_function_name][completer_name][i_section] = loss
    return loss_dict

