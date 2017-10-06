import numpy as np


def rmse(Y_hat, Y_test):
    return np.sqrt(np.mean((Y_hat - Y_test) ** 2))


def generate_data(data_list, start_iter_list, data_length, interval):
    def truncate(data, start_sec, interval):
        start_point = start_sec // interval
        end_point = start_point + (data_length // interval)
        return data[start_point:end_point]
    for start_sec_tuple in zip(*start_iter_list):
        yield tuple(truncate(data, start_sec, interval) for data, start_sec in zip(data_list, start_sec_tuple))
