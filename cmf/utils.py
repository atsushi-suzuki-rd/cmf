import numpy as np


def rmse(Y_hat, Y_test):
    return np.sqrt(np.mean((Y_hat - Y_test) ** 2))


def generate_data(data_list, section_iter_list, interval=1):
    def truncate(data, section_sec, interval=1):
        (start_point, end_point) = (section_sec[0] // interval), (section_sec[1] // interval)
        return data[start_point:end_point]
    for section_sec_tuple in zip(*section_iter_list):
        yield tuple(truncate(data, start_sec, interval) for data, start_sec in zip(data_list, section_sec_tuple))


def regression_experiment(X, Y, train_section_list, test_section_list, regressor_dict, loss_function=rmse, verbose=2):
    loss_dict = {}
    for regressor_name in regressor_dict:
        loss_dict[regressor_name] = np.full(len(train_section_list), np.nan)

    for i_section, (X_train, Y_train, X_test, Y_test) \
            in enumerate(generate_data(data_list=[X, Y, X, Y],
                                       section_iter_list=[train_section_list,
                                                          train_section_list,
                                                          test_section_list,
                                                          test_section_list])):
        for regressor_name, regressor in regressor_dict.items():
            if verbose >= 2:
                print(i_section, regressor_name)
            regressor.fit(X_train, Y_train)
            Y_hat = regressor.predict(X_test)
            loss = loss_function(Y_hat, Y_test)
            loss_dict[regressor_name][i_section] = loss
    return loss_dict


def completion_experiment(X, section_list, completer_dict, loss_function=rmse, missing_ratio=0.5, verbose=2):
    loss_dict = {}
    for completer_name in completer_dict:
        loss_dict[completer_name] = np.full(len(section_list), np.nan)

    for i_section, (X_complete, ) \
            in enumerate(generate_data(data_list=[X],
                                       section_iter_list=[section_list])):
        for completer_name, completer in completer_dict.items():
            if verbose >= 2:
                print(i_section, completer_name)
            F = np.random.binomial(1, 1.-missing_ratio, X_complete.shape)
            X_missing = X_complete * F
            Z = completer.fit_transform(X_missing, filtre=F)
            X_hat = completer.inverse_transform(Z)
            loss = loss_function(X_hat, X_complete)
            loss_dict[completer_name][i_section] = loss
    return loss_dict

