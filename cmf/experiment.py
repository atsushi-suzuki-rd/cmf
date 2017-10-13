import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from cmf.cmfpn import CMFPN
from cmf.cnmf import CNMF
from cmf.utils import rmse, generate_data, completion_experiment, absolute_deviation_scaling
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


def get_final_section(loader):
    interval = 10
    X = loader.get_diff_data_for_completion(interval=interval)
    section_list = [(36000, 38500)]
    return list(generate_data(data_list=[X], section_iter_list=[section_list], interval=interval))[0][0]


def plot_estimates_histogram(loader, completer_dict, label):
    X = get_final_section(loader=loader)
    minimum_center = -0.3
    maximum_center = 0.3
    bin_interval = 0.01
    x_lim_max = 1000
    edges = np.arange(minimum_center-bin_interval/2, maximum_center+bin_interval/2, bin_interval, dtype=np.float32)
    data_list = [('Test Data 36000-38500s', X)] + \
                [('Approximated by {}'.format(completer_name), completer.inverse_transform(completer.transform(X)))
                 for completer_name, completer in completer_dict.items()]
    fig, axs = plt.subplots(nrows=1, ncols=len(data_list)+1, figsize=(5, 5))
    for (ax, (x_label, data_elements)) in zip(axs, data_list):
        ax.hist(np.array(data_elements).reshape((-1, 1)), bins=edges, alpha=0.6, orientation='vertical')
        ax.set_ylim(minimum_center-bin_interval/2, maximum_center+bin_interval/2)
        ax.set_xlim(0, x_lim_max)
        ax.set_xlabel(x_label)
    fig.savefig('../dat/estimation/histogram_sdm_{}_{}.png'.format(label[0], label[1]), format='png')
    fig.savefig('../dat/estimation/histogram_sdm_{}_{}.pdf'.format(label[0], label[1]), format='pdf')
    return fig


def plot_ground_truth(loader, cmf, label, signal_sign_permute=np.identity(2), original_bound=30, system_bound=10, figsize=(15, 2)):
    signal_sign_permute = np.array(signal_sign_permute)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    interval = 10
    section = (36000, 38500)
    truth = np.array(loader.get_gas_diff(interval=interval, section=section))
    scaled_truth = absolute_deviation_scaling(truth)
    x_range = np.arange(36000 + interval, 38500, interval)
    axs[0].plot(x_range, scaled_truth[:, 0], label=label[0])
    axs[0].plot(x_range, scaled_truth[:, 1], label=label[1])
    axs[0].set_xlabel('time /s')
    axs[0].set_ylabel('the magnitude of the change')
    axs[0].set_ylim(-original_bound, original_bound)
    axs[0].legend(bbox_to_anchor=(1, 1.5), loc='upper right')
    X = get_final_section(loader=loader)
    Z = cmf.transform(X)
    scaled_Z = absolute_deviation_scaling(Z) @ signal_sign_permute
    axs[1].plot(x_range, (-scaled_Z / np.std(scaled_Z, axis=0))[1:, 0], label='system 0')
    axs[1].plot(x_range, (-scaled_Z / np.std(scaled_Z, axis=0))[1:, 1], label='system 1')
    axs[1].set_xlabel('time /s')
    axs[1].set_ylabel('the magnitude of the impulse responses')
    axs[1].set_ylim(-system_bound, system_bound)
    axs[1].legend(bbox_to_anchor=(1, 1.5), loc='upper right')
    # fig.tight_layout()
    fig.subplots_adjust(top=0.75)
    fig.savefig('../dat/estimation/density_signal_sdm_{}_{}.png'.format(label[0], label[1]), format='png')
    fig.savefig('../dat/estimation/density_signal_sdm_{}_{}.pdf'.format(label[0], label[1]), format='pdf')
    return fig
