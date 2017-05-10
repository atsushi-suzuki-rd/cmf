
# coding: utf-8

# In[26]:


import sys
import argparse
import os.path
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros, fromstring
from scipy.signal import get_window, stft
from numpy.random import seed, randint
import matplotlib
# sys.path.append('..')
from matplotlib import pylab as plt
from scipy.io.wavfile import read
from cmf.cnimf import CNIMF
import json
import pickle

parser = argparse.ArgumentParser(description='Decompose stereo wav file by CNMF.')
parser.add_argument('wav_file_name',                     action='store',                     nargs=None,                     const=None,                     default=None,                     type=str,                     choices=None,                     help='wav file name',                     metavar=None)
parser.add_argument('-s', '--seed_number',                     action='store',                     nargs='?',                     const=None,                     default=0,                     type=int,                     choices=None,                     help='seed_number',                     metavar=None)
parser.add_argument('-c', '--component_max',                     action='store',                     nargs='?',                     const=None,                     default=24,                     type=int,                     choices=None,                     help='the maximum of candidate number of component',                     metavar=None)
parser.add_argument('-w', '--convolution_width',                     action='store',                     nargs='?',                     const=None,                     default=32,                     type=int,                     choices=None,                     help='the convolution width in CNMF',                     metavar=None)
parser.add_argument('-l', '--loop_max',                     action='store',                     nargs='?',                     const=None,                     default=100,                     type=int,                     choices=None,                     help='the maximum number of loop iteration',                     metavar=None)
parser.add_argument('-b', '--base_max',                     action='store',                     nargs='?',                     const=None,                     default=10000.0,                     type=float,                     choices=None,                     help='the maximum of possible value in bases. This value is used only in model selection and not used in parameter estimation.',                     metavar=None)
parser.add_argument('-pd', '--pickle_dir',                     action='store',                     nargs='?',                     const=None,                     default='../dat/spectrogram/pickle',                     type=str,                     choices=None,                     help='Directory name where pickle files are stored.',                     metavar=None)
parser.add_argument('-jd', '--json_dir',                     action='store',                     nargs='?',                     const=None,                     default='../dat/spectrogram/json',                     type=str,                     choices=None,                     help='Directory name where npy files are stored.',                     metavar=None)
parser.add_argument('-ss', '--sampling_step',                     action='store',                     nargs='?',                     const=None,                     default=4,                     type=int,                     choices=None,                     help='sampling step.',                     metavar=None)
parser.add_argument('-tr', '--training_second',                     action='store',                     nargs='?',                     const=None,                     default=20,                     type=int,                     choices=None,                     help='length of training area (sec)',                     metavar=None)
parser.add_argument('-te', '--test_second',                     action='store',                     nargs='?',                     const=None,                     default=20,                     type=int,                     choices=None,                     help='length of test area (sec)',                     metavar=None)
parser.add_argument('-ds', '--down_sampling_step',                     action='store',                     nargs='?',                     const=None,                     default=2,                     type=int,                     choices=None,                     help='sampling step in down-sampled area.',                     metavar=None)
parser.add_argument('-ns', '--nperseg',                     action='store',                     nargs='?',                     const=None,                     default=256,                     type=int,                     choices=None,                     help='the number of samples per segment',                     metavar=None)
parser.add_argument('-lo', '--noverlap',                     action='store',                     nargs='?',                     const=None,                     default=None,                     type=int,                     choices=None,                     help='the number of samples per segment',                     metavar=None)


args = parser.parse_args()

loop_max = args.loop_max
base_max = args.base_max

wav_file_name = args.wav_file_name
seed_number = args.seed_number
# os.mkdir(npy_dir)
seed(seed_number)

fs, x_stereo = read(wav_file_name)
sampling_step = args.sampling_step
x = np.mean(x_stereo[::sampling_step, :], axis=1)
len_x = len(x)

nperseg = args.nperseg
if args.noverlap is None:
    noverlap = nperseg // 2
else:
    noverlap = args.noverlap
offset = nperseg // 2
train_length = (((fs // sampling_step) * args.training_second) // nperseg) * nperseg
test_length = (((fs // sampling_step) * args.test_second) // nperseg) * nperseg
print(train_length, test_length)

train_start_center = randint(nperseg, len_x - nperseg)
train_end_center = train_start_center + train_length
train_start = train_start_center - offset
train_end = train_end_center + offset
test_start_center = randint(nperseg, len_x - nperseg)
test_end_center = test_start_center + test_length
test_start = test_start_center - offset
test_end = test_end_center + offset
while train_start < test_end and train_end > test_start:
    test_start_center = randint(nperseg, x_len - nperseg)
    test_end_center = test_start_center + test_length
    test_start = test_start_center - offset
    test_end = test_end_center + offset


x_train = x[train_start:train_end]
x_test = x[test_start:test_end]
down_sampling_step = args.down_sampling_step
nperseg_for_down_sampled = nperseg // down_sampling_step
noverlap_for_down_sampled = noverlap // down_sampling_step
x_test_down_sampled = x[test_start:test_end:down_sampling_step]


(_, _, train_spec) = stft(x_train, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
(_, _, test_spec) = stft(x_test, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
(_, _, test_down_sampled_spec) = stft(x_test_down_sampled, fs, window='hann', nperseg=nperseg_for_down_sampled, noverlap=noverlap_for_down_sampled)

filtred = np.zeros(test_spec.shape)
filtred[:test_down_sampled_spec.shape[0],:] = np.abs(test_down_sampled_spec)
filtre = np.zeros(test_spec.shape)
filtre[:test_down_sampled_spec.shape[0],:] = 1

train_data = np.abs(train_spec) # np.log((np.abs(train_spec) ** 2) + 1.0)
test_data = np.abs(test_spec) # np.log((np.abs(test_spec) ** 2) + 1.0)

true_width = args.convolution_width
component_max = args.component_max

nmf = CNIMF(true_width=1, verbose=0, component_max=component_max, base_max=base_max, loop_max=loop_max)

nmf.fit(train_data.T)

nmf.transfer(test_data.T, transfer_filtre=filtre.T)

nmf_criteria_completion = nmf.evaluate_criteria(test_data.T, filtre.T)

json_file_path = args.json_dir + '/' + 'nmf_criteria_completion.json'
with open(json_file_path, 'w') as f:
    json.dump(nmf_criteria_completion, f)

print(nmf_criteria_completion)
# pickle_file_path = args.pickle_dir + '/' + 'nmf.pickle'
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(nmf, f)

cnimf = CNIMF(true_width=true_width, verbose=0, component_max=component_max, base_max=base_max, loop_max=loop_max)

cnimf.fit(train_data.T)

cnimf.transfer(test_data.T, transfer_filtre=filtre.T)

cnimf_criteria_completion = cnimf.evaluate_criteria(test_data.T, filtre.T)

json_file_path = args.json_dir + '/' + 'cnimf_criteria_completion.json'
with open(json_file_path, 'w') as f:
    json.dump(cnimf_criteria_completion, f)

print(cnimf_criteria_completion)
# pickle_file_path = args.pickle_dir + '/' + 'cnimf.pickle'
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(cnimf, f)

np.max(np.abs(train_spec))


# In[ ]:



