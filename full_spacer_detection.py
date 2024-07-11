# %%

# save_dir: Models
# save_dir_data: outputs
# folder_dir: D:/SRP_DNAencoding/Tin
# data_file_name = outputs\full_spacer_detection_10k.csv

import csv
from xml.dom.minidom import Entity
from IPython.display import clear_output
from re import L
from scipy import signal as scipy_signal
import re
from scipy import signal
from operator import index
import time
from tqdm import tqdm
from scipy.io import loadmat
import gc
import random
import math
import ScrappySequence as ss
import randomizeSequence as rSeq
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import os
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import wandb

key = '4e12773a4724334345ecccf11dcf4d485280365e'

entity = 'srp_nucleotrace'
#import warnings
# warnings.filterwarnings("error")
#import logging
# logging.basicConfig(filename="myfile.txt",level=logging.DEBUG)
# logging.captureWarnings(True)
try:
    JOB_ID = os.environ['SLURM_JOBID']
    JOB_NAME = os.environ['SLURM_JOB_NAME']
except:
    JOB_ID = 666
    JOB_NAME = 'laptop'
logfile = open(f"{JOB_NAME}.{JOB_ID}.log", 'a')


def closeLog():
    logfile.close()


def printLog(*printArgs):
    if not(JOB_NAME == 'laptop' and JOB_ID == 666):
        out = ' '.join([str(i) for i in printArgs])
        logfile.write(out)
        logfile.write('\n')
        logfile.flush()
    else:
        print(*printArgs)  # for running on laptop


def save_data_func(tensor, save_path, tensor_name, data_name):
    if not os.path.isfile(save_path):
        torch.save(torch.tensor(tensor), save_path)
    else:
        count = 0
        while os.path.isfile(save_path):
            print(save_path, ' already exist')
            save_path = os.path.join(
                folder_dir+"/data", data_name+'_'+str(count)+tensor_name+".pt")
            count += 1
        print('creating new path for:', tensor_name, ' in ', save_path)
        torch.save(torch.tensor(tensor), save_path)
    return save_path


datasets = {}


# Commented out IPython magic to ensure Python compatibility.

############
# figures from plt allow
############


# start from a checkpoint or start from scratch?
###### IMPORTANT #######
# setting to False overdrive the checkpoint in /content/drive/Shared drives/SRP-Synthetic-DNA-Storage/Models/model_name.pt


# you can change the model name to create new checkpoint for possibly new model
###### IMPORTANT #######
# the convetion is to use the notebook file name as the model_name

model_name = 'spacer_detector'
data_name_real = 'data224_0'
data_name_scrappies = 'data224'
path_to_datafile = 'SRP-Test-Data/full_data_pow2_224_clstm_onlyifcontain.mat'
csv_file = 'full_spacer_detection.csv'
# A directory to save our model to (will create it if it doesn't exist)
save_dir = 'Models'
save_dir_data = 'outputs'
folder_dir = '/mnt/lustre/projects/vs25/Tin'
if os.path.isdir(folder_dir):
    matplotlib.use("Agg")

if not os.path.isdir(folder_dir):
    folder_dir = 'D:/SRP_DNAencoding/Tin'
if not os.path.isdir(folder_dir):
    folder_dir = '/content/drive/Shared drives/SRP-Synthetic-DNA-Storage'
if not os.path.isdir(folder_dir):
    raise ValueError('folder_dir not defined')
##########################################
######### ADDING PARSER ##################
##########################################

parser = argparse.ArgumentParser(description="CNN_LSTM")
parser.add_argument('--wandb_key', default=key)
parser.add_argument('--csv_file', default=csv_file)
parser.add_argument('--wandb_entity', default=entity)
parser.add_argument("--model_name", default=model_name)
parser.add_argument("--data_name_real", default=data_name_real)
parser.add_argument("--data_name_scrappies", default=data_name_scrappies)

parser.add_argument("--path_to_datafile", default=path_to_datafile)


parser.add_argument("--folder_dir", default=folder_dir)
parser.add_argument("--save_dir", default=save_dir,
                    help='Name of the folder where to save Pytoch models')
parser.add_argument("--save_dir_data", default=save_dir_data,
                    help='name of the folder where to store data')

parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--batch_size", default=16)

parser.add_argument("--new", action='store_true',
                    help='Start training from epoch 0 when true, from last checkpoint when false.')
# skip_training is false be default
parser.add_argument("--skip_training", action='store_true')
parser.add_argument("--num_gpus", default=1)
# this is false by default
parser.add_argument("--cpu_mode", action='store_true')
parser.add_argument("--gen_data", action='store_true')
parser.add_argument("--gen_data_real", action='store_true')

if folder_dir != '/content/drive/Shared drives/SRP-Synthetic-DNA-Storage':
    args = parser.parse_args()
else:
    args = parser.parse_args("")

model_name = args.model_name
# A directory to save our model to (will create it if it doesn't exist)
save_dir = args.save_dir
save_dir_data = args.save_dir_data
folder_dir = args.folder_dir
cpu_mode = args.cpu_mode
gen_data = args.gen_data
gen_data_real = args.gen_data_real
data_name_real = args.data_name_real
data_name_scrappies = args.data_name_scrappies
path_to_datafile = args.path_to_datafile
# Define the hyperparameters
learning_rate = float(args.learning_rate)
num_epochs = int(args.num_epochs)
batch_size = int(args.batch_size)
start_from_checkpoint = not args.new  # this is True by default
skip_training = args.skip_training  # this is False by default
csv_file = args.csv_file

#! WANDB LOGIN
key = args.wandb_key
entity = args.wandb_entity
wandb.login(key=key)


# get the number of gpus available
if cpu_mode == False:
    gpuCount = torch.cuda.device_count()
    assert torch.cuda.is_available()

    if gpuCount != int(args.num_gpus):
        print("Available gpus and required gpus do not match!")
        print("Available gpus:", gpuCount, "Required gpus:", args.num_gpus)
        exit(1)


# skip_training = True
path_to_model = os.path.join(folder_dir, save_dir)
print('model name:', model_name)
print('save_dir:', save_dir, '\nsave_dir_data (outputs where you load data from):',
      save_dir_data, '\nfolder_dir:', folder_dir)
print('path_to_model:', path_to_model)
print('path to Matlab:', path_to_datafile)
print('learning rate:', learning_rate)
print('num_epochs:', num_epochs)
print('batch_size:', batch_size)
print('start_from_checkpoint:', start_from_checkpoint)
print('skip_training', skip_training)
print('csv_files: ', csv_file)


##############################################################
################ GENERATE DATA ###############################
##############################################################

data_file_name = os.path.join(
    save_dir_data, csv_file)  # F256
print('data_file_name (save_dir_data + file name): ', data_file_name)

upload = open(os.path.join(folder_dir, data_file_name), "r")
# %cd /content/drive/Shared drives/SRP-Synthetic-DNA-Storage

# necessary imports
# from sklearn.utils import shuffle
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils.np_utils import to_categorical


# adding noise to the signal, python version of the MATLAB code which was sent to me
def generate_samples(mean_curr, stdv_curr, dwell):
    signal_s = []
    index_s = []
    for i in range(0, len(mean_curr)):
        p = 1/dwell[i]
        if p > 1:
            p = 1
        r = np.random.geometric(p=p) + 1
        signal_s = np.concatenate(
            (signal_s, stdv_curr[i]*np.random.normal(size=r)+mean_curr[i]))
        index_s = np.concatenate((index_s, i*np.ones(r)))
    return signal_s, index_s

# reading the reference signals from the scrappie squiggle outputs


def read_data(datafile_path):  # 'datafile_path' refers to the squiggle output file in .csv format

    currents = []  # list of all reference currents
    std_dev = []
    dwells = []
    comments = []
    sequences = []

    sequence = []
    curr = []
    dev = []
    dw = []

    with open(datafile_path, 'r') as f:
        for line in f:

            if line.startswith('#'):
                comments.append(line)

            elif line.startswith('pos'):
                if len(curr) == 0:
                    continue
                sequences.append(''.join(sequence))
                currents.append(curr)
                std_dev.append(dev)
                dwells.append(dw)

                sequence = []
                curr = []
                dev = []
                dw = []

            else:
                parts = line.split(',')
                assert len(parts) == 5
                sequence.append(parts[1])
                curr.append(float(parts[2]))
                dev.append(float(parts[3]))
                dw.append(float(parts[4]))

    sequences.append(''.join(sequence))
    currents.append(curr)
    std_dev.append(dev)
    dwells.append(dw)

    return sequences, comments, currents, std_dev, dwells


spacer_len = 6
letter_len = 7


def generate_label(indexes):
    label = np.zeros(len(indexes))

    is_in_letter = True
    remaining = letter_len
    previous = 0
    for i, v in enumerate(indexes):  # indexes 00001111

        if v != previous:
            remaining -= 1
            previous = v
            if remaining == 0:
                if is_in_letter:
                    remaining = spacer_len
                else:
                    remaining = letter_len
                is_in_letter = not is_in_letter

        if is_in_letter:
            label[i] = 0
        else:
            label[i] = 1

#is_in_letter = T
#remaining = 6
#previous = -1
#  L     L    L      L L L L S S S S  S  S  L  L  L  L  L  L  L  S  S  S  S  S  S  L  L  L  L  L  L  L  S
    # 00000 1111 222222 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
    # 0     0000 000000 0 0 0 0 1 1 1 1  1  1  0
    return label


spacer_len = 6
letter_len = 7


def generate_spacer_label(indexes):
    indexes = list(indexes)

    first_spacer_base = indexes.index(56)
    last_spacer_base = len(indexes) - 1 - indexes[::-1].index(152)

    label = np.zeros(len(indexes))

    is_in_spacer = True
    remaining = spacer_len
    previous = 56
    for i, v in enumerate(indexes[first_spacer_base:last_spacer_base+1]):

        if v != previous:
            remaining -= 1
            previous = v
            if remaining == 0:
                if is_in_spacer:
                    remaining = letter_len
                else:
                    remaining = spacer_len
                is_in_spacer = not is_in_spacer

        if is_in_spacer:
            label[first_spacer_base + i] = 1
        else:
            label[first_spacer_base + i] = 0

    return label


# generate data (samples per ref)


# a comment is of this shape
#  #Seq[1. 4.3.42.19.61.13.26.40.9],,,,
#  where '1.' is the sequence number
#  '4' is the forward barcode
#  '3' is the reverse barcode
#  and the remaining numbers are letters
def parse_comment(comment):

    match = re.search(
        "^#Seq\[(\d+\.) (\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\],,,,$", comment)
    assert match is not None, "The comment string is not in the correct format"
    groups = list(match.groups())

    assert len(groups) == 10, f"{groups} {comment}"

    sequence_no = groups[0]
    barcodes = groups[1:3]
    letters = groups[3:10]
    assert len(barcodes) == 2 and len(
        letters) == 7, f"groups: {groups} barcodes: {len(barcodes)} letters: {len(letters)}"
    return sequence_no, [int(i) for i in barcodes], [int(i) for i in letters]


# generate data (samples per ref)

# 'num_samples_per_reference' refers to number of noisy signals generated from a single reference signal
def generate_data2(sequences, comments, currents, std_dev, dwells, num_samples_per_reference):
    assert len(sequences) == len(comments) == len(
        currents) == len(std_dev) == len(dwells)

    signals = [0]*(len(currents)*num_samples_per_reference)
    index = []
    spacer_labels = [0]*(len(currents)*num_samples_per_reference)
    letter_labels = [0]*(len(currents)*num_samples_per_reference)
    barcode_labels = [0]*(len(currents)*num_samples_per_reference)

    for i in range(0, len(currents)):  # 'len(currents)' -> number of reference signals
        seq = ss.ScrappySequence(
            sequences[i], currents[i], std_dev[i], dwells[i])
        # remove initial and final 8As
        seq = ss.ScrappySequence.fromSequenceItems(seq.slice(8, len(seq) - 8))

        # print("Sequence", i + 1, "of", len(currents))

        _, barcode_label, letter_label = parse_comment(comments[i])

        for j in range(0, num_samples_per_reference):
            # noisy signal generation for each instance
            signal_s, index_s = generate_samples(
                seq.currents, seq.sds, seq.dwells)
            spacer_label = generate_spacer_label(index_s)

            signals[i * num_samples_per_reference + j] = signal_s
            spacer_labels[i * num_samples_per_reference + j] = spacer_label
            letter_labels[i * num_samples_per_reference + j] = letter_label
            barcode_labels[i * num_samples_per_reference + j] = barcode_label

    return signals, index, spacer_labels, letter_labels, barcode_labels


# generating training data
def prepare_train2(dataset_path, num_samples_per_reference):
    sequences, comments, currents, std_dev, dwells = read_data(
        dataset_path)  # read the reference signals
    signals, index, spacer_labels, letter_labels, barcode_labels = generate_data2(
        sequences, comments, currents, std_dev, dwells, num_samples_per_reference)  # generate the noisy dataset from reference signals
    return signals, index, spacer_labels, letter_labels, barcode_labels


# extimating dataset size
alphabet = 256
samples_per_sequence = 1

# processed the noisy signals to obtain fixed size vectors for each instance
# truncated the signals to size of minimum length signal


def resize(sig, size):
    return scipy_signal.resample(sig, size, t=None, axis=0, window=None)


avg_signal_len = 0


def process_data(signals, labels):
    global avg_signal_len

    # randomizer = rSeq.SignalRandomizer().add_samples_in_big_jumps()
    # randomizer.build()
    # print("Randomizing signals...")
    # for i, signal in enumerate(signals):
    #   signals[i] = randomizer.randomize(signal)

    min = 1000
    print("Calculating average length...")

    sizeSum = sum((len(i) for i in signals))
    avgLen = round(sizeSum / len(signals))
    print("Average signal length:", avgLen)
    # avg_signal_len = 2342
    avg_signal_len = 1928

    print("Resizing signals to", avg_signal_len, "...")
    resizer = rSeq.SignalRandomizer().resize(length=avg_signal_len)
    resizer.build()
    for i, signal in enumerate(signals):
        signals[i] = resizer.randomize(signal)
        labels[i] = resizer.randomize(labels[i])


avg_signal_len2 = 0


def process_data2(signals):
    global avg_signal_len2

    # randomizer = rSeq.SignalRandomizer().add_samples_in_big_jumps()
    # randomizer.build()
    # print("Randomizing signals...")
    # for i, signal in enumerate(signals):
    #   signals[i] = randomizer.randomize(signal)

    min = 1000
    print("Calculating average length...")

    sizeSum = sum((len(i) for i in signals))
    avgLen = round(sizeSum / len(signals))
    print("Average signal length:", avgLen)
    avg_signal_len2 = avgLen
    # avg_signal_len = avgLen

    print("Resizing signals to", avg_signal_len2, "...")
    resizer = rSeq.SignalRandomizer().resize(length=avg_signal_len2)
    resizer.build()
    for i, signal in enumerate(signals):
        signals[i] = resizer.randomize(signal)
        # labels[i] = resizer.randomize(labels[i])


start = time.monotonic()
signals, index, spacer_labels, letter_labels, barcode_labels = prepare_train2(
    data_file_name, samples_per_sequence)
end = time.monotonic()
print(
    f"Generated {len(signals)} signals and {len(spacer_labels)} labels", f"in {end-start}")
#Sequential: 26.099690708000026
#Parallel: 102.181611471

# import scipy.io
# scipy.io.savemat('test.mat', {'mydata': signals})


plt.figure(figsize=(30, 10))
i = 15
plt.plot(signals[i])
plt.plot(spacer_labels[i])
print("Letters:", letter_labels[i], "barcodes:", barcode_labels[i])
plt.savefig(os.path.join(folder_dir, 'pics/pic1.png'))

# print("Cleaning up dataset...")
process_data(signals, spacer_labels)  # process the dataset

X_data = np.array(signals)
y_data = np.array(spacer_labels)

# torch.save(X_data,'./data/50k_full_signals_X_data_temp.pt')
# torch.save(y_data,'./data/50k_full_signals_y_data_temp.pt')
# torch.save(letter_labels,'./data/50k_full_signals_letter_labels_temp.pt')
# torch.save(barcode_labels,'./data/50k_full_signals_barcode_labels_temp.pt')

# X_data = torch.load('./data/50k_full_signals_X_data_temp.pt')
# y_data = torch.load('./data/50k_full_signals_y_data_temp.pt')
# letter_labels = torch.load('./data/50k_full_signals_letter_labels_temp.pt')
# barcode_labels = torch.load('./data/50k_full_signals_barcode_labels_temp.pt')


##########################################################
#################### PYTORCH #############################
##########################################################

# from scipy.io import loadmat
# #matlabfile=loadmat('/content/drive/Shareddrives/SRP-Synthetic-DNA-Storage/SRP-Test-Data/fixing1.mat')
# matlabfile=loadmat('/content/drive/Shareddrives/SRP-Synthetic-DNA-Storage/New Alphabet Real Data/For_ML/new_alpha_letter.mat')

# matlabfile['data'].shape[0]
# all_labels=[]
# for i in range(matlabfile['data'].shape[0]):
#   for j in range(len(matlabfile['data'][i][0])):
#     all_labels.append(list(matlabfile['data'][i][0][j] -1))
# print(len(all_labels))
# list_test_signals=[]
# for i in range(matlabfile['data'].shape[0]):
#   for j in range(len(matlabfile['data'][i][1])):
#     list_test_signals.append(np.squeeze(list(matlabfile['data'][i][1][j])[0]))
# print(len(list_test_signals))

# len(list_test_signals), len(all_labels), all_labels[10], plt.plot(list_test_signals[1]), min(all_labels), max(all_labels)

# np.array(all_labels,dtype=np.uint8)

# plt.plot(list_test_signals[0])

avg_signal_len
# all_labels

if avg_signal_len == 0:
    assert len(X_data[0]) == len(X_data[1]) == len(
        X_data[-1]) == len(X_data[-2]) == 1928
    avg_signal_len = len(X_data[0])

# for i,v in enumerate(list_test_signals):
#   list_test_signals[i] = resize(v, avg_signal_len)

# test_signals=torch.tensor(list_test_signals)
# test_labels=torch.from_numpy(np.array(all_labels,dtype=np.uint8)).squeeze().long()
# print(test_signals.shape)
# print(test_labels.shape)

# Set device to GPU_indx if GPU is avaliable
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
device

# splitting into train and test dataset


X_train, X_validation, y_train, y_validation = train_test_split(
    X_data, y_data, test_size=0.15)


#y_train_1hot=to_categorical(y_train, num_classes=alphabet)
#y_test_1hot=to_categorical(y_test, num_classes=alphabet)

# initialise what epoch we start from
start_epoch = 0
# initialise best valid accuracy
best_valid_acc = 0
# where to load/save the dataset from
data_set_root = "/content/drive/Shared drives/SRP-Synthetic-DNA-Storage"

# Create a "SignalDataset" class by importing the Pytorch Dataset class


class SignalDataset(Dataset):
    """ Data noisey sinewave dataset
        num_datapoints - the number of datapoints you want
    """

    def __init__(self, x, y):
        self.x_data = torch.unsqueeze(torch.tensor(x), 1)
        self.y_data = torch.tensor(y)

    # called by the dataLOADER class whenever it wants a new mini-batch
    # returns corresponding input datapoints AND the corresponding labels
    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index]

    # length of the dataset
    def __len__(self):
        return self.x_data.shape[0]


class SignalDataset2(Dataset):
    """ Data noisey sinewave dataset
        num_datapoints - the number of datapoints you want
    """

    def __init__(self, x, y, letter_labels):
        self.x_data = torch.unsqueeze(torch.tensor(x), 1)
        self.y_data = torch.tensor(y)
        self.letter_labels = torch.tensor(letter_labels)

    # called by the dataLOADER class whenever it wants a new mini-batch
    # returns corresponding input datapoints AND the corresponding labels
    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index], self.letter_labels[index, :]

    # length of the dataset
    def __len__(self):
        return self.x_data.shape[0]

# #load test data from matlab file
# def loadSignalsMat(path):
#   from scipy.io import loadmat
#   annots = loadmat(path)

#   testSignals = []

#   for _, v in enumerate(annots['curr_alph'][0]):
#     sig = v[0][0][0][0]
#     testSignals.append(sig)

#   return testSignals


# X_test = loadSignalsMat("/content/drive/Shared drives/SRP-Synthetic-DNA-Storage/" + "SRP-Test-Data/FLAT_SPC_TEST_SET_F256_BUFFERED.mat")
# y_test = [i for i in range(len(X_test))]


# for i, v in enumerate(X_test):
#   X_test[i] = resize(v, avg_signal_len)

# X_test = torch.tensor(X_test)
# y_test = torch.tensor(y_test)
# X_test.shape

batch_size = 16

dataset_train = SignalDataset(X_train, y_train)
dataset_validation = SignalDataset(X_validation, y_validation)

# TODO: DELETE THIS LINE IF YOU GET 0%
# TODO: DELETE THIS LINE IF YOU GET 0%
# TODO: DELETE THIS LINE IF YOU GET 0%
# TODO: DELETE THIS LINE IF YOU GET 0%
# TODO: DELETE THIS LINE IF YOU GET 0%
#test_labels -= 1

# dataset_test = SignalDataset(test_signals, test_labels)

# Now we need to pass the dataSET to the Pytorch dataLOADER class along with some other arguments
# batch_size - the size of our mini-batches
# shuffle - whether or not we want to shuffle the dataset
data_loader_train = DataLoader(
    dataset=dataset_train, batch_size=batch_size, shuffle=True)
data_loader_validation = DataLoader(
    dataset=dataset_validation, batch_size=batch_size, shuffle=False)
# data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

DL_iter = iter(data_loader_train)
data, labels = next(DL_iter)
print("Input data shape:", data.shape)
print("Labels shape:", labels.shape)
# data.float() # float32 otherwise it si float64
labels.dtype


class TempConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        # Call the __init__ function of the parent nn.module class
        super().__init__()
        # how to handle channel width change
        self.conv1 = nn.Conv1d(channels_in, channels_out,
                               kernel_size=4, stride=1, padding=0, dilation=1)
        # maxpool retain the number of channels only change dims
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x


class DilatedTempConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        # Call the __init__ function of the parent nn.module class
        super().__init__()
        # how to handle channel width change
        self.conv1 = nn.Conv1d(channels_in, channels_out,
                               kernel_size=4, stride=1, padding=0, dilation=2)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x

# We will use the above blocks to create a "Deep" neural network with many layers!


class CNN_LSTM(nn.Module):
    def __init__(self, dim_out=2, layer_type1=DilatedTempConv, layer_type2=TempConv, num_blocks1=1, num_blocks2=1):
        # Call the __init__ function of the parent nn.module class
        super().__init__()

        # dilated
        # num_blocks can only be >1 if in_dims = out_dims
        self.layers11 = self.create_blocks(
            num_blocks=num_blocks1, block_type=layer_type1, channels_in=1, channels_out=8)
        # self.layers12 = self.create_blocks(num_blocks=num_blocks1, block_type=layer_type1, channels_in = 8, channels_out=16)

        # non_dilated
        self.layers21 = self.create_blocks(
            num_blocks=num_blocks2, block_type=layer_type2, channels_in=8, channels_out=16)
        self.layers22 = self.create_blocks(
            num_blocks=num_blocks2, block_type=layer_type2, channels_in=16, channels_out=32)
        # self.layers23 = self.create_blocks(num_blocks=num_blocks2, block_type=layer_type2, channels_in = 64, channels_out=128)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=5,
                               stride=1, padding=0, dilation=1)

        self.bilstm = nn.LSTM(input_size=232, hidden_size=(1928+6) //
                              2, num_layers=2, batch_first=True, bidirectional=True)  # 16,64,304

        self.conv2 = nn.Conv1d(64, 64, kernel_size=4,  # 16,128,302
                               stride=1, padding=0, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=1)  # 16,128,300
        # self.dim_out=dim_out
        self.fc1 = nn.Linear(64, 32)

        self.fc2 = nn.Linear(32, dim_out)

    def create_blocks(self, num_blocks, block_type, channels_in, channels_out):
        blocks = []

        # We will add some number of the res/skip blocks!
        for _ in range(num_blocks):
            blocks.append(block_type(channels_in, channels_out))

        return nn.Sequential(*blocks)

    def forward(self, x):

        x = self.layers11(x)
        # x = self.layers12(x)
        x = self.layers21(x)
        x = self.layers22(x)
        # x = self.layers23(x)
        x = self.conv1(x)  # -> torch.Size([16, 256, 5]) # B,C,D
        # print('conv1', x.shape)
        x, _ = self.bilstm(x)  # 16,64,300

        # # only take the out
        # # x = x.flatten(1)
        # print('bilstm', x.shape)

        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        # print('bilstm_conv2', x.shape)
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        # print('fc1', x.shape)
        x = self.fc2(x)
        return x.permute(0, 2, 1)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# create model
# model = CNN_Filter().to(device)
model = CNN_LSTM().to(device)

# Initialize the optimizer with above parameters
optimizer = optim.Adam(model.parameters())

# Define the loss function
loss_fn = nn.CrossEntropyLoss()  # cross entropy
# loss_fn = nn.L1Loss()  # Mean Absolute Error
# loss_fn = nn.MSELoss()  # Mean Squared Error

data, label = next(iter(data_loader_train))
out = model(data.float().to(device))
print('input_shape:', data.shape)

print('label_shape:', label.shape)
print('output_shape:', out.shape)

print('output_argmax_CE_shapep:', out.argmax(1).shape)
# plt.figure(figsize=(20,10))
# plt.plot(out.detach().cpu()[0])
# plt.plot(label[0])

# pytorch.onnx.export(model, data, 'model.onnx')

print(len(data_loader_train.dataset))  # how many datapoint in trainset
print(len(data_loader_train))  # how many batches in trainset

start_from_checkpoint = True
best_valid_acc

# Create Save Path from save_dir and model_name, we will save and load our checkpoint here
#model_name = 'spacer_5samples'
save_path = os.path.join(save_dir, model_name + ".pt")
best_valid_acc = 0


def load_checkpoint():
    global start_epoch
    global best_valid_acc
    print('load checkpoint from:', save_path)
    # Create the save directory if it does note exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load Checkpoint
    if start_from_checkpoint:
        # Check if checkpoint exists
        if os.path.isfile(save_path):
            # load Checkpoint
            check_point = torch.load(save_path)
            # Checkpoint is saved as a python dictionary
            # https://www.w3schools.com/python/python_dictionaries.asp
            # here we unpack the dictionary to get our previous training states
            model.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
            start_epoch = check_point['epoch']
            best_valid_acc = check_point['valid_acc']
            print("Checkpoint loaded, starting from epoch:", start_epoch)
        else:
            # Raise Error if it does not exist
            # raise ValueError("Checkpoint Does not exist")
            print("Checkpoint Does not exist for", save_path)
    else:
        # If checkpoint does exist and Start_From_Checkpoint = False
        # Raise an error to prevent accidental overwriting
        count = 0
        while os.path.isfile(save_path):
            # raise ValueError("Warning Checkpoint exists")
            print("check_point already exist, create new save path")
            save_path = os.path.join(
                path_to_model, model_name + "COPY" + str(count)+".pt")
            training_loss_logger = []
            validation_acc_logger = []
            training_acc_logger = []
            count += 1
        else:
            print("Starting from scratch")


best_valid_acc
print(len(data_loader_train.dataset))
print(data_loader_train.dataset.x_data.shape)

data, label = next(iter(data_loader_train))
data = data.float().to(device)
out = model(data)

# This function should perform a single training epoch using our training data


def train(net, device, loader, optimizer, loss_fun, loss_logger):

    # Set Network in train mode
    net.train()

    for i, (x, y) in enumerate(tqdm(loader)):
        # Forward pass of image through network and get output
        x = x.float()
        fx = net(x.to(device))
        y = y.long()
        # Calculate loss using loss function
        to_dev = y.to(device)
        loss = loss_fun(fx, to_dev)

        # Zero Gradents
        optimizer.zero_grad()
        # Backpropagate Gradents
        loss.backward()
        # Do a single optimization step
        optimizer.step()

        # log the loss for plotting
        loss_logger.append(loss.item())

    # return the avaerage loss and acc from the epoch as well as the logger array
    return loss_logger

# This function should perform a single evaluation epoch, it WILL NOT be used to train our model


def evaluate(net, device, loader):

    # initialise counter
    epoch_acc = 0

    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values
    # (More on these layer types soon!)
    net.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            # Forward pass of image through network
            x = x.float()
            fx = net(x.to(device))
            y = y.long()

            # fx

            # log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

    # return the accuracy from the epoch
    return epoch_acc / (len(loader.dataset) * y.shape[1])

# train !!!!!!!!!!!!!!!!!!!!!!!!!!


def train_multi_epochs(wandb_username, training_loss_logger, validation_acc_logger, training_acc_logger, loss_fun):
    global start_epoch
    global best_valid_acc
    # settings=wandb.Settings(symlink=False)
    wandb.init(entity=wandb_username, project='SRP', name=model_name, config={
               "epochs": num_epochs, "batch_size": 16, "start_epoch": start_epoch})
    wandb.watch(model, loss_fun, log="all", log_freq=10)
    wandb.save(os.path.join(wandb.run.dir, "spacer_detect1.onnx"))

    for epoch in range(start_epoch, num_epochs):
        print('starting_epoch:', start_epoch)
        print('current_best_valid_acc', best_valid_acc)
        # call the training function and pass training dataloader etc
        training_loss_logger = train(
            model, device, data_loader_train, optimizer, loss_fn, training_loss_logger)

        # call the evaluate function and pass the dataloader for both ailidation and training
        train_acc = evaluate(model, device, data_loader_train)
        valid_acc = evaluate(model, device, data_loader_validation)
        validation_acc_logger.append(valid_acc)
        training_acc_logger.append(train_acc)

        # If this model has the highest performace on the validation set
        # then save a checkpoint
        # {} define a dictionary, each entry of the dictionary is indexed with a string
        if (valid_acc > best_valid_acc):
            best_valid_acc = valid_acc
            print("Saving Model...")
            torch.save({
                'epoch':                 epoch,
                'model_state_dict':      model.state_dict(),
                'optimizer_state_dict':  optimizer.state_dict(),
                'train_acc':             train_acc,
                'valid_acc':             valid_acc,
            }, save_path)
            # torch.onnx.export(model,data,"spacer_detect1.onnx")
            # wandb.save("spacer_detect1.onnx")
            wandb.save(os.path.join(wandb.run.dir, "spacer_detect1.onnx"))

        # clear_output(True)
        print(
            f'| Epoch: {epoch+1:02} | Train Acc: {train_acc*100:05.2f}% | Val. Acc: {valid_acc*100:05.2f}% |')
        wandb.log({"epoch": epoch, "train_acc": train_acc *
                  100, "valid_acc": valid_acc*100})
    return training_loss_logger, validation_acc_logger, training_acc_logger


if skip_training == False:
    # This cell implements our training loop
    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []
    train_multi_epochs(entity, training_loss_logger,
                       validation_acc_logger, training_acc_logger, loss_fn)

print("Training Complete")


def evaluate2(net, device, loader):

    # initialise counter
    epoch_acc = 0

    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values
    # (More on these layer types soon!)
    net.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            # Forward pass of image through network
            x = x.float()  # [:,:,0:-2]
            # y=y[:,0:-2]
            fx = net(x.to(device))

            # log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

    # return the accuracy from the epoch
    acc = epoch_acc/(len(loader.dataset)*y.shape[1])
    return [fx, y, acc]


def evaluate3(net, device, loader, max_count=15):

    # initialise counter
    epoch_acc = 0

    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values
    # (More on these layer types soon!)
    net.eval()
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            count += 1
            # Forward pass of image through network
            x = x.float()
            fx = net(x.to(device))

            # log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

            if count == max_count:
                break

    # return the accuracy from the epoch
    acc = epoch_acc/(16*count*y.shape[1])
    return [fx, y, acc]


print(model_name)

# load model in pytorch
# check the dictionary for what epoch
a = torch.load('./Models/'+model_name+'.pt')['model_state_dict']
# a=torch.load('./Nucleotrace/Alphabet_Classifier/alphabet_classifier_model.pt')['model_state_dict'] # check the dictionary for what epoch
# print(a) # use this orderdict like a dictionary

model_pytorch = CNN_LSTM().to(device)  # using the best model
model_pytorch.load_state_dict(a)
model_pytorch.eval()
start = time.time()


fx_check, y_check, acc = evaluate2(
    model_pytorch, device, data_loader_validation)

# end = time.time()
# print('max acc: ',acc)
# print(f"Time: {end - start:.3f} s")
# print(f"Speed: {len(data_loader_validation.dataset)/(end - start):.3f} signals/s")
# print('label: ',y_check)
# print('model output: ', fx_check.argmax(1))

# this model has the best weights

torch.load('./Models/'+model_name+'.pt')['model_state_dict']

# torch.onnx.export(model_pytorch,data.to(device).float(),"model.onnx")
# wandb.save("model.onnx")


def findOneRegion(spacer):
    regions = []  # a list of tuples (start:end)

    currStart = 0
    currEnd = 0
    isInRegion = False

    for i in range(len(spacer)):
        if spacer[i] == 1:
            if not isInRegion:
                isInRegion = True
                currStart = i
                currEnd = i

            currEnd += 1

        elif isInRegion:
            regions.append((currStart, currEnd))
            isInRegion = False

    if isInRegion:
        regions.append((currStart, currEnd))

    sortedRegions = sorted(
        regions, key=lambda x: x[1] - x[0], reverse=True)  # already sorted
    return sortedRegions


print('trained_model acc:', acc)
print('fx_check_argmax shape:', fx_check.argmax(1).shape)
print('y_check shape ( should be similar to fx_argmax):', y_check.shape)
print('fx signal dimension shape: ', fx_check[0][0].shape)
plt.figure(figsize=(30, 10))

print('output_argmax shape', fx_check.argmax(1).shape)

plt.plot(fx_check.argmax(1)[0].detach().cpu())
plt.plot(y_check[0])
plt.savefig(os.path.join(folder_dir, 'pics/pic2.png'))

a = np.zeros((1, len(y_check[0]))).squeeze()
# print(a)
print('len of regions of one (spacers detected): ',
      len(findOneRegion(fx_check.argmax(1)[0])))
print('len of regions of one (spacers detected) _ label: ',
      len(findOneRegion(y_check[0])))
print('list regions of one: ', findOneRegion(fx_check.argmax(1)[0]))
print('list regions of one _ label: ', findOneRegion(y_check[0]))
# np.array(findOneRegion(y_check)).shape


def extract_letter(net, device, loader, num_spacer):

    # initialise counter
    epoch_acc = 0
    letters = []
    letters_y = []
    letters_label = []

    not_enough_spacers_count = 0

    # Set network in evaluation mode
    net.eval()

    with torch.no_grad():
        # x of shape 16,1,1928 y of shape 16,1928 z of shape 16,7
        for i, (x, y, z) in enumerate(tqdm(loader)):
            # Forward pass of signal through network

            x = x.float()

            fx = net(x.to(device))

            for j in range(x.shape[0]):  # bs
                # should be a list of tuple (start, end)
                one_regions = findOneRegion(fx.argmax(1)[j])[:num_spacer]
                # print(one_regions)
                # print(len(one_regions))
                if len(one_regions) < num_spacer:
                    not_enough_spacers_count += 1
                    continue

                # should be a list of tuple
                one_regions_y = findOneRegion(y[j])
                # print(len(one_regions),len(one_regions_y))
                assert len(one_regions) == len(one_regions_y)

                one_regions = sorted(one_regions, key=lambda x: x[0])
                one_regions_y = sorted(one_regions_y, key=lambda x: x[0])

                zero_regions = []
                zero_regions_y = []
                for i in range(len(one_regions)-1):
                    zero_regions.append(
                        (one_regions[i][1]+1, one_regions[i+1][0]))
                    zero_regions_y.append(
                        (one_regions_y[i][1]+1, one_regions_y[i+1][0]))

                assert len(zero_regions) == 7, zero_regions_y == 7

                letters.extend([x[j][0][zero_regions[k][0]:zero_regions[k][1]]
                               for k in range(len(zero_regions))])
                letters_label.extend(z[j])
                letters_y.extend([x[j][0][zero_regions_y[k][0]:zero_regions_y[k][1]]
                                 for k in range(len(zero_regions_y))])

                # print([x[j][0][zero_regions[k][0]:zero_regions[k][1]] for k in range(len(zero_regions))])
                # print(z[j])
                # break

                # print(np.array(one_regions).shape)
                # print(np.array(one_regions_y).shape)

                # print(x.shape)
                # print(x[j].shape)
                # print(letters)
                # print(len(letters))
                # print(len(letters[0]))
            # return letters, letters_y

            # log the cumulative sum of the acc
            # epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()
            # break
    # return the accuracy from the epoch
    # acc = epoch_acc/(len(loader.dataset)*y.shape[1])
    print(f"Found {not_enough_spacers_count} signals with less than 8 spacers")
    return letters, letters_y, letters_label

# if (not os.path.exists('/content/drive/Shareddrives/SRP-Synthetic-DNA-Storage/d2s2_val.pt')):
#   letters,letters_y=extract_letter(model_pytorch, device, data_loader_validation, 19)
#   len(letters[0])
#   len(letters_y[0])
#   torch.save([letters,letters_y],'d2s2_val.pt')


print('current dir:', os.getcwd())

dataset_all = SignalDataset2(X_data, y_data, letter_labels)
data_loader_all = DataLoader(
    dataset=dataset_all, batch_size=batch_size, shuffle=False)

x, y, z = next(iter(data_loader_all))
print(x.shape)
print(y.shape)
print(z.shape)
print(z)

# if (not os.path.exists(os.path.join(folder_dir, 'letters_extracted.pt'))):
letters, letters_y, letters_label_filtered = extract_letter(
    model_pytorch, device, data_loader_all, 8)
print('done')
torch.save([letters, letters_y, letters_label_filtered],
           'letters_extracted.pt')
print('saving extracted letter in:', os.path.join(
    folder_dir, 'letters_extracted.pt'))
# else:
#     print('already exist extracted letter in',
#           os.path.join(folder_dir, 'letters_extracted.pt'))
#     [letters, letters_y, letters_label_filtered] = torch.load(
#         'letters_extracted.pt')


print(len(letters))
print(len(letters_y))
print(len(letters_label_filtered))
# letters_label_filtered

avg_letter = round(sum(len(letter)
                   for letter in letters) / len(letters))  # should be 70
avg_letter = 152  # to match the existing classifier
letters_resize = []
letters_resize_label_filtered = []
letters_y_resize = []
for l, label, l_y in zip(letters, letters_label_filtered, letters_y):
    if len(l) > avg_letter/2:
        l_resize = resize(l, avg_letter)

        letters_resize.append(l_resize)
        letters_y_resize.append(resize(l_y, avg_letter))
        letters_resize_label_filtered.append(label)

print(f"{len(letters)} -> {len(letters_resize)}")

print(len(letters_resize))
print(len(letters_y_resize))
print(len(letters_resize_label_filtered))
# letters_label_filtered
len(letters_resize[0])

letter_dataset = SignalDataset(letters_resize, letters_resize_label_filtered)
letter_loader = DataLoader(dataset=letter_dataset,
                           batch_size=batch_size, shuffle=False)

l, label = next(iter(letter_loader))
print(l.shape)
print(label.shape)

torch.save(letter_loader, 'letter7loader_152_10k.pt')

# print(len(letters))
# print(len(letters[0])) # letters 0 is a list of letter signal
# print(letters[0])

print(len(data_loader_validation.dataset))
print(len(letters))  # letters should be an array of 7 letters
print(data_loader_validation.dataset.x_data.shape)
print(len(letters[0]))
# print(letters[0])
# print(letters_y[0])

letters_list = []
for x in letters:
    if len(x) != 7:
        print(x)

    for y in x:
        letters_list.append(y)

print(len(letters_list))
print(len(letter_labels))

# letters_validation  = SignalDataset(X_letters,y_letters)
# letters_loader_validation = DataLoader(dataset=letters_validation, batch_size=batch_size, shuffle=False)

"""
threshold must be in the range [0, 1).
It represent the minimum probability that the best class should have. 
"""


def evaluateThreshold(net, device, loader, threshold):
    assert threshold >= 0 and threshold < 1, "threshold must be >= 0 and < 1"
    # initialise counter
    correct = 0
    discarded = 0
    false_positives = 0
    false_negatives = 0
    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values
    # (More on these layer types soon!)
    net.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            # Forward pass of image through network
            x = x.float()

            fx = net(x.to(device))

            max_probs = fx.softmax(1).max(1)
            probs = max_probs.values.cpu().numpy()
            indices = max_probs.indices.cpu().numpy()

            # print("topk:", torch.topk(fx, 3))
            # fx[i][argmax[i]]
            for k in range(len(probs)):
                if probs[k] > threshold:
                    if y[k] == indices[k]:
                        correct += 1
                    else:
                        false_positives += 1
                else:
                    discarded += 1
                    if y[k] == indices[k]:
                        false_negatives += 1

            # log the cumulative sum of the acc
            # epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

    # return the accuracy from the epoch
    print("Dataset size:", len(loader.dataset))
    print(f"{correct + false_positives} or {(correct + false_positives)/len(loader.dataset)*100:.2f}% letters had a probability higher than threshold.")
    print(f"{correct} or {correct/len(loader.dataset)*100:.2f}% letters were classified correctly.")
    print(f"{false_positives} or {false_positives/len(loader.dataset)*100:.2f}% letters were false positives.")
    print(f"{discarded} or {discarded/len(loader.dataset)*100:.2f}% letters had a probability less than threshold.")
    print(f"{discarded - false_negatives} or {(discarded - false_negatives)/len(loader.dataset)*100:.2f}% letters were discarded correctly.")
    print(f"{false_negatives} or {false_negatives/len(loader.dataset)*100:.2f}% letters were discarded incorrectly (false negatives).")

    acc = correct/len(loader.dataset)
    acc_over_t = correct/(correct + false_positives)

    fp_acc = false_positives/len(loader.dataset)
    fn_acc = false_negatives/len(loader.dataset)

    return fx, y, acc, acc_over_t, fp_acc, fn_acc


"""
difference must be in the range [0, 1).
It represent the probability difference that there must be between the first and second classs. 
"""


def evaluateTop2Difference(net, device, loader, difference):
    assert threshold >= 0 and threshold < 1, "threshold must be >= 0 and < 1"
    # initialise counter
    correct = 0
    discarded = 0
    false_positives = 0
    false_negatives = 0
    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values
    # (More on these layer types soon!)
    net.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            # Forward pass of image through network
            x = x.float()

            fx = net(x.to(device))

            top2 = torch.topk(fx, 2, 1)

            probs = top2.values.softmax(1).cpu().numpy()

            indices = top2.indices.cpu().numpy()

            # fx[i][argmax[i]]
            for k in range(len(probs)):
                if probs[k][0] - probs[k][1] > difference:
                    if y[k] == indices[k][0]:
                        correct += 1
                    else:
                        false_positives += 1
                else:
                    discarded += 1
                    if y[k] == indices[k][0]:
                        false_negatives += 1

            # TODO:Delete this

            # log the cumulative sum of the acc
            # epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

    # return the accuracy from the epoch
    print("Dataset size:", len(loader.dataset))
    print(f"{correct + false_positives} or {(correct + false_positives)/len(loader.dataset)*100:.2f}% letters had a probability higher than threshold.")
    print(f"{correct} or {correct/len(loader.dataset)*100:.2f}% letters were classified correctly.")
    print(f"{false_positives} or {false_positives/len(loader.dataset)*100:.2f}% letters were false positives.")
    print(f"{discarded} or {discarded/len(loader.dataset)*100:.2f}% letters had a probability less than threshold.")
    print(f"{discarded - false_negatives} or {(discarded - false_negatives)/len(loader.dataset)*100:.2f}% letters were discarded correctly.")
    print(f"{false_negatives} or {false_negatives/len(loader.dataset)*100:.2f}% letters were discarded incorrectly (false negatives).")

    acc = correct/len(loader.dataset)
    acc_over_t = correct/(correct + false_positives)

    fp_acc = false_positives/len(loader.dataset)
    fn_acc = false_negatives/len(loader.dataset)

    return fx, y, acc, acc_over_t, fp_acc, fn_acc


fx_check, y_check, acc = evaluate2(model, device, data_loader_validation)

fx_check.argmax(1)

y_check

acc

# # plot loss function
# plt.figure(figsize=(10, 10))
# train_x = np.linspace(0, num_epochs, len(training_loss_logger))
# plt.plot(train_x, training_loss_logger, c="y")

# # plot accuracy
# plt.figure(figsize=(10, 10))
# test_x = np.linspace(0, num_epochs, len(validation_acc_logger))
# plt.plot(test_x, validation_acc_logger, c="k")

# print("Maximum validation accuracy:",
#       f"{max(validation_acc_logger) * 100:.3f}%")


# plot_confusion_matrix_img(test_confusion_matrix, normalize=False)

# plot_confusion_matrix(valid_confusion_matrix, [
#                       i for i in range(256)], title="train")
