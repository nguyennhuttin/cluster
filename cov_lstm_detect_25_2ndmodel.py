

from tqdm import tqdm
import time
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
datasets = {}


# Commented out IPython magic to ensure Python compatibility.

############
# figures from plt allow
matplotlib.use("Agg")
############


# start from a checkpoint or start from scratch?
###### IMPORTANT #######
# setting to False overdrive the checkpoint in /content/drive/Shared drives/SRP-Synthetic-DNA-Storage/Models/model_name.pt


# you can change the model name to create new checkpoint for possibly new model
###### IMPORTANT #######
# the convetion is to use the notebook file name as the model_name

model_name = 'Cov_LSTM_detect_25_2ndmodel.ipynb'
# A directory to save our model to (will create it if it doesn't exist)
save_dir = '/Models'
save_dir_data = '/outputs'
folder_dir = '/mnt/lustre/projects/vs25/Tin'
assert torch.cuda.is_available()

# folder_dir = 'D:/SRP_DNAencoding/Tin'

##########################################
######### ADDING PARSER ##################
##########################################

parser = argparse.ArgumentParser(description="CNN_LSTM")

parser.add_argument("--model_name", default=model_name)
parser.add_argument("--folder_dir", default=folder_dir)
parser.add_argument("--save_dir", default=save_dir,
                    help='Name of the folder where to save Pytoch models')
parser.add_argument("--save_dir_data", default=save_dir_data,
                    help='name of the folder where to store data')

parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--num_epochs", default=100)
parser.add_argument("--batch_size", default=16)

parser.add_argument("--new", action='store_true',
                    help='Start training from epoch 0 when true, from last checkpoint when false.')
# skip_training is false be default
parser.add_argument("--skip_training", action='store_true')

args = parser.parse_args()

model_name = args.model_name
# A directory to save our model to (will create it if it doesn't exist)
save_dir = args.save_dir
save_dir_data = args.save_dir_data
folder_dir = args.folder_dir

# Define the hyperparameters
learning_rate = float(args.learning_rate)
num_epochs = int(args.num_epochs)
batch_size = int(args.batch_size)
start_from_checkpoint = not args.new  # this is True by default
skip_training = args.skip_training  # this is False by default

print(model_name)
print(save_dir, save_dir_data, folder_dir)
print('lr:', learning_rate)
print('n_epochs:', num_epochs)
print('bs:', batch_size)
print('start from checkpoint', start_from_checkpoint)
print('skip_training', skip_training)

##############################################################


data_file_name = os.path.join(
    save_dir_data, 'real_alphabets_F256_pow2.csv')  # F256
print(data_file_name)

upload = open(folder_dir+data_file_name, "r")
print('upload', upload)
# %cd /content/drive/Shared drives/SRP-Synthetic-DNA-Storage
# necessary imports
# from sklearn.utils import shuffle
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils.np_utils import to_categorical


# given the index of the sequence in the POW2 alphabet
# return the two letters that compose this sequence
def letters_from_index(index, alphabetSize=256):
    first = index // alphabetSize
    second = index % alphabetSize
    return first, second

# one hot encode an alphabet index suitable for pytorch


def oneHotEncodeIndex(index, letter, alphabetSize=256):
    first, second = letters_from_index(index, alphabetSize=alphabetSize)

    if first != letter and second != letter:
        return 0
    if first != letter and second == letter:
        return 1
    if first == letter and second != letter:
        return 2
    if first == letter and second == letter:
        return 3

    raise Exception("Impossible case")

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
    data = pd.read_csv(datafile_path, header=None)
    currents = []  # list of all reference currents
    std_dev = []
    dwells = []
    curr = []
    dev = []
    dw = []
    for i in range(2, len(data)):
        df = data.iloc[i]
        if(df[0] == 'pos'):  # change in signal identified by appearance of 'pos' in the 1st column
            currents.append(curr)
            std_dev.append(dev)
            dwells.append(dw)
            curr = []
            dev = []
            dw = []
        elif(str(df[2]) != 'nan'):  # signal same as previous one
            curr.append(float(df[2]))
            dev.append(float(df[3]))
            dw.append(float(df[4]))
    currents.append(curr)
    std_dev.append(dev)
    dwells.append(dw)
    return currents, std_dev, dwells

# generate data (samples per ref)

# 8 + 12 + 8 + 12 + 8 = 48

# 0 for separator, 1 for 25, 2 for everything else


def classifySignal(index, label):
    if label == 0:
        order = [0, 2, 0, 2, 0]
    elif label == 1:
        order = [0, 2, 0, 1, 0]
    elif label == 2:
        order = [0, 1, 0, 2, 0]
    else:
        order = [0, 1, 0, 1, 0]

    boundaries = [7, 19, 27, 39, 47]

    currentBoundary = 0

    result = [0]*len(index)
    j = 0
    for i in index:
        if i > boundaries[currentBoundary]:
            currentBoundary += 1
        result[j] = order[currentBoundary]
        j += 1

    assert len(result) == len(index), f"{len(result)}, {len(index)}"

    return result

# randomize signal and label assuming the signal was generated without any randomisation
# and the sequence is of the shape SEPARATO 12MER [SEPARATOR 12MER ...] SEPARATOR
# Example: AAAAAAAA CCCCCCCCCCCC AAAAAAAA
#            2       9        -10     -3


def randomizeSignalAndLabel(index, signal, label):
    leftRange = [2, 9]
    maxIndex = index[-1]  # assume the signal ends with a separator
    rightRange = [maxIndex - 10, maxIndex - 3]

    indexesInLeftRange = []
    for i in range(len(index)):
        if index[i] < leftRange[0]:
            continue
        if leftRange[0] <= index[i] <= leftRange[1]:
            indexesInLeftRange.append(i)
        else:
            break

    indexesInRightRange = []
    for i in range(len(index) - 1, -1, -1):
        if index[i] > rightRange[1]:
            continue
        if rightRange[0] <= index[i] <= rightRange[1]:
            indexesInRightRange.append(i)
        else:
            break

    left = random.choice(indexesInLeftRange)
    right = random.choice(indexesInRightRange)

    return signal[left:right+1], label[left:right+1]


r = 0.2  # we want to skip 80% of the cases labelled as 0
# generating our noisy dataset by adding noise to reference signals


# 'num_samples_per_reference' refers to number of noisy signals generated from a single reference signal
def generate_data(currents, std_dev, dwells, num_samples_per_reference, letter=25, alphabetSize=256):
    signals = []
    indexes = []
    labels = []
    tolerance = 4
    merLen = 12+8+12
    # 256*256
    for i in range(0, len(currents)):  # 'len(currents)' -> number of reference signals
        seq = ss.ScrappySequence(
            ['A']*len(currents[i]), currents[i], std_dev[i], dwells[i])
        seq = rSeq.removeInitial6As(seq)

        label = oneHotEncodeIndex(i, letter, alphabetSize=alphabetSize)

        luck = random.random()
        if label == 0 and luck > r:
            continue

        print("Sequence", i + 1, "of", len(currents),
              ":", f"{(i+1)/len(currents)*100:.2f}%")

        for j in range(0, num_samples_per_reference):
            #randSeq = rSeq.randomizeSeparatorAsymmetric(seq, 6, 2, 2, 6, merLen=merLen)
            # noisy signal generation for each instance
            signal_s, index_s = generate_samples(
                seq.currents, seq.sds, seq.dwells)
            actualLabel = classifySignal(index_s, label)

            signal_s, actualLabel = randomizeSignalAndLabel(
                index_s, signal_s, actualLabel)

            signals.append(signal_s)
            labels.append(actualLabel)

    return signals, indexes, labels

# generating training data


def prepare_train(dataset_path, num_samples_per_reference):
    if dataset_path in datasets:
        currents, std_dev, dwells = datasets[dataset_path]
    else:
        currents, std_dev, dwells = read_data(
            dataset_path)  # read the reference signals
        datasets[dataset_path] = [currents, std_dev, dwells]

    # generate the noisy dataset from reference signals
    signals, indexes, labels = generate_data(
        currents, std_dev, dwells, num_samples_per_reference)
    return signals, labels, indexes


# extimating dataset size
alphabet = 256
samples_per_sequence = 50
total_samples = alphabet * samples_per_sequence
print("Estimated number of samples:", total_samples)
average_signal_length = 100
total_datapoints = average_signal_length * total_samples
float32_bytes = 4
total_datapoints_bytes = float32_bytes * total_datapoints
print(f"Estimated input data size: {total_datapoints_bytes/1000000} MB")
int_bytes = 4
total_lables_bytes = int_bytes * total_datapoints
print(f"Estimated label size: {total_lables_bytes/1000000} MB")
print(
    f"Estimated Total Dataset size: {(total_lables_bytes + total_datapoints_bytes)/1000000} MB")
# processed the noisy signals to obtain fixed size vectors for each instance
# truncated the signals to size of minimum length signal


def resize(signal, length):
    res = [0]*length
    ratio = len(signal)/length  # > 1
    for i in range(len(res)):
        x = i * ratio
        decimalLeft = x - math.floor(x)
        decimalRight = 1 - decimalLeft
        res[i] = (1 - decimalLeft) * signal[math.floor(x)] + (1 - decimalRight) * \
            signal[math.ceil(x) if math.ceil(
                x) < len(signal) else len(signal) - 1]

    return res


avg_signal_len = 0


def process_data(signals, labels):
    global avg_signal_len

    print("Calculating average length...")

    sizeSum = sum((len(i) for i in signals))
    avgLen = round(sizeSum / len(signals))
    avg_signal_len = avgLen

    print("Resizing signals...")
    resizer = rSeq.SignalRandomizer().resize(length=avgLen)
    resizer.build()
    for i, signal in enumerate(signals):
        signals[i] = resizer.randomize(signal)
        labels[i] = resizer.randomize(labels[i])
        labels[i] = np.round(np.array(labels[i]))
        if i % 3000 == 0:
            print(f"{(i+1)/len(signals):0.2%}")

# generate the dataset
# import time
# start = time.monotonic()
# signals,labels,indexes=prepare_train(data_file_name, samples_per_sequence)
# end = time.monotonic()
# print(f"Generated {len(signals)} signals and {len(labels)} labels", f"in {end-start}")


# save_path_signals = os.path.join("data", model_name + "_signal.npy")
# save_path_labels = os.path.join("data", model_name + "_labels.npy")

#np.save(save_path_signals, np.array(signals))
#np.save(save_path_labels, np.array(labels))

# Accidentally overwitten with processed data :(
# signals = np.load(save_path_signals, allow_pickle=True)
# labels = np.load(save_path_labels, allow_pickle=True)

# print("Cleaning up dataset...")

# process_data(signals, labels) #process the dataset
# for i in range(len(signals)):
#   signals[i] = np.array(signals[i])
#   labels[i] = np.array(labels[i])

# signals=np.vstack(signals).astype(np.float)
# labels=np.vstack(labels).astype(np.float)
# ###########################################################
# ################# Load Processed Signals ##################
# ###########################################################
oldModelName = "Copy of Cov_LSTM.ipynb"
save_path_signals = os.path.join(
    folder_dir + "/data", oldModelName + "_signal.npy")
save_path_labels = os.path.join(
    folder_dir + "/data", oldModelName + "_labels.npy")

#np.save(save_path_signals, signals)
#np.save(save_path_labels, labels)

X_data = np.load(save_path_signals, allow_pickle=True)
y_data = np.load(save_path_labels, allow_pickle=True)

# from scipy import signal
# # y_data is 324 -> downsample 156
# y_data = signal.resample(y_data, 256, t=None, axis=1, window=None)
# y_data.shape

gc.collect()

len(X_data[0]), len(X_data[1]), len(X_data[2]), len(
    X_data[3]), len(X_data[4]), len(X_data[5])

print("Input data shape:", X_data.shape, X_data[0].shape)
print("Label data shape:", y_data.shape, y_data[0].shape)

X_data

##########################################################
#################### PYTORCH #############################
##########################################################

##########################################################
############## LOAD REAL DATA ############################
##########################################################

# matlabfile = loadmat(os.path.join(folder_dir, '/SRP-Test-Data/real_corr.mat'))

# matlabfile['data'].shape[0]
# all_labels = []
# for i in range(matlabfile['data'].shape[0]):
#     for j in range(len(matlabfile['data'][i][0])):
#         all_labels.append(list(matlabfile['data'][i][0][j]))
# print(len(all_labels))
# list_test_signals = []
# for i in range(matlabfile['data'].shape[0]):
#     for j in range(len(matlabfile['data'][i][1])):
#         list_test_signals.append(np.squeeze(
#             list(matlabfile['data'][i][1][j])[0]))
# print(len(list_test_signals))

# plt.plot(list_test_signals[0])

# # np.squeeze(list(matlabfile['data'][0][1][0])[0])
# matlabfile['data'][0][0][0]

# X_data[1000].shape[0]

# for i, v in enumerate(list_test_signals):
#     list_test_signals[i] = resize(v, X_data[0].shape[0])

# test_signals = torch.tensor(list_test_signals)
# test_labels = torch.tensor(all_labels).squeeze().long()
# print(test_signals.shape)
# print(test_labels.shape)


# sequences = ss.parseScrappieOutput(os.path.join(
#     folder_dir, "outputs/real_alphabets_F256.txt"))

# fig, axs = plt.subplots(4, figsize=(10, 10))

# plt.ylim((-2, 2))

# s = sequences[5]

# for j in range(1):
#     seq = rSeq.removeInitial6As(s)
#     signal = rSeq.randomizeSeparatorAsymmetricSignal(seq, 6, 2, 2, 6)
#     signal_time = [i for i in range(len(signal))]
#     axs[0].plot(signal_time, signal, label="Random sig1")

#     signal = rSeq.randomizeSeparatorAsymmetricSignal(seq, 6, 2, 2, 6)
#     signal_time = [i for i in range(len(signal))]
#     axs[1].plot(signal_time, signal, label="Random sig2")

#     signal = rSeq.randomizeSeparatorAsymmetricSignal(seq, 6, 2, 2, 6)
#     signal_time = [i for i in range(len(signal))]
#     axs[2].plot(signal_time, signal, label="Random sig3")

#     #signal = rSeq.removeInitial6As(s).generateExpectedSignal()
#     #signal_time = [i for i in range(len(signal))]
#     axs[3].plot(test_signals[0],  label="Line")

# plt.show()

# Set device to GPU_indx if GPU is avaliable
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')

# splitting into train and test dataset
X_train, X_validation, y_train, y_validation = train_test_split(
    X_data, y_data, test_size=0.15)
#y_train_1hot=to_categorical(y_train, num_classes=alphabet)
#y_test_1hot=to_categorical(y_test, num_classes=alphabet)

del X_data
del y_data

gc.collect()

# initialise what epoch we start from
start_epoch = 0
# initialise best valid accuracy
best_valid_acc = 0
# where to load/save the dataset from
data_set_root = folder_dir

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

# a=np.vstack(a).astype(np.float)

dataset_train = SignalDataset(X_train, y_train)
dataset_validation = SignalDataset(X_validation, y_validation)
#dataset_test = SignalDataset(test_signals, test_labels)

# Now we need to pass the dataSET to the Pytorch dataLOADER class along with some other arguments
# batch_size - the size of our mini-batches
# shuffle - whether or not we want to shuffle the dataset
data_loader_train = DataLoader(
    dataset=dataset_train, batch_size=batch_size, shuffle=True)
data_loader_validation = DataLoader(
    dataset=dataset_validation, batch_size=batch_size, shuffle=False)
#data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

X_train_shape_1 = X_train.shape[1]

del X_train
del y_train
del X_validation
del y_validation

gc.collect()

DL_iter = iter(data_loader_train)
data, labels = next(DL_iter)
print("Input data shape:", data.shape)
print("Labels shape:", labels.shape)
# data.float() # float32 otherwise it si float64
labels.dtype

# DL_iter=iter(data_loader_test)
# data,labels=next(DL_iter)
# print("Input data shape:", data.shape)
# print("Labels shape:", labels.shape)
# # data.float() # float32 otherwise it si float64
# labels.dtype


class TempConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        # Call the __init__ function of the parent nn.module class
        super().__init__()
        # how to handle channel width change
        self.conv1 = nn.Conv1d(channels_in, channels_out,
                               kernel_size=2, stride=1, padding=0, dilation=1)
        # maxpool retain the number of channels only change dims
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

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
                               kernel_size=2, stride=1, padding=0, dilation=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x


# We will use the above blocks to create a "Deep" neural network with many layers!
class CNN_LSTM(nn.Module):
    def __init__(self, dim_out, layer_type1=DilatedTempConv, layer_type2=TempConv, num_blocks1=1, num_blocks2=1):
        # Call the __init__ function of the parent nn.module class
        super().__init__()

        # dilated
        # num_blocks can only be >1 if in_dims = out_dims
        self.layers11 = self.create_blocks(
            num_blocks=num_blocks1, block_type=layer_type1, channels_in=1, channels_out=32)
        # self.layers12 = self.create_blocks(num_blocks=num_blocks1, block_type=layer_type1, channels_in = 8, channels_out=16)

        # non_dilated
        self.layers21 = self.create_blocks(
            num_blocks=num_blocks2, block_type=layer_type2, channels_in=32, channels_out=64)
        self.layers22 = self.create_blocks(
            num_blocks=num_blocks2, block_type=layer_type2, channels_in=64, channels_out=128)
        # self.layers23 = self.create_blocks(num_blocks=num_blocks2, block_type=layer_type2, channels_in = 64, channels_out=128)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=3,
                               stride=1, padding=0, dilation=1)
        self.bilstm = nn.LSTM(input_size=36, hidden_size=316 //
                              2, num_layers=2, batch_first=True, bidirectional=True)

        # self.dim_out=dim_out
        self.fc1 = nn.Linear(256, 128)

        self.fc2 = nn.Linear(128, dim_out)

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
        x, _ = self.bilstm(x)

        # # 1 direction
        # # 1LSTM-> torch.Size([16, 256, 5]) torch.Size([1, 16, 5])torch.Size([1, 16, 5])
        # # 2LSTM -> torch.Size([16, 256, 5]) torch.Size([2, 16, 5]) torch.Size([2, 16, 5])

        # # with bidirectional
        # # 1LSTM-> torch.Size([16, 256, 10]) torch.Size([2, 16, 5]) torch.Size([2, 16, 5]) # out,h,c
        # # 2LSTM-> torch.Size([16, 256, 10]) torch.Size([4, 16, 5]) torch.Size([4, 16, 5])
        # # only take the out
        # # x = x.flatten(1) #-> 16,2560
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 2, 1)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
# create model
# dim_out is the downsampled version of indexes
model = CNN_LSTM(dim_out=3).to(device)

# Initialize the optimizer with above parameters
optimizer = optim.Adam(model.parameters())

# Define the loss function
loss_fn = nn.CrossEntropyLoss()  # cross entropy

data, label = next(iter(data_loader_train))
a = model(data.float().to(device))
print("checking model data, label, output shape")
print(data.shape)
print(label.shape)
print(a.shape)


print(len(data_loader_train.dataset))  # how many datapoint in trainset
print(len(data_loader_train))  # how many batches in trainset

# Create Save Path from save_dir and model_name, we will save and load our checkpoint here
save_path = os.path.join(folder_dir+save_dir, model_name + ".pt")

# Create the save directory if it does note exist
if not os.path.isdir(folder_dir+save_dir):
    os.makedirs(folder_dir+save_dir)

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
        print("Checkpoint Does not exist")
else:
    # If checkpoint does exist and Start_From_Checkpoint = False
    # Raise an error to prevent accidental overwriting
    count = 0
    while os.path.isfile(save_path):
        # raise ValueError("Warning Checkpoint exists")
        print("check_point already exist, create new save path")
        save_path = os.path.join(
            folder_dir+save_dir, model_name + "COPY" + str(count)+".pt")
        training_loss_logger = []
        validation_acc_logger = []
        training_acc_logger = []
        count += 1

    else:
        print("Starting from scratch")

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
            # log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

    # return the accuracy from the epoch
    return epoch_acc / (len(loader.dataset)*316)


training_loss_logger = []
validation_acc_logger = []
training_acc_logger = []

# This cell implements our training loop
if skip_training == False:
    print("start training")
    for epoch in range(start_epoch, num_epochs):

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
            print("Saving Model")
            torch.save({
                'epoch':                 epoch+1,
                'model_state_dict':      model.state_dict(),
                'optimizer_state_dict':  optimizer.state_dict(),
                'train_acc':             train_acc,
                'valid_acc':             valid_acc,
                'training_loss_logger': training_loss_logger,
                'training_acc_logger': training_acc_logger,
                'validation_acc_logger': validation_acc_logger,
            }, save_path)

        print(
            f'| Epoch: {epoch+1:02} | Train Acc: {train_acc*100:05.2f}% | Val. Acc: {valid_acc*100:05.2f}% |')

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
            x = x.float()
            fx = net(x.to(device))
            y = y.long()
            # log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

    # return the accuracy from the epoch
    acc = epoch_acc / (len(loader.dataset)*316)
    return [fx, y, acc]


def evaluate3(net, device, loader, max_count=1):

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
            y = y.long()
            # log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()

            if count == max_count:
                break

    # return the accuracy from the epoch
    acc = epoch_acc/(16*count*316)
    return [fx, y, acc]


#############
#### PLOT ###
#############

# # plot loss function
# plt.figure(figsize=(10, 10))
# train_x = np.linspace(0, num_epochs, len(training_loss_logger))
# plt.savefig(os.path.join(folder_dir, "figures/train_loss.png"))
# # plt.plot(train_x, training_loss_logger, c = "y")

# # plot accuracy
# plt.figure(figsize=(10, 10))
# test_x = np.linspace(0, num_epochs, len(validation_acc_logger))
# # plt.plot(test_x, validation_acc_logger, c = "k")
# plt.savefig(os.path.join(folder_dir, "figures/validation_acc.png"))
# print("Maximum validation accuracy:",
#       f"{max(validation_acc_logger) * 100:.3f}%")


##################################################################
############### LOAD AND CHECK TRAINED MODEL #####################
############### NOT USED FOR NOW #################################
##################################################################

# for name, param in shallow_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

a = torch.load('./Models/'+model_name+'.pt')
dir(a)

# load model in pytorch
# check the dictionary for what epoch
a = torch.load('./Models/'+model_name+'.pt')['model_state_dict']
# print(a) # use this orderdict like a dictionary

model_pytorch = CNN_LSTM(dim_out=3).to(device)  # using the best model
model_pytorch.load_state_dict(a)
start = time.time()
model_pytorch.eval()
fx_check, y_check, acc = evaluate2(
    model_pytorch, device, data_loader_validation)
end = time.time()
print(save_path, os.path.isfile(save_path))
print('eval time:', end-start)
print('max acc: ', acc)
print('max acc percentage: ', acc*100, '%')
print('label: ', y_check)
print('model output: ', fx_check.argmax(1))
# this model has the best weights
# max acc:  0.00014076576576576576

# def all_predict(loader,net):
#     with torch.no_grad():
#         all_predicts=torch.Tensor([]).to(device)
#         all_labels=torch.Tensor([]).to(device)
#         for i, (image, label) in enumerate(loader):
#             #Forward pass of image through network
#             image=image.to(device).float()
#             label=label.long().to(device)
#             out = net(image)
#             score, predict = torch.max(out,1)
#             all_predicts=torch.cat((all_predicts, predict))
#             all_labels=torch.cat((all_labels, label))

#     return all_predicts.cpu(), all_labels.cpu()

# valid_predict_all, valid_label_all=all_predict(data_loader_validation,model_pytorch)
# test_predict_all, test_label_all=all_predict(data_loader_test,model_pytorch)
# train_predict_all, train_label_all=all_predict(data_loader_train,model_pytorch)

# valid_confusion_matrix = confusion_matrix(valid_label_all,valid_predict_all)
# test_confusion_matrix = confusion_matrix(test_label_all,test_predict_all)
# train_confusion_matrix = confusion_matrix(train_label_all,train_predict_all)

# print("Confusion matrix shape:", test_confusion_matrix.shape)
# test_confusion_matrix

# # plot confusioni matrix

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     #print(cm)
#     plt.figure(figsize=(12,12))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
