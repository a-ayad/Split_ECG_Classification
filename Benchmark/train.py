import utils
from models.fastai_model import fastai_model
import struct
import socket
import pickle
import json
from torch.optim import SGD, Adam, AdamW
import sys
import time
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import string
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics
import sklearn.preprocessing
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, hamming_loss
import pandas as pd
import wfdb
import ast
import time
#np.set_printoptions(threshold=np.inf)

epoch = 50
lr = 0.01
batchsize = 64
batch_concat = 1
autoencoder = 0
detailed_output = 0
count_flops = 0
plots = 0
autoencoder_train = 0
deactivate_train_after_num_epochs = 0

# load data from json file
#Dataset import class:

class PTB_XL_train(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        global X_train
        global y_train
        self.y_train = y_train
        self.X_train = X_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):

        sample = self.X_train[idx].transpose((1, 0)), self.y_train[idx]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class PTB_XL_val():
    def __init__(self, transform=None):
        self.transform = transform
        global y_val
        global X_val
        self.y_val = y_val
        self.X_val = X_val

    def __len__(self):
        return len(y_val)

    def __getitem__(self, idx):

        sample = self.X_val[idx].transpose((1, 0)),  self.y_val[idx]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

def init():
    train_dataset = PTB_XL_train()
    val_dataset = PTB_XL_val()
    global train_loader
    global val_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            norm_type='bachnorm'
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size, #padding=1 #to match the AE input/output
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        #x = self.dropout1(x)

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        #x = self.dropout2(x)

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        #x = self.dropout3(x)

        #x = self.pool(x)
        return x


class ConvNormPool1(nn.Module):
    """Conv Skip-connection module"""

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            norm_type='bachnorm'
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size // 4,
            kernel_size=kernel_size, #padding=1 #to match the AE input/output
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size // 4,
            out_channels=hidden_size // 2,
            kernel_size=kernel_size,
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size // 2,
            out_channels=hidden_size,
            kernel_size=kernel_size,
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size // 4
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size // 2
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size // 4)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size // 2)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv3)#conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)
        return x


class ConvNormPool2(nn.Module):
    """Conv Skip-connection module"""

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            norm_type='bachnorm'
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv2d(
            in_channels=input_size,
            out_channels=hidden_size // 2,
            kernel_size=kernel_size,  # padding=1 #to match the AE input/output
        )
        self.conv_2 = nn.Conv2d(
            in_channels=hidden_size // 2,
            out_channels=hidden_size // 4,
            kernel_size=kernel_size,
        )
        self.conv_3 = nn.Conv2d(
            in_channels=hidden_size // 4,
            out_channels=hidden_size // 8,
            kernel_size=kernel_size,
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size // 2
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size // 4
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size // 8
            )
        else:
            self.normalization_1 = nn.BatchNorm2d(num_features=hidden_size // 2)
            self.normalization_2 = nn.BatchNorm2d(num_features=hidden_size // 4)
            self.normalization_3 = nn.BatchNorm2d(num_features=hidden_size // 8)

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv3)#conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)
        return x


class Client(nn.Module):
    def __init__(
            self,
            input_size=1,
            hid_size=128,
            kernel_size=5,
            num_classes=5,
    ):
        super().__init__()

        self.conv1 = ConvNormPool1(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool2(
            input_size=hid_size,
            hidden_size=hid_size*2,
            kernel_size=kernel_size,
        )

    def forward(self, input):
        return input


class Encode(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode, self).__init__()
        self.conva = nn.Conv1d(128, 64, 2, stride=2,  padding=0)
        self.convb = nn.Conv1d(64, 16, 2, stride=2, padding=0)
        self.convc = nn.Conv1d(16, 8, 2, stride=2,  padding=0)
        self.convd = nn.Conv1d(8, 4, 2, stride=1, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        x = self.convb(x)
        x = self.convc(x)
        x = self.convd(x)
        return x

class Decode(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode, self).__init__()
        self.t_conva = nn.ConvTranspose1d(4, 8, 2, stride=1)
        self.t_convb = nn.ConvTranspose1d(8, 16, 2, stride=2)
        self.t_convc = nn.ConvTranspose1d(16, 64, 2, stride=2)
        self.t_convd = nn.ConvTranspose1d(64, 128, 2, stride=2)

    def forward(self, x):
        x = self.t_conva(x)
        x = self.t_convb(x)
        x = self.t_convc(x)
        x = self.t_convd(x)
        return x

class FullModel(nn.Module):
    """
    client model
    """
    def __init__(self):
        super(FullModel, self).__init__()
        self.drop0 = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv1d(12, 128, 3, 1)
        self.norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv4 = nn.Conv1d(128, 256, 3, 1)
        self.norm4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        #self.pool4 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.drop4 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(256, 128, 3, 1)
        self.norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.drop2 = nn.Dropout(p=0.5)
        #self.conv3 = nn.Conv1d(128, 128, 3, 1)
        #self.norm3 = nn.BatchNorm1d(128)
        #self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1)
        #self.drop3 = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.linear1 = nn.Linear(in_features=128, out_features=5, bias=True)

    def forward(self, x):
        x = self.drop0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        #print("input pool1: ", x.shape)
        x = self.pool1(x)
        #print("output pool1: ", x.shape)
        #x = self.drop1(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        #x = self.pool4(x)
        #x = self.drop4(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        #x = self.pool2(x)
        #x = self.drop2(x)
        #x = self.conv3(x)
        #x = self.norm3(x)
        #x = self.relu3(x)
        #x = self.pool3(x)
        #x = self.drop3(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = torch.softmax(self.linear1(x), dim=1)
        return x


class FullModel1(nn.Module):
    def __init__(
            self,
            input_size=12,
            hid_size=128,
            kernel_size=5,
            num_classes=5,
    ):
        super().__init__()

        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size, out_features=num_classes)

    def forward(self, input):
        #print("conv1 input: ", input.shape)
        x = self.drop1(input)
        x = self.conv1(x)
        #print("conv1 output: ", x.shape)
        x = self.drop2(x)
        x = self.conv2(x)
        #print("conv2 output: ", x.shape)
        x = self.drop3(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        x = self.avgpool(x)
        # print(x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        x = torch.sigmoid(self.fc(x))
        return x


def str_to_number(label):
    a = np.zeros(5)
    if not label:
        return a
    for i in label:
        if i == 'NORM':
            a[0] = 1
        if i == 'MI':
            a[1] = 1
        if i == 'STTC':
            a[2] = 1
        if i == 'HYP':
            a[3] = 1
        if i == 'CD':
            a[4] = 1
    return a





def train():
    """
    actuall function, which does the training from the train loader and
    testing from the testloader epoch/batch wise
    :param s: socket
    """
    train_losses = []
    train_accs = []
    total_f1 = []
    test_losses = []
    test_accs = []
    data_send_per_epoch_total = []
    data_recieved_per_epoch_total = []
    flops_client_forward_total = []
    flops_client_encoder_total = []
    flops_client_backprob_total = []
    time_train_test = 0
    train_losses.append(0)
    train_accs.append(0)
    test_losses.append(0)
    test_accs.append(0)
    total_f1.append(0)

    # Specify AE configuration
    train_active = 0  # default: AE is pretrained
    if autoencoder_train:
        train_active = 1

    start_time_training = time.time()
    for e in range(epoch):
        print(f"Starting epoch: {e}/{epoch}")
        if e >= deactivate_train_after_num_epochs:  # condition to stop AE training
            train_active = 0  # AE training off
        if e == 1:
            print("estimated_time_total: ", time_train_test * epoch / 60, " min")

        #train_set, val_set = torch.utils.data.random_split(dataset, [18000, 3837])
        #print("train_set: ", train_set.size)

        # add comments
        train_loss = 0.0
        correct_train, total_train = 0, 0
        loss_test = 0.0
        correct_test, total_test = 0, 0
        active_training_time_epoch_client = 0
        active_training_time_epoch_server = 0

        concat_counter_send = 0
        concat_counter_recv = 0

        batches_aborted, total_train_nr, total_test_nr = 0, 0, 0

        concat_tensors = []
        concat_labels = []
        epoch_start_time = time.time()

        recieve_time_total = 0
        send_time_total = 0
        recieve_and_backprop_total = 0
        forward_time_total = 0
        time_batches_total = 0
        time_for = time.time()
        time_for_bed_total = 0
        time_for_bed = 0

        meter = Meter()
        meter.init_metrics()
        num_batches = 0
        hamming_epoch = 0
        accuracy_epoch = 0
        precision_epoch = 0
        recall_epoch = 0
        f1_epoch = 0
        exact_match_epoch = 0
        oloss_epoch = 0


        for b, batch in enumerate(train_loader):
            if time_for_bed != 0:
                time_for_bed_total += time.time() - time_for_bed
            inputs, label = batch
            inputs = inputs.to(device)#DoubleTensor(inputs)
            if b == 0:
                print("inputs shape: ", inputs.shape)
            #label = label.to(device)
            #print("inputs: ", inputs.shape) # shape (64, 1000, 12)
            #print("label: ", label) # list of labels
            #print("inputs: ", inputs)
            #label_iterator = map(float, label)
            #new_label = []
            #for l in label[0]:
            #    a = str_to_number(l)
            #    new_label.append(a)
            #print(label_iterator)
            #label = list(label_iterator)
            #print("label: ", new_label)
            #label = torch.LongTensor(label).to(device)
            #label.to(device)
            #Sprint("labels: ", label)
            label = label.double().to(device)
            #print(label)
            #label = float(label)

            #optimizer_full_model.zero_grad()
            #output = full_model(inputs)
            #loss = error(output, label)
            #loss.backward()
            #optimizer_full_model.step()
            #train_loss += loss

            optimizer_model.zero_grad()
            output = model(inputs)
            loss = error(output, label)
            loss.backward()
            optimizer_model.step()
            train_loss += loss

            #print("output[0]: ", output[0])
            #print("label[0]: ", label[0])
            #label_binarizer = sklearn.preprocessing.LabelBinarizer()
            #label_metric = label_binarizer.transform(label.detach().clone().cpu())\
            #label_metric = label.detach().clone().cpu()
            #for l in label:
            #    for lm in label_metric:
            #        lm = np.argmax(l.cpu())
            #print("label_metric[0]: ", label_metric[0])

            #output_metric = output.detach().clone().cpu()#np.argmax(output.detach().clone().cpu(), axis=1)
            #print("output_metric[0]: ", output_metric[0])
            num_batches += 1
            output = torch.round(output)
            #if np.sum(label.cpu().detach().numpy()[0]) > 1:
            #    if np.sum(output.cpu().detach().numpy()[0] > 1):
            #        print("output[0]: ", output.cpu().detach().numpy()[0])
            #        print("label [0]: ", label.cpu().detach().numpy()[0])
            #if (b % 100 == 0):
                #print("output[0]: ", output.cpu().detach().numpy()[0])
                #print("label [0]: ", label.cpu().detach().numpy()[0])

            #oloss_epoch = np.any(label.detach().clone().cpu() != output.detach().clone().cpu(), axis=1).mean()
            #exact_match_epoch = np.all(label.detach().clone().cpu() == output.detach().clone().cpu(), axis=1).mean()
            hamming_epoch += hamming_score(label.detach().clone().cpu(), output.detach().clone().cpu()).item()
            #accuracy_epoch += Accuracy(label.detach().clone().cpu(), output.detach().clone().cpu()).item()
            #print('Hamming Loss: {0}'.format(Hamming_Loss(label.detach().clone().cpu(), output.detach().clone().cpu())))
            precision_epoch += Precision(label.detach().clone().cpu(), output.detach().clone().cpu()).item()
            #recall_epoch += Recall(label.detach().clone().cpu(), output.detach().clone().cpu()).item()
            f1_epoch += F1Measure(label.detach().clone().cpu(), output.detach().clone().cpu()).item()
            #print('Exact Match Ratio: {0}'.format(sklearn.metrics.accuracy_score(label_metric, output_metric, normalize=True, sample_weight=None)))
            #print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(label_metric, output_metric)))
            #print('Recall: {0}'.format(sklearn.metrics.precision_score(y_true=label_metric, y_pred=output_metric, average='samples')))
            #print('Precision: {0}'.format(sklearn.metrics.recall_score(y_true=label_metric, y_pred=output_metric, average='samples')))
            #print('F1 Measure: {0}'.format(sklearn.metrics.f1_score(y_true=label_metric, y_pred=output_metric, average='samples')))

            total_train_nr += 1
            #if b % 200 == 0:
            #    print("batch: ", b ," / ", total_batch)
            #meter.update(output, label, loss)
        """
        #metrics = meter.get_metrics()
        #metrics = {k: v / b for k, v in metrics.items()}
        #df_logs = pd.DataFrame([metrics])
        #confusion_matrix = meter.get_confusion_matrix()

        #print('{}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
              )
        if e % 11 == 10:
            fig, ax = plt.subplots(figsize=(5, 5))
            cm_ = ax.imshow(confusion_matrix, cmap='hot')
            ax.set_title('Confusion matrix', fontsize=15)
            ax.set_xlabel('Actual', fontsize=13)
            ax.set_ylabel('Predicted', fontsize=13)
            plt.colorbar(cm_)
            plt.show()
        """

        epoch_endtime = time.time() - epoch_start_time
        status_epoch_train = "epoch: {}, hamming_epoch: {:.4f}, precision_epoch: {:.4f}, recall_epoch: {:.4f}, f1_epoch: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}, train_loss: {:.4f}  ".format(
            e+1, hamming_epoch / total_train_nr, precision_epoch / total_train_nr, recall_epoch / total_train_nr,
                f1_epoch / total_train_nr, epoch_endtime, batches_aborted / total_train_nr, train_loss / total_train_nr)
        print("status_epoch_train: ", status_epoch_train)


        time_for = time.time() - time_for
        epoch_endtime = time.time() - epoch_start_time
        test_time = time.time()
        total_val_nr = 0
        hamming_epoch = 0
        precision_epoch = 0
        recall_epoch = 0
        f1_epoch = 0
        val_loss_total = 0.0
        with torch.no_grad():
            for b_t, batch_t in enumerate(val_loader):
                inputs, label = batch_t
                inputs = torch.DoubleTensor(inputs).to(device)
                # label = label.to(device)
                # print("inputs: ", inputs.shape) # shape (64, 1000, 12)
                # print("label: ", label) # list of labels
                # print("inputs: ", inputs)
                # label_iterator = map(str_to_number, label)
                # print(label_iterator)
                # label = list(label_iterator)
                #print("label: ", new_label)
                #label = label.double().to(device)
                #val_output = full_model(inputs)
                #val_loss = error(val_output, label)
                #val_loss_total += val_loss

                val_output = model(inputs)
                val_loss = error(val_output, label)
                val_loss_total += val_loss


                output_val_server = torch.round(val_output)
                total_val_nr += 1
                # meter_val.update(output_val_server, label_val, loss_val / total_val_nr)
                hamming_epoch += hamming_score(label.detach().clone().cpu(),
                                                     output_val_server.detach().clone().cpu()).item()
                precision_epoch += Precision(label.detach().clone().cpu(),
                                                   output_val_server.detach().clone().cpu()).item()
                #recall_epoch += Recall(label.detach().clone().cpu(),
                #                             output_val_server.detach().clone().cpu()).item()
                f1_epoch += F1Measure(label.detach().clone().cpu(),
                                            output_val_server.detach().clone().cpu()).item()
                #meter.update(val_output, label, val_loss)


        #total_f1.append(metrics['f1'])

        test_time = time.time() - test_time
        # print("test time: ", test_time)
        time_train_test = test_time + epoch_endtime

        #wandb.log({"epoch": e + 1})
        #wandb.log({"loss": metrics['loss']})
        #wandb.log({"test_acc": metrics['accuracy']})
        #wandb.log({"f1": metrics['f1']})
        #wandb.watch(full_model)
        status_epoch_val = "epoch: {}, hamming_epoch: {:.4f}, precision_epoch: {:.4f}, recall_epoch: {:.4f}, f1_epoch: {:.4f}, val_loss: {:.4f}".format(
            e+1, hamming_epoch / total_val_nr, precision_epoch / total_val_nr, recall_epoch / total_val_nr,
                   f1_epoch / total_val_nr, val_loss_total / total_val_nr)
        print("status_epoch_val: ", status_epoch_val)

        #status_epoch = "epoch: {}, train-loss: {:.4f}, train-acc: {:.2f}%, test-loss: {:.4f}, test-acc: {:.2f}%, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}  ".format(
        #    e + 1, train_loss / total_train_nr, (correct_train / total_train) * 100, loss_test / total_test_nr,
        #    (correct_test / total_test) * 100, epoch_endtime, batches_aborted / total_train_nr)
        #print("status_epoch: ", status_epoch)

    total_training_time = time.time() - start_time_training
    time_info = "trainingtime for {} epochs: {:.2f}min".format(epoch, total_training_time / 60)
    print("time_info: ", time_info)
    data_transfer_per_epoch = 0
    for data in data_send_per_epoch_total:
        data_transfer_per_epoch += data
    for data in data_recieved_per_epoch_total:
        data_transfer_per_epoch += data
    print("Average data transfer/epoch: ", data_transfer_per_epoch / epoch / 1000000, " MB")
    total_flops_forward = 0
    total_flops_encoder = 0
    total_flops_backprob = 0
    for flop in flops_client_forward_total:
        total_flops_forward += flop[0]
    for flop in flops_client_encoder_total:
        total_flops_encoder += flop[0]
    for flop in flops_client_backprob_total:
        total_flops_backprob += flop[0]
    print("total FLOPs forward: ", total_flops_forward)
    print("total FLOPs encoder: ", total_flops_encoder)
    print("total FLOPs backprob: ", total_flops_backprob)
    print("total FLOPs client: ", total_flops_backprob + total_flops_encoder + total_flops_forward)
    # plot(test_accs, train_accs, train_losses, test_losses, total_f1)
    plt.show()


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]



def Precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return temp / y_true.shape[0]


def Recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return temp / y_true.shape[0]

def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]


class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))

    def update(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        self.metrics['accuracy'] += accuracy_score(x, y)
        self.metrics['f1'] += f1_score(x, y, average='macro')
        self.metrics['precision'] += precision_score(x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x, y, average='macro', zero_division=1)

        self._compute_cm(x, y)

    def _compute_cm(self, x, y):
        for prob, target in zip(x, y):
            if prob == target:
                self.confusion[target][target] += 1
            else:
                self.confusion[target][prob] += 1

    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        self.metrics['precision'] = 0
        self.metrics['recall'] = 0

    def get_metrics(self):
        return self.metrics

    def get_confusion_matrix(self):
        return self.confusion


def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """
    #global config
    #config = Config()
    global X_train
    global X_val

    global y_val
    sampling_frequency = 100
    datafolder = 'C:/Users/maria/PycharmProjects/Medical-MESL-Debug/Medical-Dataset/Normal/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    task = 'superdiagnostic'
    outputfolder = 'C:/Users/maria/PycharmProjects/Medical-MESL-Debug/Medical-Dataset/Normal/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/output/'

    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)
    input_shape = data[0].shape
    print(input_shape)

    # 1-9 for training
    X_train = data[labels.strat_fold < 10]
    global y_train
    y_train = Y[labels.strat_fold < 10]
    # 10 for validation
    X_val = data[labels.strat_fold == 10]
    y_val = Y[labels.strat_fold == 10]

    num_classes = 5  # <=== number of classes in the finetuning dataset
    input_shape = [1000, 12]  # <=== shape of samples, [None, 12] in case of different lengths

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    import pickle

    standard_scaler = pickle.load(open('C:/Users/maria/PycharmProjects/PTB-XL/standard_scaler.pkl', "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    init()
    #


    experiment = 'exp0'
    modelname = 'fastai_xresnet1d101'
    pretrainedfolder = 'C:/Users/maria/PycharmProjects/PTB-XL/fastai_xresnet1d101'
    mpath = 'C:/Users/maria/PycharmProjects/PTB-XL/output/'  # <=== path where the finetuned model will be stored
    n_classes_pretrained = 71  # <=== because we load the model from exp0, this should be fixed because this depends the experiment
    global model
    model = fastai_model(
        modelname,
        num_classes,
        sampling_frequency,
        mpath,
        input_shape=input_shape,
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained,
        pretrained=True,
        epochs_finetuning=2,
    )
    #if torch.cuda.is_available():
    #    model.to('cuda')

    global optimizer_model
    # optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)
    #optimizer_model = AdamW(model.parameters(), lr=lr)
    model.fit(X_train, y_train, X_val, y_val)
    y_val_pred = model.predict(X_val)
    utils.evaluate_experiment(y_val, y_val_pred)

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if (torch.cuda.is_available()):
        print("training on gpu")
    print("training on,", device)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global full_model
    full_model = FullModel()
    full_model.double().to(device)

    global optimizer_full_model
    #optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)
    optimizer_full_model = AdamW(full_model.parameters(), lr=lr)

    global error
    #error = nn.CrossEntropyLoss()
    error = nn.BCELoss()
    #error.double()
    #error.to(device)


    train()


if __name__ == '__main__':
    main()
