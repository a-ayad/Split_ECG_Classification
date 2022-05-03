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
#import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import Metrics
import wfdb
import ast
import math
import os.path
import utils
import Models
#np.set_printoptions(threshold=np.inf)
cwd = os.path.dirname(os.path.abspath(__file__))
mlb_path = os.path.join(cwd, "..","Benchmark", "output", "mlb.pkl")
scaler_path = os.path.join(cwd, "..","Benchmark", "output", "standard_scaler.pkl")
ptb_path = os.path.join(cwd, "..", "server", "../server/PTB-XL", "ptb-xl/")

import wandb

wandb.init(project="non-IID,clean", entity="split-learning-medical")
client_num = 1
num_classes = 2
pretrain_this_client = 0
simultrain_this_client = 0
pretrain_epochs = 50
IID = 0

f = open('parameter_client.json', )
data = json.load(f)

# set parameters fron json file
#epoch = data["training_epochs"]
lr = data["learningrate"]
batchsize = data["batchsize"]
batch_concat = data["batch_concat"]
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
autoencoder = data["autoencoder"]
detailed_output = data["detailed_output"]
count_flops = data["count_flops"]
plots = data["plots"]
autoencoder_train = data["autoencoder_train"]
deactivate_train_after_num_epochs = data["deactivate_train_after_num_epochs"]
grad_encode = data["grad_encode"]
train_gradAE_active = data["train_gradAE_active"]
deactivate_grad_train_after_num_epochs = data["deactivate_grad_train_after_num_epochs"]

wandb.init(config={
  "learning_rate": lr,
  #"epochs": epoch,
  "batch_size": batchsize,
    "autoencoder": autoencoder
})

wandb.config.update({"learning_rate": lr, "PC: ": 2})


def print_json():
    print("learningrate: ", lr)
    print("grad_encode: ", grad_encode)
    print("gradAE_train: ", train_gradAE_active)
    print("deactivate_grad_train_after_num_epochs: ", deactivate_grad_train_after_num_epochs)
    #print("Getting the metadata epoch: ", epoch)
    print("Getting the metadata host: ", host)
    print("Getting the metadata port: ", port)
    print("Getting the metadata batchsize: ", batchsize)
    print("Autoencoder: ", autoencoder)
    print("detailed_output: ", detailed_output)
    print("count_flops: ", count_flops)
    print("plots: ", plots)
    print("autoencoder_train: ", autoencoder_train)
    print("deactivate_train_after_num_epochs: ", deactivate_train_after_num_epochs)

# load data from json file
class PTB_XL(Dataset):
    def __init__(self, stage=None):
        self.stage = stage
        if self.stage == 'train':
            global X_train
            global y_train
            self.y_train = y_train
            self.X_train = X_train
        if self.stage == 'val':
            global y_val
            global X_val
            self.y_val = y_val
            self.X_val = X_val
        if self.stage == 'test':
            global y_test
            global X_test
            self.y_test = y_test
            self.X_test = X_test
        if self.stage == 'raw':
            global y_raw
            global X_raw
            self.y_raw = y_raw
            self.X_raw = X_raw

    def __len__(self):
        if self.stage == 'train':
            return len(self.y_train)
        if self.stage == 'val':
            return len(self.y_val)
        if self.stage == 'test':
            return len(self.y_test)
        if self.stage == 'raw':
            return len(self.y_raw)

    def __getitem__(self, idx):
        if self.stage == 'train':
            sample = self.X_train[idx].transpose((1, 0)), self.y_train[idx]
        if self.stage == 'val':
            sample = self.X_val[idx].transpose((1, 0)), self.y_val[idx]
        if self.stage == 'test':
            sample = self.X_test[idx].transpose((1, 0)), self.y_test[idx]
        if self.stage == 'raw':
            sample = self.X_raw[idx].transpose((1, 0)), self.y_raw[idx]
        return sample




def init():
    train_dataset = PTB_XL('train')
    val_dataset = PTB_XL('val')
    if IID:
        train_1, rest1 = torch.utils.data.random_split(train_dataset, [3853, 15414], generator=torch.Generator().manual_seed(42))
        train_2, rest2 = torch.utils.data.random_split(rest1, [3853, 11561], generator=torch.Generator().manual_seed(42))
        train_3, rest3 = torch.utils.data.random_split(rest2, [3853, 7708], generator=torch.Generator().manual_seed(42))
        train_4, train_5 = torch.utils.data.random_split(rest3, [3853, 3855], generator=torch.Generator().manual_seed(42))
        if client_num == 1: train_dataset = train_1
        if client_num == 2: train_dataset = train_2
        if client_num == 3: train_dataset = train_3
        if client_num == 4: train_dataset = train_4
        if client_num == 5: train_dataset = train_5
    if pretrain_this_client:
        raw_dataset = PTB_XL('raw')
        print("len raw dataset", len(raw_dataset))
        pretrain_dataset, no_dataset = torch.utils.data.random_split(raw_dataset, [963, 18304],
                                                       generator=torch.Generator().manual_seed(42))
        print("pretrain_dataset length: ", len(pretrain_dataset))
        global pretrain_loader
        pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batchsize, shuffle=True)

    if simultrain_this_client:
        raw_dataset = PTB_XL('raw')
        print("len raw dataset", len(raw_dataset))
        pretrain_dataset, no_dataset = torch.utils.data.random_split(raw_dataset, [963, 18304],
                                                                     generator=torch.Generator().manual_seed(42))
        print("len train dataset", len(train_dataset))
        train_dataset = torch.utils.data.ConcatDataset((pretrain_dataset, train_dataset))
        print("len mixed-train dataset", len(train_dataset))
    print("train_dataset length: ", len(train_dataset))
    print("val_dataset length: ", len(train_dataset))
    global train_loader
    global val_loader

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
"""
def new_split():
    global train_loader
    global val_loader
    train_dataset, val_dataset = torch.utils.data.random_split(training_dataset,
                                                               [size_train_dataset,
                                                                len(training_dataset) - size_train_dataset])
    print("train_dataset size: ", size_train_dataset)
    print("val_dataset size: ", len(training_dataset) - size_train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
"""

if count_flops: #Does not work on the Jetson Nano yet. The amount of FLOPs doesn't depend on the architecture. Measuring FLOPs on the PC and JetsonNano would result in the same outcome.
    # The paranoid switch prevents the FLOPs count
    # Solution: sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
    # Needs to be done after every restart of the PC
    from ptflops import get_model_complexity_info
    from pypapi import events, papi_high as high


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


#send/recieve system:
def send_msg(sock, getid, content):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = [getid, content]  # add getid
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg  # add 4-byte length in network byte order
    #print("communication overhead send: ", sys.getsizeof(msg), " bytes")
    global data_send_per_epoch
    data_send_per_epoch += sys.getsizeof(msg)
    sock.sendall(msg)


def recieve_msg(sock):
    """
    recieves the meassage with helper function, umpickles the message and separates the getid from the actual massage content
    :param sock: socket
    """

    msg = recv_msg(sock)  # receive client message from socket
    msg = pickle.loads(msg)
    return msg


def recieve_request(sock):
    """
    recieves the meassage with helper function, umpickles the message and separates the getid from the actual massage content
    :param sock: socket
    """

    msg = recv_msg(sock)  # receive client message from socket
    msg = pickle.loads(msg)
    getid = msg[0]
    content = msg[1]
    handle_request(sock, getid, content)


def recv_msg(sock):
    """
    gets the message length (which corresponds to the first
    4 bytes of the recieved bytestream) with the recvall function

    :param sock: socket
    :return: returns the data retrieved from the recvall function
    """
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]

    #print("Message length:", msglen)
    global data_recieved_per_epoch
    data_recieved_per_epoch += msglen
    # read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    """
    returns the data from a recieved bytestream, helper function
    to receive n bytes or return None if EOF is hit
    :param sock: socket
    :param n: length in bytes (number of bytes)
    :return: message
    """
    #
    data = b''
    while len(data) < n:
        if detailed_output:
            print("Start function sock.recv")
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    # print("Daten: ", data)
    return data


def handle_request(sock, getid, content):
    """
    executes the requested function, depending on the get id, and passes the recieved message

    :param sock: socket
    :param getid: id of the function, that should be executed if the message is recieved
    :param content: message content
    """
    #print("request mit id:", getid)
    switcher = {
        0: initialize_model,
        1: train_epoch,
        2: val_stage,
        3: test_stage,
    }
    switcher.get(getid, "invalid request recieved")(sock, content)


def serverHandler(conn):
    while True:
        recieve_request(conn)

def grad_postprocessing(grad):
    grad_new = grad.numpy()
    for a in range(64):
        #scaler.fit(grad[a])
        grad_new[a] = scaler.inverse_transform(grad[a])
    grad_new = torch.DoubleTensor(grad_new).to(device)
    return grad_new


def train_epoch(s, pretraining):
    #new_split() #new random dist between train and val
    loss_grad_total = 0
    global epoch
    epoch += 1
    flops_forward_epoch, flops_encoder_epoch, flops_backprop_epoch, flops_rest, flops_send = 0,0,0,0,0
    #Specify AE configuration
    train_active = 0 #default: AE is pretrained
    train_grad_active = 0
    if epoch < deactivate_train_after_num_epochs:
        if autoencoder_train:
            train_active = 1
    if epoch < deactivate_grad_train_after_num_epochs:
        if train_gradAE_active:
            train_grad_active = 1

    global data_send_per_epoch, data_recieved_per_epoch, data_send_per_epoch_total, data_recieved_per_epoch_total
    data_send_per_epoch, data_recieved_per_epoch = 0, 0
    correct_train, total_train, train_loss = 0, 0, 0
    batches_aborted, total_train_nr, total_val_nr, total_test_nr = 0, 0, 0, 0
    hamming_epoch, precision_epoch, recall_epoch, f1_epoch, auc_train = 0, 0, 0, 0, 0
    #encoder_grad_server = 0

    epoch_start_time = time.time()

    loader = pretrain_loader if pretraining else train_loader

    for b, batch in enumerate(loader):
        if count_flops:
            x = high.read_counters()
        #print("batch: ", b)
        # print("FLOPs dataloader: ", x)
        # if b % 100 == 0:
        # print("batch ", b, " / ", total_batch)

        forward_time = time.time()
        active_training_time_batch_client = 0
        start_time_batch_forward = time.time()

        # define labels and data per batch
        x_train, label_train = batch
        x_train = x_train.to(device)
        # x_train = x_train.to(device)
        label_train = label_train.double().to(device)

        if len(x_train) != 64:
            break

        if count_flops:
            x = high.read_counters()
            flops_rest += x[0] # reset Flop Counter
        optimizer.zero_grad()  # sets gradients to 0 - start for backprop later

        client_output_backprop = client(x_train)
        client_output_train = client_output_backprop.detach().clone()

        if count_flops:
            x = high.read_counters()
            #print("FLOPs forward: ", x)
            flops_forward_epoch += x[0]

        client_output_train_without_ae_send = 0
        if autoencoder:
            if train_active:
                optimizerencode.zero_grad()
            # client_output_train_without_ae = client_output_train.clone().detach().requires_grad_(False)
            client_encoded = encode(client_output_train)
            client_output_send = client_encoded.detach().clone()
            if train_active:
                client_output_train_without_ae_send = client_output_train.detach().clone()
        else:
            client_output_send = client_output_train.detach().clone()
        # client_output_send = encode(client_output_train)

        if count_flops:
            x = high.read_counters()
            flops_encoder_epoch += x[0]


        global encoder_grad_server
        msg = {
            'client_output_train': client_output_send,
            'client_output_train_without_ae': client_output_train_without_ae_send,
            'label_train': label_train,  # concat_labels,
            'batch_concat': batch_concat,
            'batchsize': batchsize,
            'train_active': train_active,
            'encoder_grad_server': encoder_grad_server,
            'train_grad_active': train_grad_active,
            'grad_encode': grad_encode
        }
        active_training_time_batch_client += time.time() - start_time_batch_forward
        if detailed_output:
            print("Send the message to server")
        send_msg(s, 0, msg)


        # while concat_counter_recv < concat_counter_send:
        msg = recieve_msg(s)
        # print("msg: ", msg)
        if pretraining == 0:
            wandb.log({"dropout_threshold": msg["dropout_threshold"]}, commit=False)

        # decode grad:
        client_grad_without_encode = msg["client_grad_without_encode"]
        client_grad = msg["grad_client"]

        global scaler
        scaler = msg["scaler"]
        if msg["grad_encode"]:
            if train_grad_active:
                # print("train_active")
                optimizer_grad_decoder.zero_grad()
            client_grad = Variable(client_grad, requires_grad=True)
            client_grad_decode = grad_decoder(client_grad)
            if train_grad_active:
                loss_grad_autoencoder = error_grad_autoencoder(client_grad_without_encode, client_grad_decode)
                loss_grad_total += loss_grad_autoencoder.item()
                loss_grad_autoencoder.backward()
                encoder_grad_server = client_grad.grad.detach().clone()#
                optimizer_grad_decoder.step()
                # print("loss_grad_autoencoder: ", loss_grad_autoencoder)
            else:
                encoder_grad_server = 0
            client_grad_decode = grad_postprocessing(client_grad_decode.detach().clone().cpu())
        else:
            if msg["client_grad_abort"] == 0:
                client_grad_decode = client_grad.detach().clone()
            #else:
            #    client_grad = "abort"
            encoder_grad_server = 0

        start_time_batch_backward = time.time()

        encoder_grad = msg["encoder_grad"]
        if client_grad == "abort":
            # print("client_grad: ", client_grad)
            train_loss_add, add_correct_train, add_total_train = msg["train_loss"], msg["add_correct_train"], \
                                                                 msg["add_total_train"]
            correct_train += add_correct_train
            total_train_nr += 1
            total_train += add_total_train
            train_loss += train_loss_add
            batches_aborted += 1

            output_train = msg["output_train"]
            # print("train_loss: ", train_loss/total_train_nr)
            # meter.update(output_train, label_train, train_loss/total_train_nr)
            pass
        else:
            if train_active:
                client_encoded.backward(encoder_grad)
                optimizerencode.step()

            # concat_tensors[concat_counter_recv].to(device)
            # concat_tensors[concat_counter_recv].backward(client_grad)
            # client_output_backprob.to(device)
            # if b % 1000 == 999:
            #    print("Backprop with: ", client_grad)
            if count_flops:
                x = high.read_counters() # reset counter
                flops_rest += x[0]
                flops_send += x[0]

            client_output_backprop.backward(client_grad_decode)
            optimizer.step()

            if count_flops:
                x = high.read_counters()
                # print("FLOPs backprob: ", x)
                flops_backprop_epoch += x[0]

            train_loss_add, add_correct_train, add_total_train = msg["train_loss"], msg["add_correct_train"], \
                                                                 msg["add_total_train"]

            correct_train += add_correct_train
            total_train_nr += 1
            total_train += add_total_train
            train_loss += train_loss_add

            output_train = msg["output_train"]
            # print("train_loss: ", train_loss/total_train_nr)
            # meter.update(output_train, label_train, train_loss/total_train_nr)

        # wandb.watch(client, log_freq=100)
        output = torch.round(output_train)
        # if np.sum(label.cpu().detach().numpy()[0]) > 1:
        #    if np.sum(output.cpu().detach().numpy()[0] > 1):
        #        print("output[0]: ", output.cpu().detach().numpy()[0])
        #        print("label [0]: ", label.cpu().detach().numpy()[0])
        #if (total_train_nr % 100 == 0):
        #    print("output[0]: ", output.cpu().detach().numpy()[0])
        #    print("label [0]: ", label_train.cpu().detach().numpy()[0])

        #global batches_abort_rate_total
        #batches_abort_rate_total.append(batches_aborted / total_train_nr)


        active_training_time_batch_client += time.time() - start_time_batch_backward
        #active_training_time_batch_server = msg["active_trtime_batch_server"]
        #active_training_time_epoch_client += active_training_time_batch_client
        #active_training_time_epoch_server += active_training_time_batch_server
        #
        try:
            roc_auc = roc_auc_score(label_train.detach().clone().cpu(), torch.round(output).detach().clone().cpu(),average='micro')
            auc_train += roc_auc
        except:
            # print("auc_train_exception: ")
            # print("label: ", label)
            # print("output: ", output)
            pass

        hamming_epoch += Metrics.Accuracy(label_train.detach().clone().cpu(), torch.round(output).detach().clone().cpu())
        # accuracy_score(label_train.detach().clone().cpu(), torch.round(output).detach().clone().cpu())
        precision_epoch += precision_score(label_train.detach().clone().cpu(),
                                           torch.round(output).detach().clone().cpu(),
                                           average='micro', zero_division=0)
        # recall_epoch += Plots.Recall(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
        recall_epoch += recall_score(label_train.detach().clone().cpu(), torch.round(output).detach().clone().cpu(),
                                     average='micro', zero_division=0)
        # f1_epoch += Plots.F1Measure(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
        f1_epoch += f1_score(label_train.detach().clone().cpu(), torch.round(output).detach().clone().cpu(),
                             average='micro', zero_division=0)

    epoch_endtime = time.time() - epoch_start_time

    if pretraining:
        status_epoch_train = "epoch: {}, AUC_train: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}, train_loss: {:.4f}  ".format(
            epoch, auc_train / total_train_nr, hamming_epoch / total_train_nr, precision_epoch / total_train_nr,
                   recall_epoch / total_train_nr,
                   f1_epoch / total_train_nr, epoch_endtime, batches_aborted / total_train_nr,
                   train_loss / total_train_nr)
        print("status_epoch_pretrain: ", status_epoch_train)

    else:
        flops_client_forward_total.append(flops_forward_epoch)
        flops_client_encoder_total.append(flops_encoder_epoch)
        flops_client_backprop_total.append(flops_backprop_epoch)

        print("data_send_per_epoch: ", data_send_per_epoch / 1000000, " MegaBytes")
        print("data_recieved_per_epoch: ", data_recieved_per_epoch / 1000000, "MegaBytes")
        data_send_per_epoch_total.append(data_send_per_epoch)
        data_recieved_per_epoch_total.append(data_recieved_per_epoch)

        status_epoch_train = "epoch: {}, AUC_train: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}, train_loss: {:.4f}  ".format(
            epoch, auc_train / total_train_nr, hamming_epoch / total_train_nr, precision_epoch / total_train_nr,
                   recall_epoch / total_train_nr,
                   f1_epoch / total_train_nr, epoch_endtime, batches_aborted / total_train_nr,
                   train_loss / total_train_nr)
        print("status_epoch_train: ", status_epoch_train)
        if count_flops:
            print("MegaFLOPS_forward_epoch", flops_forward_epoch / 1000000)
            print("MegaFLOPS_encoder_epoch", flops_encoder_epoch / 1000000)
            print("MegaFLOPS_backprop_epoch", flops_backprop_epoch / 1000000)
            print("MegaFLOPS_rest", flops_rest / 1000000)
            print("MegaFLOPS_send", flops_send / 1000000)

        wandb.log({"Batches Abortrate": batches_aborted / total_train_nr,
                   "MegaFLOPS Client Encoder": flops_encoder_epoch / 1000000,
                   "MegaFLOPS Client Forward": flops_forward_epoch / 1000000,
                   "MegaFLOPS Client Backprop": flops_backprop_epoch / 1000000},
                  commit=False)

        global auc_train_log
        auc_train_log = auc_train / total_train_nr
        global accuracy_train_log
        accuracy_train_log = hamming_epoch / total_train_nr
        global batches_abort_rate_total
        batches_abort_rate_total.append(batches_aborted / total_train_nr)

        initial_weights = client.state_dict()
        send_msg(s, 2, initial_weights)

        msg = 0

        send_msg(s, 3, msg)


def val_stage(s, pretraining=0):
    total_val_nr, val_loss_total, correct_val, total_val = 0, 0, 0, 0
    val_losses, val_accs  = [], []
    hamming_epoch, precision_epoch, recall_epoch, f1_epoch, accuracy, auc_val = 0, 0, 0, 0, 0, 0
    val_time = time.time()
    with torch.no_grad():
        for b_t, batch_t in enumerate(val_loader):

            x_val, label_val = batch_t
            x_val, label_val = x_val.to(device), label_val.double().to(device)
            optimizer.zero_grad()
            output_val = client(x_val, drop=False)
            client_output_val = output_val.clone().detach().requires_grad_(True)
            if autoencoder:
                client_output_val = encode(client_output_val)

            msg = {'client_output_val/test': client_output_val,
                   'label_val/test': label_val,
                   }
            if detailed_output:
                print("The msg is:", msg)
            send_msg(s, 1, msg)
            if detailed_output:
                print("294: send_msg success!")
            msg = recieve_msg(s)
            if detailed_output:
                print("296: recieve_msg success!")
            correct_val_add = msg["correct_val/test"]
            val_loss = msg["val/test_loss"]
            output_val_server = msg["output_val/test_server"]
            val_loss_total += val_loss
            correct_val += correct_val_add
            total_val_add = len(label_val)
            total_val += total_val_add
            total_val_nr += 1

            try:
                roc_auc = roc_auc_score(label_val.detach().clone().cpu(), torch.round(output_val_server).detach().clone().cpu(), average='micro')
                auc_val += roc_auc
            except:
                # print("auc_train_exception: ")
                # print("label: ", label)
                # print("output: ", output)
                pass

            output_val_server = torch.round(output_val_server)
            hamming_epoch += Metrics.Accuracy(label_val.detach().clone().cpu(), output_val_server.detach().clone().cpu())
                #accuracy_score(label_val.detach().clone().cpu(),
                 #                           torch.round(output_val_server).detach().clone().cpu())
            precision_epoch += precision_score(label_val.detach().clone().cpu(),
                                                output_val_server.detach().clone().cpu(),
                                                average='micro', zero_division=0)
            # recall_epoch += Plots.Recall(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
            recall_epoch += recall_score(label_val.detach().clone().cpu(), output_val_server.detach().clone().cpu(),
                                         average='micro', zero_division=0)
            # f1_epoch += Plots.F1Measure(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
            f1_epoch += f1_score(label_val.detach().clone().cpu(), output_val_server.detach().clone().cpu(),
                                 average='micro', zero_division=0)


    status_epoch_val = "epoch: {},AUC_val: {:.4f} ,Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, val_loss: {:.4f}".format(
        epoch, auc_val / total_val_nr, hamming_epoch / total_val_nr, precision_epoch / total_val_nr,
        recall_epoch / total_val_nr,
        f1_epoch / total_val_nr, val_loss_total / total_val_nr)
    print("status_epoch_val: ", status_epoch_val)

    if pretraining == 0:
        wandb.log({"Loss_val": val_loss_total / total_val_nr,
                   "Accuracy_val_micro": hamming_epoch / total_val_nr,
                   "F1_val": f1_epoch / total_val_nr,
                   "AUC_val": auc_val / total_val_nr,
                   "AUC_train": auc_train_log,
                   "Accuracy_train_micro": accuracy_train_log})
        send_msg(s, 3, 0)


def test_stage(s, epoch):
    loss_test = 0.0
    correct_test, total_test = 0, 0
    hamming_epoch = 0
    precision_epoch = 0
    recall_epoch = 0
    f1_epoch = 0
    total_test_nr = 0
    with torch.no_grad():
        for b_t, batch_t in enumerate(val_loader):
            x_test, label_test = batch_t
            x_test, label_test = x_test.to(device), label_test.double().to(device)
            optimizer.zero_grad()
            output_test = client(x_test, drop=False)
            client_output_test = output_test.clone().detach().requires_grad_(True)
            if autoencoder:
                client_output_test = encode(client_output_test)

            msg = {'client_output_val/test': client_output_test,
                   'label_val/test': label_test,
                   }
            if detailed_output:
                print("The msg is:", msg)
            send_msg(s, 1, msg)
            if detailed_output:
                print("294: send_msg success!")
            msg = recieve_msg(s)
            if detailed_output:
                print("296: recieve_msg success!")
            correct_test_add = msg["correct_val/test"]
            test_loss = msg["val/test_loss"]
            output_test_server = msg["output_val/test_server"]
            loss_test += test_loss
            correct_test += correct_test_add
            total_test_add = len(label_test)
            total_test += total_test_add
            total_test_nr += 1

            output_test_server = torch.round(output_test_server)
            hamming_epoch += Metrics.Accuracy(label_test.detach().clone().cpu(), output_test_server.detach().clone().cpu())
                                #accuracy_score(label_test.detach().clone().cpu(),
                              #torch.round(output_test_server).detach().clone().cpu())
            precision_epoch += precision_score(label_test.detach().clone().cpu(),
                                               output_test_server.detach().clone().cpu(),
                                               average='micro')
            # recall_epoch += Plots.Recall(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
            recall_epoch += recall_score(label_test.detach().clone().cpu(),
                                         output_test_server.detach().clone().cpu(),
                                         average='micro')
            # f1_epoch += Plots.F1Measure(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
            f1_epoch += f1_score(label_test.detach().clone().cpu(),
                                 output_test_server.detach().clone().cpu(),
                                 average='micro')

    status_test = "test: hamming_epoch: {:.4f}, precision_epoch: {:.4f}, recall_epoch: {:.4f}, f1_epoch: {:.4f}".format(
        hamming_epoch / total_test_nr, precision_epoch / total_test_nr, recall_epoch / total_test_nr,
        f1_epoch / total_test_nr)
    print("status_test: ", status_test)


    global data_send_per_epoch_total
    global data_recieved_per_epoch_total
    global batches_abort_rate_total


    data_transfer_per_epoch = 0
    average_dismissal_rate = 0
    total_flops_forward = 0
    total_flops_encoder = 0
    total_flops_backprob = 0
    for data in data_send_per_epoch_total:
        data_transfer_per_epoch += data
    for data in data_recieved_per_epoch_total:
        data_transfer_per_epoch += data
    for data in batches_abort_rate_total:
        average_dismissal_rate += data
    for flop in flops_client_forward_total:
        total_flops_forward += flop
    for flop in flops_client_encoder_total:
        total_flops_encoder += flop
    for flop in flops_client_backprop_total:
        total_flops_backprob += flop
    total_flops = total_flops_backprob + total_flops_encoder + total_flops_forward
    print("total FLOPs forward: ", total_flops_forward)
    print("total FLOPs encoder: ", total_flops_encoder)
    print("total FLOPs backprob: ", total_flops_backprob)
    print("total FLOPs client: ", total_flops)
    print("Average data transfer/epoch: ", data_transfer_per_epoch / epoch / 1000000, " MB")
    print("Average dismissal rate: ", average_dismissal_rate / epoch)

    wandb.config.update({"Average data transfer/epoch (MB): ": data_transfer_per_epoch / epoch / 1000000,
                         "Average dismissal rate: ": average_dismissal_rate / epoch,
                         "total_MegaFLOPS_forward": total_flops_forward/1000000, "total_MegaFLOPS_encoder": total_flops_encoder/1000000,
                         "total_MegaFLOPS_backprob": total_flops_backprob/1000000, "total_MegaFLOPS": total_flops/1000000})

    msg = 0
    send_msg(s, 3, msg)


def initialize_model(s, msg):
    """
    if new connected client is not the first connected client,
    the initial weights are fetched from the server
    :param conn:
    """
    #msg = recieve_msg(s)
    if msg == 0:
        #print("msg == 0")
        pass
    else:
        print("msg != 0")
        client.load_state_dict(msg, strict=False)
        print("model successfully initialized")
    #print("start_training")
    # start_training(s)
    #train_epoch(s)


def initIID():
    global X_train, X_val, y_val, y_train, y_test, X_test
    sampling_frequency = 100
    datafolder = ptb_path
    task = 'superdiagnostic'
    outputfolder = mlb_path

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
    y_train = Y[labels.strat_fold < 10]
    # 10 for validation
    X_val = data[labels.strat_fold == 10]
    y_val = Y[labels.strat_fold == 10]

    # X_test = data[labels.strat_fold == 10]
    # y_test = Y[labels.strat_fold == 10]

    num_classes = 5  # <=== number of classes in the finetuning dataset
    input_shape = [1000, 12]  # <=== shape of samples, [None, 12] in case of different lengths

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)  # , X_test.shape, y_test.shape)

    import pickle

    standard_scaler = pickle.load(open(scaler_path, "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)

    global X_raw, y_raw
    X_raw = X_train
    y_raw = y_train


def init_nonIID():
    global X_train, X_val, y_val, y_train, y_test, X_test
    norm, mi, sttc, hyp, cd = [],[],[],[],[]
    for a in range(len(y_train)):
        if label_class(y_train[a], 0):
            sttc.append(X_train[a])
        if label_class(y_train[a], 1):
            hyp.append(X_train[a])
        if label_class(y_train[a], 2):
            mi.append(X_train[a])
        if label_class(y_train[a], 3):
            norm.append(X_train[a])
        if label_class(y_train[a], 4):
            cd.append(X_train[a])

    """
    print("norm shape: ", len(norm))
    print("mi shape: ", len(mi))
    print("sttc shape: ", len(sttc))
    print("hyp shape: ", len(hyp))
    print("cd shape: ", len(cd))

    print("norm label: ", label_norm[0])
    print("mi label: ", label_mi[0])
    print("sttc label: ", label_sttc[0])
    print("hyp label: ", label_hyp[0])
    print("cd label: ", label_cd[0])

    print("norm label: ", len(label_norm))
    print("mi label: ", len(label_mi))
    print("sttc label: ", len(label_sttc))
    print("hyp label: ", len(label_hyp))
    print("cd label: ", len(label_cd))
    """

    if client_num == 1:
        if num_classes == 1:
            print("Client number: ", client_num, " Class norm")
            X_train = norm
            y_train = label_norm
        if num_classes == 2:
            print("Client number: ", client_num, " Class norm, mi")
            X_train = np.concatenate((norm, mi), axis=0)
            y_train = np.concatenate((label_norm, label_mi), axis=0)
        if num_classes == 3:
            print("Client number: ", client_num, " Class norm, mi, sttc")
            X_train = np.concatenate((norm, mi), axis=0)
            X_train = np.concatenate((X_train, sttc), axis=0)
            y_train = np.concatenate((label_norm, label_mi), axis=0)
            y_train = np.concatenate((y_train, label_sttc), axis=0)
    if client_num == 2:
        if num_classes == 1:
            print("Client number: ", client_num, " Class mi")
            X_train = mi
            y_train = label_mi
        if num_classes == 2:
            print("Client number: ", client_num, " Class mi, sttc")
            X_train = np.concatenate((mi, sttc), axis=0)
            y_train = np.concatenate((label_mi, label_sttc), axis=0)
        if num_classes == 3:
            print("Client number: ", client_num, " Class mi, sttc, hyp")
            X_train = np.concatenate((mi, sttc), axis=0)
            X_train = np.concatenate((X_train, hyp), axis=0)
            y_train = np.concatenate((label_mi, label_sttc), axis=0)
            y_train = np.concatenate((y_train, label_hyp), axis=0)
    if client_num == 3:
        if num_classes == 1:
            print("Client number: ", client_num, " Class sttc")
            X_train = sttc
            y_train = label_sttc
        if num_classes == 2:
            print("Client number: ", client_num, " Class sttc, hyp")
            X_train = np.concatenate((sttc, hyp), axis=0)
            y_train = np.concatenate((label_sttc, label_hyp), axis=0)
        if num_classes == 3:
            print("Client number: ", client_num, " Class sttc, hyp, cd")
            X_train = np.concatenate((sttc, hyp), axis=0)
            X_train = np.concatenate((X_train, cd), axis=0)
            y_train = np.concatenate((label_sttc, label_hyp), axis=0)
            y_train = np.concatenate((y_train, label_cd), axis=0)
    if client_num == 4:
        if num_classes == 1:
            print("Client number: ", client_num, " Class hyp")
            X_train = hyp
            y_train = label_hyp
        if num_classes == 2:
            print("Client number: ", client_num, " Class hyp, cd")
            X_train = np.concatenate((hyp, cd), axis=0)
            y_train = np.concatenate((label_hyp, label_cd), axis=0)
        if num_classes == 3:
            print("Client number: ", client_num, " Class hyp, cd, norm")
            X_train = np.concatenate((hyp, cd), axis=0)
            X_train = np.concatenate((X_train, norm), axis=0)
            y_train = np.concatenate((label_hyp, label_cd), axis=0)
            y_train = np.concatenate((y_train, label_norm), axis=0)
    if client_num == 5:
        if num_classes == 1:
            print("Client number: ", client_num, " Class cd")
            X_train = cd
            y_train = label_cd
        if num_classes == 2:
            print("Client number: ", client_num, " Class cd, norm")
            X_train = np.concatenate((cd, norm), axis=0)
            y_train = np.concatenate((label_cd, label_norm), axis=0)
        if num_classes == 3:
            print("Client number: ", client_num, " Class cd, norm, mi")
            X_train = np.concatenate((cd, norm), axis=0)
            X_train = np.concatenate((X_train, mi), axis=0)
            y_train = np.concatenate((label_cd, label_norm), axis=0)
            y_train = np.concatenate((y_train, label_mi), axis=0)

def label_class(label, clas):
    if clas == 0:
        if label[0] == 1:
            label_sttc.append(label)
            return True
    if clas == 1:
        if label[1] == 1:
            label_hyp.append(label)
            return True
    if clas == 2:
        if label[2] == 1:
            label_mi.append(label)
            return True
    if clas == 3:
        if label[3] == 1:
            label_norm.append(label)
            return True
    if clas == 4:
        if label[4] == 1:
            label_cd.append(label)
            return True


def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """
    global label_sttc, label_hyp, label_mi, label_norm, label_cd
    label_sttc, label_hyp, label_mi, label_norm, label_cd = [],[],[],[],[]
    global X_train, X_val, y_val, y_train, y_test, X_test

    initIID()
    init_nonIID()

    print_json()
    if count_flops:
        # Starts internal FLOPs counter | If there is an Error: See "from pypapi import events"
        high.start_counters([events.PAPI_FP_OPS,])

    global flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total
    flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total = [], [], []


    #X_test = utils.apply_standardizer(X_test, standard_scaler)


    init()

    if plots: #visualize data
        Metrics.load_dataset()
        Metrics.plotten()
        Metrics.ecg_signals()

    global epoch
    epoch = 0
    global encoder_grad_server
    encoder_grad_server = 0

    global data_send_per_epoch_total
    data_send_per_epoch_total = []
    global data_recieved_per_epoch_total
    data_recieved_per_epoch_total = []
    global batches_abort_rate_total
    batches_abort_rate_total = []

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

    global client
    client = Models.Client()
    print("Start Client")
    client.double().to(device)

    global optimizer
    #optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)
    optimizer = AdamW(client.parameters(), lr=lr)
    print("Start Optimizer")

    global error
    #error = nn.CrossEntropyLoss()
    error = nn.BCELoss()
    print("Start loss calcu")

    global data_send_per_epoch
    global data_recieved_per_epoch
    data_send_per_epoch = 0
    data_recieved_per_epoch = 0

    #global scaler
    #scaler = MinMaxScaler()

    if autoencoder:
        global encode
        encode = Models.Encode()
        print("Start Encoder")
        if autoencoder_train == 0:
            encode.load_state_dict(torch.load("./convencoder_medical.pth"))  # CPU
            print("Encoder model loaded")
        encode.eval()
        print("Start eval")
        encode.double().to(device)

        global optimizerencode
        optimizerencode = Adam(encode.parameters(), lr=lr)  ###

    if grad_encode:
        global grad_decoder
        grad_decoder = Models.Grad_Decoder()
        #grad_decoder.load_state_dict(torch.load("./grad_decoder_medical.pth"))
        grad_decoder.double().to(device)
        print("Grad decoder model loaded")

        global optimizer_grad_decoder
        optimizer_grad_decoder = Adam(grad_decoder.parameters(), lr=0.0001)

    global error_grad_autoencoder
    error_grad_autoencoder = nn.MSELoss()

    s = socket.socket()
    print("Start socket connect")
    s.connect((host, port))
    print("Socket connect success, to.", host, port)

    if pretrain_this_client:
        print("Pretrain active")
        for a in range(pretrain_epochs):
            train_epoch(s, pretraining=1)
            val_stage(s, pretraining=1)
        initial_weights = client.state_dict()
        send_msg(s, 2, initial_weights)
        send_msg(s, 3, 0)
        epoch = 0

    #initialize_model(s)
    serverHandler(s)



if __name__ == '__main__':
    main()
