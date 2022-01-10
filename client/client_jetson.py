import struct
import socket
import pickle
import json
from torch.optim import SGD, Adam, AdamW
import sys
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import wfdb
import ast

#import wandb

#b.init(project="test-project", entity="split-learning-medical")

f = open('parameter_client.json', )
data = json.load(f)

# set parameters fron json file
epoch = data["training_epochs"]
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

#wandb.config = {
#  "learning_rate": lr,
#  "epochs": epoch,
#  "batch_size": batchsize
#}

# load data from json file
def init():
    print("Getting the metadata epoch: ", epoch)
    print("Getting the metadata host: ", host)
    print("Getting the metadata port: ", port)
    print("Getting the metadata batchsize: ", batchsize)
    print("Autoencoder: ", autoencoder)
    print("detailed_output: ", detailed_output)
    print("count_flops: ", count_flops)
    print("plots: ", plots)
    print("autoencoder_train: ", autoencoder_train)
    print("deactivate_train_after_num_epochs: ", deactivate_train_after_num_epochs)
    print("learningrate: ", lr)

if count_flops: #Does not work on the Jetson Nano yet. The amount of FLOPs doesn't depend on the architecture. Measuring FLOPs on the PC and JetsonNano would result in the same outcome.
    # The paranoid switch prevents the FLOPs count
    # Solution: sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
    # Needs to be done after every restart of the PC
    from ptflops import get_model_complexity_info
    from pypapi import events, papi_high as high

#Dataset import class:


class PTB_XL(Dataset):
    def __init__(self, transform=None):
        start_dataloading = time.time()
        self.transform = transform

        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        path = 'PTB-XL/ptb-xl/'
        sampling_rate = 100

        # load and convert annotation data
        Y = pd.read_csv('~/split-learning/MESL-main/client/PTB-XL/ptb-xl/' + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, sampling_rate, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        print("dataloader: ", time.time()-start_dataloading)

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass

        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
        test_fold = 0
        self.X_all = X#[np.where(Y.strat_fold != test_fold)]
        self.y_all = Y['diagnostic_superclass']#[(Y.strat_fold != test_fold)].diagnostic_superclass
        self.y_all = np.resize(self.y_all, (21837, ))

    def __len__(self):
        return len(self.y_all)

    def __getitem__(self, idx):
        y_temp = str_to_number(self.y_all[idx])
        #print("X_all size: ", self.X_all[idx].shape)
        X_temp = self.X_all[idx]#.transpose((1, 0))
        ##print("X_temp_array: ", X_temp)
        #X_temp = X_temp[0]
        ##print("before: ", X_temp[0])
        #X_temp = X_temp.reshape(1, 1000)

        sample = X_temp, y_temp

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


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

#Client-Model:
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
            kernel_size=kernel_size, #padding=3 #to match the AE input/output
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
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)
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
            out_channels=hidden_size // 2,
            kernel_size=kernel_size, #padding=1 #to match the AE input/output
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size // 2,
            out_channels=hidden_size // 4,
            kernel_size=kernel_size,
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size // 4,
            out_channels=hidden_size // 4,
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
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size // 2)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size // 4)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size // 4)

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
            input_size=1000,
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
        """
        self.conv2 = ConvNormPool(
            input_size=hid_size//2,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        """

    def forward(self, input):
        x = self.conv1(input)
        #x = self.conv2(x)
        return x


class Encode(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode, self).__init__()
        self.conva = nn.Conv1d(32, 16, 2, stride=1, padding=0)
        self.convb = nn.Conv1d(16, 8, 2, stride=2,  padding=0)
        self.convc = nn.Conv1d(8, 4, 2, stride=2, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        return x

class Grad_Decoder(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Grad_Decoder, self).__init__()
        self.t_conva = nn.ConvTranspose1d(4, 8, 2, stride=2)
        self.t_convb = nn.ConvTranspose1d(8, 16, 2, stride=2)
        self.t_convc = nn.ConvTranspose1d(16, 32, 2, stride=1)

    def forward(self, x):
        x = self.t_conva(x)
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        return x


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


def start_training(s):
    """
    actuall function, which does the training from the train loader and
    testing from the testloader epoch/batch wise
    :param s: socket
    """
    train_losses = []
    train_accs = []
    total_f1 = []
    val_losses = []
    val_accs = []
    data_send_per_epoch_total = []
    data_recieved_per_epoch_total = []
    flops_client_forward_total = []
    flops_client_encoder_total = []
    flops_client_backprob_total = []
    batches_abort_rate_total = []
    time_train_val = 0
    train_losses.append(0)
    train_accs.append(0)
    val_losses.append(0)
    val_accs.append(0)
    total_f1.append(0)
    encoder_grad_server = 0

    #Specify AE configuration
    train_active = 0 #default: AE is pretrained
    train_grad_active = 0
    if autoencoder_train:
        train_active = 1
    if train_gradAE_active:
        train_grad_active = 1
    if count_flops:
        # Starts internal FLOPs counter | If there is an Error: See "from pypapi import events" (line 55)
        high.start_counters([events.PAPI_FP_OPS, ])

    full_dataset = PTB_XL()
    test_size = 0.1
    train_size = 0.8
    val_size = 0.2
    size_test_dataset = int(test_size * len(full_dataset))
    test_dataset, training_dataset = torch.utils.data.random_split(full_dataset, [size_test_dataset,
                                                                                  int(len(
                                                                                      full_dataset)) - size_test_dataset])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    start_time_training = time.time()
    for e in range(epoch):
        #print(f"Starting epoch: {e}/{epoch}")
        if e >= deactivate_train_after_num_epochs: #condition to stop AE training
            train_active = 0 #AE training off
        if e >= deactivate_grad_train_after_num_epochs:
            train_grad_active = 0
        if e == 1:
            print("estimated_time_total: ", time_train_val*epoch/60, " min")

        global data_send_per_epoch
        global data_recieved_per_epoch
        data_send_per_epoch = 0
        data_recieved_per_epoch = 0

        # add comments
        train_loss = 0.0
        correct_train, total_train = 0, 0
        loss_val = 0.0
        correct_val, total_val = 0, 0
        loss_test = 0.0
        correct_test, total_test = 0, 0
        active_training_time_epoch_client = 0
        active_training_time_epoch_server = 0

        concat_counter_send = 0
        concat_counter_recv = 0

        batches_aborted, total_train_nr, total_val_nr, total_test_nr = 0, 0, 0, 0

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
        hamming_epoch = 0
        accuracy_epoch = 0
        precision_epoch = 0
        recall_epoch = 0
        f1_epoch = 0
        if count_flops:
            x = high.read_counters()# reset Flop counter

        # new random split for every epoch
        size_train_dataset = int(train_size * len(training_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(training_dataset,
                                                                   [size_train_dataset,
                                                                    len(training_dataset) - size_train_dataset])
        #print("train_dataset size: ", size_train_dataset)
        #print("val_dataset size: ", len(training_dataset) - size_train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)

        for b, batch in enumerate(train_loader):
            if time_for_bed != 0:
                time_for_bed_total += time.time() - time_for_bed
            if count_flops:
                x = high.read_counters()
            #print("FLOPs dataloader: ", x)
            #if b % 100 == 0:
                #print("batch ", b, " / ", total_batch)

            forward_time = time.time()
            active_training_time_batch_client = 0
            start_time_batch_forward = time.time()

            #define labels and data per batch
            x_train, label_train = batch
            x_train = torch.DoubleTensor(x_train).to(device)
            #x_train = x_train.to(device)
            label_train = torch.DoubleTensor(label_train).to(device)

            #if b == 0:
                #print("input shape: ", x_train.shape)

            if count_flops:
                x = high.read_counters() #reset Flop Counter
            optimizer.zero_grad() #sets gradients to 0 - start for backprop later

            client_output_backprop = client(x_train)
            client_output_train = client_output_backprop.detach().clone()
            #for k in range(batch_concat - 1):
            #    client_output_train = torch.cat((client_output_train, concat_tensors[k + 1]), 0)
            """
            if concat_counter_send < batch_concat:
                concat_tensors.append(client(x_train))
                concat_labels.append(label_train)
                concat_counter_send += 1
                continue
            else:
                client_output_train = concat_tensors[0]
                for k in range(batch_concat - 1):
                    client_output_train = torch.cat((client_output_train, concat_tensors[k + 1]), 0)
                if detailed_output:
                    print("Calculate client_output_train")
            """

            if count_flops:
                x = high.read_counters()
                #print("FLOPs forward: ", x)
                flops_client_forward_total.append(x)

            client_output_train_without_ae_send = 0
            if autoencoder:
                if train_active:
                    optimizerencode.zero_grad()
                #client_output_train_without_ae = client_output_train.clone().detach().requires_grad_(False)
                client_encoded = encode(client_output_train)
                client_output_send = client_encoded.detach().clone()
                if train_active:
                    client_output_train_without_ae_send = client_output_train.detach().clone()
            else:
                client_output_send = client_output_train.detach().clone()
            #client_output_send = encode(client_output_train)

            msg = {
                'client_output_train': client_output_send,
                'client_output_train_without_ae': client_output_train_without_ae_send,
                'label_train': label_train,#concat_labels,
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
            send_time = time.time()
            send_msg(s, 0, msg)
            send_time_total += time.time() - send_time

            concat_labels = []


            recieve_and_backprop = time.time()
            forward_time_total += time.time() - forward_time

            #while concat_counter_recv < concat_counter_send:
            recieve_time = time.time()
            msg = recieve_msg(s)
            recieve_time_total += time.time() - recieve_time

            # decode grad:
            client_grad_without_encode = msg["client_grad_without_encode"]
            client_grad = msg["grad_client"]

            if msg["grad_encode"]:
                if train_grad_active:
                    #print("train_active")
                    optimizer_grad_decoder.zero_grad()
                client_grad = Variable(client_grad, requires_grad=True)
                client_grad_decode = grad_decoder(client_grad)
                if train_grad_active:
                    loss_grad_autoencoder = error_grad_autoencoder(client_grad_without_encode, client_grad_decode)
                    loss_grad_autoencoder.backward()
                    encoder_grad_server = client_grad.grad.detach().clone()  #
                    optimizer_grad_decoder.step()
                    # print("loss_grad_autoencoder: ", loss_grad_autoencoder)
                else:
                    encoder_grad_server = 0
            else:
                if msg["client_grad_abort"] == 1:
                    client_grad_decode = client_grad.detach().clone()
                else:
                    client_grad = "abort"
                encoder_grad_server = 0

            start_time_batch_backward = time.time()

            encoder_grad = msg["encoder_grad"]
            if client_grad == "abort":
                #print("client_grad: ", client_grad)
                train_loss_add, add_correct_train, add_total_train = msg["train_loss"], msg["add_correct_train"], \
                                                                     msg["add_total_train"]
                correct_train += add_correct_train
                total_train_nr += 1
                total_train += add_total_train
                train_loss += train_loss_add
                batches_aborted += 1
                if count_flops:
                    x = high.read_counters()
                    flops_client_encoder_total.append(x)
                    flops_client_backprob_total.append([0])

                output_train = msg["output_train"]
                # print("train_loss: ", train_loss/total_train_nr)
                #meter.update(output_train, label_train, train_loss/total_train_nr)
                pass
            else:
                if train_active:
                    client_encoded.backward(encoder_grad)
                    optimizerencode.step()
                if count_flops:
                    x = high.read_counters()
                    # print("FLOPs encoder: ", x)
                    flops_client_encoder_total.append(x)

                # concat_tensors[concat_counter_recv].to(device)
                # concat_tensors[concat_counter_recv].backward(client_grad)
                #client_output_backprob.to(device)
                #if b % 1000 == 999:
                #    print("Backprop with: ", client_grad)
                client_output_backprop.backward(client_grad_decode)
                optimizer.step()

                if count_flops:
                    x = high.read_counters()
                    # print("FLOPs backprob: ", x)
                    flops_client_backprob_total.append(x)

                train_loss_add, add_correct_train, add_total_train = msg["train_loss"], msg["add_correct_train"], \
                                                                     msg["add_total_train"]

                correct_train += add_correct_train
                total_train_nr += 1
                total_train += add_total_train
                train_loss += train_loss_add

                output_train = msg["output_train"]
                #print("train_loss: ", train_loss/total_train_nr)
                #meter.update(output_train, label_train, train_loss/total_train_nr)



            #wandb.watch(client, log_freq=100)
            output = torch.round(output_train)
            # if np.sum(label.cpu().detach().numpy()[0]) > 1:
            #    if np.sum(output.cpu().detach().numpy()[0] > 1):
            #        print("output[0]: ", output.cpu().detach().numpy()[0])
            #        print("label [0]: ", label.cpu().detach().numpy()[0])
            if (total_train_nr % 100 == 0):
                print("output[0]: ", output.cpu().detach().numpy()[0])
                print("label [0]: ", label_train.cpu().detach().numpy()[0])


            concat_counter_recv += 1

            active_training_time_batch_client += time.time() - start_time_batch_backward
            active_training_time_batch_server = msg["active_trtime_batch_server"]
            active_training_time_epoch_client += active_training_time_batch_client
            active_training_time_epoch_server += active_training_time_batch_server
            #

            recieve_and_backprop_total += (time.time() - recieve_and_backprop)
            time_batches_total += time.time() - forward_time

            concat_counter_send = 0
            concat_counter_recv = 0
            concat_tensors = []

            hamming_epoch += hamming_score(label_train.detach().clone().cpu(),
                                                 output.detach().clone().cpu()).item()
            precision_epoch += Precision(label_train.detach().clone().cpu(),
                                               output.detach().clone().cpu()).item()
            recall_epoch += Recall(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
            f1_epoch += F1Measure(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()


        epoch_endtime = time.time() - epoch_start_time
        status_epoch_train = "epoch: {}, hamming_score: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}  ".format(
            e + 1, hamming_epoch / total_train_nr, precision_epoch / total_train_nr, recall_epoch / total_train_nr,
            f1_epoch / total_train_nr, epoch_endtime, batches_aborted / total_train_nr)
        print("status_epoch_train: ", status_epoch_train)
        hamming_epoch = 0
        precision_epoch = 0
        recall_epoch = 0
        f1_epoch = 0
        val_time = time.time()
        with torch.no_grad():
            for b_t, batch_t in enumerate(val_loader):

                x_val, label_val = batch_t
                x_val, label_val = x_val.to(device), label_val.to(device)
                optimizer.zero_grad()
                output_val = client(x_val)
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
                loss_val += val_loss
                correct_val += correct_val_add
                total_val_add = len(label_val)
                total_val += total_val_add
                total_val_nr += 1

                #meter_val.update(output_val_server, label_val, loss_val / total_val_nr)
                hamming_epoch += hamming_score(label_train.detach().clone().cpu(),
                                                     output.detach().clone().cpu()).item()
                precision_epoch += Precision(label_train.detach().clone().cpu(),
                                                   output.detach().clone().cpu()).item()
                recall_epoch += Recall(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
                f1_epoch += F1Measure(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()

        #print("hamming_epoch: ", hamming_epoch / total_train_nr)
        #print("precision_epoch: ", precision_epoch / total_train_nr)
        #print("recall_epoch: ", recall_epoch / total_train_nr)
        #print("f1_epoch: ", f1_epoch / total_train_nr)

        initial_weights = client.state_dict()
        send_msg(s, 2, initial_weights)

        train_losses.append((train_loss / total_train_nr))
        train_accs.append((correct_train / total_train))
        val_losses.append((loss_val / total_val_nr))
        val_accs.append((correct_val / total_val))

        val_time = time.time() - val_time
        #print("val time: ", val_time)
        time_train_val = val_time+epoch_endtime

        #time_log = time.time()
        #wandb.log({"epoch": e+1})
        #wandb.log({"hamming_score": hamming_epoch / total_val_nr})
        #wandb.log({"precision": precision_epoch / total_val_nr})
        #wandb.log({"recall": recall_epoch / total_val_nr})
        #wandb.log({"f1": f1_epoch / total_val_nr})
        #wandb.watch(client)
        #print("time_log: ", time.time()-time_log)

        status_epoch_val = "epoch: {}, hamming_score: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, trainingtime for epoch: {:.6f}s".format(
            e + 1, hamming_epoch / total_val_nr, precision_epoch / total_val_nr, recall_epoch / total_val_nr,
            f1_epoch / total_val_nr, epoch_endtime)
        print("status_epoch_val: ", status_epoch_val)
        if autoencoder:
            if autoencoder_train:
                print("Autoencoder_train status: ", train_active)
            if grad_encode:
                print("Grad AE train status: ", train_grad_active)
        print("data_send_per_epoch: ", data_send_per_epoch/1000000, " MegaBytes")
        print("data_recieved_per_epoch: ", data_recieved_per_epoch/1000000, "MegaBytes")
        data_send_per_epoch_total.append(data_send_per_epoch)
        data_recieved_per_epoch_total.append(data_recieved_per_epoch)
        batches_abort_rate_total.append(batches_aborted / total_train_nr)


    total_training_time = time.time() - start_time_training
    time_info = "trainingtime for {} epochs: {:.2f}min".format(epoch, total_training_time / 60)
    print("\n", time_info)
    print("Start testing:")


    hamming_epoch = 0
    precision_epoch = 0
    recall_epoch = 0
    f1_epoch = 0


    with torch.no_grad():
        for b_t, batch_t in enumerate(test_loader):

            x_test, label_test = batch_t
            x_test, label_test = x_test.to(device), label_test.to(device)
            optimizer.zero_grad()
            output_test = client(x_test)
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

            #meter_test.update(output_test_server, label_test, loss_test / total_test_nr)
            hamming_epoch += hamming_score(label_train.detach().clone().cpu(),
                                                 output.detach().clone().cpu()).item()
            precision_epoch += Precision(label_train.detach().clone().cpu(),
                                               output.detach().clone().cpu()).item()
            recall_epoch += Recall(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()
            f1_epoch += F1Measure(label_train.detach().clone().cpu(), output.detach().clone().cpu()).item()

    status_test = "test: hamming_epoch: {:.4f}, precision_epoch: {:.4f}, recall_epoch: {:.4f}, f1_epoch: {:.4f}".format(
        hamming_epoch / total_test_nr, precision_epoch / total_test_nr, recall_epoch / total_test_nr,
        f1_epoch / total_test_nr)
    print("status_test: ", status_test)

    data_transfer_per_epoch = 0
    average_dismissal_rate = 0
    for data in data_send_per_epoch_total:
        data_transfer_per_epoch += data
    for data in data_recieved_per_epoch_total:
        data_transfer_per_epoch += data
    for data in batches_abort_rate_total:
        average_dismissal_rate += data
    print("Average data transfer/epoch: ", data_transfer_per_epoch/epoch/1000000, " MB")
    print("Average dismissal rate: ", average_dismissal_rate/epoch)
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
    print("total FLOPs client: ", total_flops_backprob+total_flops_encoder+total_flops_forward)
    #plot(val_accs, train_accs, train_losses, val_losses, total_f1)
    #plt.show()


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


def initialize_model(s):
    """
    if new connected client is not the first connected client,
    the initial weights are fetched from the server
    :param conn:
    """
    msg = recieve_msg(s)
    if msg == 0:
        print("msg == 0")
        pass
    else:
        print("msg != 0")
        client.load_state_dict(msg, strict=False)
        print("model successfully initialized")
    print("start_training")
    start_training(s)


def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """
    init()


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
    client = Client()
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

    if autoencoder:
        global encode
        encode = Encode()
        print("Start Encoder")
        if autoencoder_train == 0:
            encode.load_state_dict(torch.load("./convencoder_medical.pth"))  # CPU
            print("Encoder model loaded")
        encode.eval()
        print("Start eval")
        encode.double().to(device)

        global optimizerencode
        optimizerencode = Adam(encode.parameters(), lr=lr)  ###

    if autoencoder_train:
        global grad_decoder
        grad_decoder = Grad_Decoder()
        grad_decoder.to(device)

        global optimizer_grad_decoder
        optimizer_grad_decoder = Adam(grad_decoder.parameters(), lr=0.0001)

    global error_grad_autoencoder
    error_grad_autoencoder = nn.MSELoss()

    s = socket.socket()
    print("Start socket connect")
    s.connect((host, port))
    print("Socket connect success, to.", host, port)
    initialize_model(s)


if __name__ == '__main__':
    main()
