import struct
import socket
import pickle
import json
from torch.optim import SGD, Adam, AdamW
import sys
import time
import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score
import Metrics
import os.path
import utils
import wandb
import Models
import Communication
# Set path variables to load the PTB-XL dataset and its scaler
cwd = os.path.dirname(os.path.abspath(__file__))
mlb_path = os.path.join(cwd,  "PTB-XL", "ptb-xl", "output", "mlb.pkl")
scaler_path = os.path.join(cwd,  "PTB-XL", "ptb-xl", "output", "standard_scaler.pkl/")
ptb_path = os.path.join(cwd, "PTB-XL", "ptb-xl/")


# Set parameters fron json file
f = open('client\parameter_client.json', )
data = json.load(f)

lr = data["learningrate"]
batchsize = data["batchsize"]
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
autoencoder = data["autoencoder"]
count_flops = data["count_flops"]
autoencoder_train = data["autoencoder_train"]
deactivate_train_after_num_epochs = data["deactivate_train_after_num_epochs"]
grad_encode = data["grad_encode"]
train_gradAE_active = data["train_gradAE_active"]
deactivate_grad_train_after_num_epochs = data["deactivate_grad_train_after_num_epochs"]
weights_and_biases = 0

# Synchronisation with Weights&Biases
if weights_and_biases:
    wandb.init(project="Basis", entity="split-learning-medical")
    wandb.init(config={
        "learning_rate": lr,
        "batch_size": batchsize,
        "autoencoder": autoencoder
    })
    wandb.config.update({"learning_rate": lr, "PC: ": 2})


def print_json():
    """
    Prints all json settings
    """
    print("learningrate: ", lr)
    print("grad_encode: ", grad_encode)
    print("gradAE_train: ", train_gradAE_active)
    print("deactivate_grad_train_after_num_epochs: ", deactivate_grad_train_after_num_epochs)
    print("Getting the metadata host: ", host)
    print("Getting the metadata port: ", port)
    print("Getting the metadata batchsize: ", batchsize)
    print("Autoencoder: ", autoencoder)
    print("count_flops: ", count_flops)
    print("autoencoder_train: ", autoencoder_train)
    print("deactivate_train_after_num_epochs: ", deactivate_train_after_num_epochs)


class PTB_XL(Dataset):
    """
    Class to load sample-sets (train, val, test)
    """
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

    def __len__(self):
        if self.stage == 'train':
            return len(self.y_train)
        if self.stage == 'val':
            return len(self.y_val)
        if self.stage == 'test':
            return len(self.y_test)

    def __getitem__(self, idx):
        if self.stage == 'train':
            sample = self.X_train[idx].transpose((1, 0)), self.y_train[idx]
        if self.stage == 'val':
            sample = self.X_val[idx].transpose((1, 0)), self.y_val[idx]
        if self.stage == 'test':
            sample = self.X_test[idx].transpose((1, 0)), self.y_test[idx]
        return sample


def init():
    """
    Innitialization of the training and validation datasets
    """
    train_dataset = PTB_XL('train')
    val_dataset = PTB_XL('val')
    global train_loader
    global val_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)


if count_flops:
    # Imports to count FLOPs
    # Does not work on every architecture
    # The paranoid switch prevents the FLOPs count
    # Solution: sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
    from ptflops import get_model_complexity_info
    from pypapi import events, papi_high as high


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


def recieve_request(sock):
    """
    recieves the meassage with helper function, umpickles the message and separates the getid from the actual massage content
    :param sock: socket
    """

    msg = Communication.recv_msg(sock)  # receive client message from socket
    msg = pickle.loads(msg)
    getid = msg[0]
    content = msg[1]
    handle_request(sock, getid, content)


def serverHandler(conn):
    while True:
        recieve_request(conn)


def start_training(s, content):
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
    batches_abort_rate_total = []
    time_train_val = 0
    train_losses.append(0)
    train_accs.append(0)
    val_losses.append(0)
    val_accs.append(0)
    total_f1.append(0)



    start_time_training = time.time()
    for e in range(epoch):
        #print(f"Starting epoch: {e}/{epoch}")
        if e >= deactivate_train_after_num_epochs: #condition to stop AE training
            train_active = 0 #AE training off
        if e >= deactivate_grad_train_after_num_epochs:
            train_grad_active = 0
        if e == 1:
            print("estimated_time_total: ", time_train_val*epoch/60, " min")
        if count_flops:
            x = high.read_counters()# reset Flop counter

        train_epoch(s)

        initial_weights = client.state_dict()
        Communication.send_msg(s, 2, initial_weights)

        msg = 0

        Communication.send_msg(s, 3, msg)

        if autoencoder:
            if autoencoder_train:
                print("Autoencoder_train status: ", train_active)
            if grad_encode:
                print("Grad AE train status: ", train_grad_active)

        #batches_abort_rate_total.append(batches_aborted / total_train_nr)


    total_training_time = time.time() - start_time_training
    time_info = "trainingtime for {} epochs: {:.2f}min".format(epoch, total_training_time / 60)
    print("\n", time_info)
    print("Start testing:")

    test_stage(s)


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
    #for flop in flops_client_forward_total:
    #    total_flops_forward += flop[0]
    #for flop in flops_client_encoder_total:
    #    total_flops_encoder += flop[0]
    #for flop in flops_client_backprob_total:
    #    total_flops_backprob += flop[0]
    print("total FLOPs forward: ", total_flops_forward)
    print("total FLOPs encoder: ", total_flops_encoder)
    print("total FLOPs backprob: ", total_flops_backprob)
    print("total FLOPs client: ", total_flops_backprob+total_flops_encoder+total_flops_forward)
    #plot(val_accs, train_accs, train_losses, val_losses, total_f1)
    #plt.show()


def grad_postprocessing(grad):
    grad_new = grad.numpy()
    for a in range(64):
        #scaler.fit(grad[a])
        grad_new[a] = scaler.inverse_transform(grad[a])
    grad_new = torch.DoubleTensor(grad_new).to(device)
    return grad_new


def train_epoch(s, content):
    #new_split() #new random dist between train and val
    loss_grad_total = 0
    global epoch
    epoch += 1
    flops_forward_epoch, flops_encoder_epoch, flops_backprop_epoch, flops_rest, flops_send, flops_recieve = 0,0,0,0,0,0
    #Specify AE configuration
    train_active = 0 #default: AE is pretrained
    train_grad_active = 0
    if epoch < deactivate_train_after_num_epochs:
        if autoencoder_train:
            train_active = 1
    if epoch < deactivate_grad_train_after_num_epochs:
        if train_gradAE_active:
            train_grad_active = 1

    Communication.reset_tracker()
    correct_train, total_train, train_loss = 0, 0, 0
    batches_aborted, total_train_nr, total_val_nr, total_test_nr = 0, 0, 0, 0
    hamming_epoch, precision_epoch, recall_epoch, f1_epoch, auc_train = 0, 0, 0, 0, 0
    #encoder_grad_server = 0

    epoch_start_time = time.time()

    for b, batch in enumerate(train_loader):
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
            'batchsize': batchsize,
            'train_active': train_active,
            'encoder_grad_server': encoder_grad_server,
            'train_grad_active': train_grad_active,
            'grad_encode': grad_encode
        }
        active_training_time_batch_client += time.time() - start_time_batch_forward
        Communication.send_msg(s, 0, msg)

        if count_flops:
            x = high.read_counters()  # reset counter
            flops_send += x[0]

        # while concat_counter_recv < concat_counter_send:
        msg = Communication.recieve_msg(s)
        client_grad_without_encode = msg["client_grad_without_encode"]
        client_grad = msg["grad_client"]
        # print("msg: ", msg)
        if weights_and_biases:
            wandb.log({"dropout_threshold": msg["dropout_threshold"]},
                      commit=False)

        if count_flops:
            x = high.read_counters()  # reset counter
            flops_recieve += x[0]

        # decode grad:
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
            if count_flops:
                x = high.read_counters() # reset counter
                flops_rest += x[0]

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
            roc_auc = roc_auc_score(label_train.detach().clone().cpu(), torch.round(output).detach().clone().cpu(), average='micro')
            auc_train += roc_auc
        except:
            # print("auc_train_exception: ")
            # print("label: ", label)
            # print("output: ", output)
            pass

        hamming_epoch += Metrics.Accuracy(label_train.detach().clone().cpu(), output.detach().clone().cpu())#accuracy_score(label_train.detach().clone().cpu(), output.detach().clone().cpu())
        precision_epoch += precision_score(label_train.detach().clone().cpu(),
                                           output.detach().clone().cpu(), average='micro', zero_division=0)
        recall_epoch += recall_score(label_train.detach().clone().cpu(), output.detach().clone().cpu(), average='micro')
        f1_epoch += f1_score(label_train.detach().clone().cpu(), output.detach().clone().cpu(), average='micro')

    flops_client_forward_total.append(flops_forward_epoch)
    flops_client_encoder_total.append(flops_encoder_epoch)
    flops_client_backprop_total.append(flops_backprop_epoch)
    flops_client_send_total.append(flops_send)
    flops_client_recieve_total.append(flops_recieve)
    flops_client_rest_total.append(flops_rest)


    print("data_send_per_epoch: ", Communication.get_data_send_per_epoch() / 1000000, " MegaBytes")
    print("data_recieved_per_epoch: ", Communication.get_data_recieved_per_epoch() / 1000000, "MegaBytes")
    data_send_per_epoch_total.append(Communication.get_data_send_per_epoch())
    data_recieved_per_epoch_total.append(Communication.get_data_recieved_per_epoch())

    epoch_endtime = time.time() - epoch_start_time
    status_epoch_train = "epoch: {}, AUC_train: {:.4f}, Accuracy_micro: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}, train_loss: {:.4f}  ".format(
        epoch, auc_train / total_train_nr, hamming_epoch / total_train_nr, precision_epoch / total_train_nr,
        recall_epoch / total_train_nr,
        f1_epoch / total_train_nr, epoch_endtime, batches_aborted / total_train_nr, train_loss / total_train_nr)
    print("status_epoch_train: ", status_epoch_train)
    print("MegaFLOPS_forward_epoch", flops_forward_epoch/1000000)
    print("MegaFLOPS_encoder_epoch", flops_encoder_epoch/1000000)
    print("MegaFLOPS_backprop_epoch", flops_backprop_epoch/1000000)
    print("MegaFLOPS_rest", flops_rest/1000000)
    print("MegaFLOPS_send", flops_send/1000000)
    print("MegaFLOPS_recieve", flops_recieve/1000000)

    if weights_and_biases:
        wandb.log({"Batches Abortrate": batches_aborted / total_train_nr, "MegaFLOPS Client Encoder": flops_encoder_epoch/1000000,
                   "MegaFLOPS Client Forward": flops_forward_epoch / 1000000,
                   "MegaFLOPS Client Backprop": flops_backprop_epoch / 1000000, "MegaFLOPS Send": flops_send / 1000000,
                   "MegaFLOPS Recieve": flops_recieve / 1000000},
                  commit=False)

    global auc_train_log
    auc_train_log = auc_train / total_train_nr
    global accuracy_train_log
    accuracy_train_log = hamming_epoch / total_train_nr
    global batches_abort_rate_total
    batches_abort_rate_total.append(batches_aborted / total_train_nr)

    initial_weights = client.state_dict()
    Communication.send_msg(s, 2, initial_weights)

    msg = 0

    Communication.send_msg(s, 3, msg)


def val_stage(s, content):
    total_val_nr, val_loss_total, correct_val, total_val = 0, 0, 0, 0
    val_losses, val_accs  = [], []
    hamming_epoch, precision_epoch, recall_epoch, f1_epoch, accuracy, auc_val, accuracy_sklearn,  accuracy_custom = 0, 0, 0, 0, 0, 0, 0, 0
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
            Communication.send_msg(s, 1, msg)
            msg = Communication.recieve_msg(s)
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
            accuracy_sklearn += accuracy_score(label_val.detach().clone().cpu(),
                                            output_val_server.detach().clone().cpu())
            accuracy_custom += Metrics.Accuracy(label_val.detach().clone().cpu(),
                                                output_val_server.detach().clone().cpu())
            precision_epoch += precision_score(label_val.detach().clone().cpu(),
                                               output_val_server.detach().clone().cpu(), average='micro',
                                               zero_division=0)
            recall_epoch += recall_score(label_val.detach().clone().cpu(), output_val_server.detach().clone().cpu(),
                                         average='micro', zero_division=0)
            f1_epoch += f1_score(label_val.detach().clone().cpu(), output_val_server.detach().clone().cpu(),
                                 average='micro', zero_division=0)

    if weights_and_biases:
        wandb.log({"Loss_val": val_loss_total / total_val_nr,
               "Accuracy_val_micro": accuracy_custom / total_val_nr,
               "F1_val": f1_epoch / total_val_nr,
               "AUC_val": auc_val / total_val_nr,
               "AUC_train": auc_train_log,
               "Accuracy_train": accuracy_train_log})

    status_epoch_val = "epoch: {},AUC_val: {:.4f} ,Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, val_loss: {:.4f}".format(
        epoch, auc_val / total_val_nr, accuracy_custom / total_val_nr, precision_epoch / total_val_nr,
        recall_epoch / total_val_nr,
        f1_epoch / total_val_nr, val_loss_total / total_val_nr)
    print("status_epoch_val: ", status_epoch_val)

    msg = 0
    Communication.send_msg(s, 3, msg)


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
            Communication.send_msg(s, 1, msg)
            msg = Communication.recieve_msg(s)
            correct_test_add = msg["correct_val/test"]
            test_loss = msg["val/test_loss"]
            output_test_server = msg["output_val/test_server"]
            loss_test += test_loss
            correct_test += correct_test_add
            total_test_add = len(label_test)
            total_test += total_test_add
            total_test_nr += 1

            output_test_server = torch.round(output_test_server)
            hamming_epoch += accuracy_score(label_test.detach().clone().cpu(),
                                            output_test_server.detach().clone().cpu())
            precision_epoch += precision_score(label_test.detach().clone().cpu(),
                                               output_test_server.detach().clone().cpu(), average='micro', zero_division=0)
            recall_epoch += recall_score(label_test.detach().clone().cpu(), output_test_server.detach().clone().cpu(),
                                         average='micro')
            f1_epoch += f1_score(label_test.detach().clone().cpu(), output_test_server.detach().clone().cpu(), average='micro')

    status_test = "test: hamming_epoch: {:.4f}, precision_epoch: {:.4f}, recall_epoch: {:.4f}, f1_epoch: {:.4f}".format(
        hamming_epoch / total_test_nr, precision_epoch / total_test_nr, recall_epoch / total_test_nr,
        f1_epoch / total_test_nr)
    print("status_test: ", status_test)


    global data_send_per_epoch_total
    global data_recieved_per_epoch_total
    global batches_abort_rate_total


    data_transfer_per_epoch, average_dismissal_rate, total_flops_forward, total_flops_encoder, total_flops_backprob, total_flops_send, total_flops_recieve,total_flops_rest = 0,0,0,0,0,0,0,0
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
    for flop in flops_client_send_total:
        total_flops_send += flop
    for flop in flops_client_recieve_total:
        total_flops_recieve += flop
    for flop in flops_client_rest_total:
        total_flops_rest += flop
    total_flops_model = total_flops_backprob + total_flops_encoder + total_flops_forward
    total_flops_all = total_flops_model+total_flops_send+total_flops_recieve+total_flops_rest
    print("total FLOPs forward: ", total_flops_forward)
    print("total FLOPs encoder: ", total_flops_encoder)
    print("total FLOPs backprob: ", total_flops_backprob)
    print("total FLOPs Model: ", total_flops_model)
    print("total FLOPs: ", total_flops_all)
    print("Average data transfer/epoch: ", data_transfer_per_epoch / epoch / 1000000, " MB")
    print("Average dismissal rate: ", average_dismissal_rate / epoch)

    if weights_and_biases:
        wandb.config.update({"Average data transfer/epoch (MB): ": data_transfer_per_epoch / epoch / 1000000,
                         "Average dismissal rate: ": average_dismissal_rate / epoch,
                         "total_MegaFLOPS_forward": total_flops_forward/1000000, "total_MegaFLOPS_encoder": total_flops_encoder/1000000,
                         "total_MegaFLOPS_backprob": total_flops_backprob/1000000,"total_MegaFLOPS modal": total_flops_model/1000000 ,"total_MegaFLOPS": total_flops_all/1000000})

    msg = 0
    Communication.send_msg(s, 3, msg)


def initialize_model(s, msg):
    """
    if new connected client is not the first connected client,
    the initial weights are fetched from the server
    :param conn:
    """
    if msg == 0:
        pass
    else:
        print("msg != 0")
        client.load_state_dict(msg, strict=False)
        print("model successfully initialized")


def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """
    print_json()
    if count_flops:
        # Starts internal FLOPs counter | If there is an Error: See "from pypapi import events"
        high.start_counters([events.PAPI_FP_OPS,])

    global flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total
    flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total = [], [], [], [], [], []

    global X_train, X_val, y_val, y_train, y_test, X_test
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
    #X_test = utils.apply_standardizer(X_test, standard_scaler)


    init()

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
    #initialize_model(s)
    serverHandler(s)



if __name__ == '__main__':
    main()
