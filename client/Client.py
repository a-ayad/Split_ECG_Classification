import multiprocessing
import socket
import pickle
import json
from torch.optim import SGD, Adam, AdamW
import time
import numpy as np  # linear algebra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import norm
import Metrics
import os.path
import utils
import copy
import Models
import Flops
import pandas as pd
import argparse
import Communication
import warnings

warnings.simplefilter("ignore", UserWarning)
from torchmetrics.classification import Accuracy, F1Score, AUROC, AveragePrecision

# np.set_printoptions(threshold=np.inf)


def print_json():
    print("learningrate: ", lr)
    print("Getting the metadata host: ", host)
    print("Getting the metadata port: ", port)
    print("Getting the metadata batchsize: ", batchsize)
    print("Autoencoder: ", autoencoder)
    print("count_flops: ", count_flops)


# load data from json file
class PTB_XL(Dataset):
    def __init__(self, X, y, stage):
        self.stage = stage
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx].transpose((1, 0)), self.y[idx]
        return sample


def init_train_val_dataset():
    X_train, y_train = pickle.loads(
        open(f"train_dataset_{client_num}.pkl", "rb").read()
    )

    if malicious:
        X_train, y_train = poison_data(X_train, y_train)

    train_dataset = PTB_XL(X_train, y_train, stage="train")
    val_dataset = pickle.loads(open(f"val_dataset_{client_num}.pkl", "rb").read())

    os.remove(f"train_dataset_{client_num}.pkl")
    os.remove(f"val_dataset_{client_num}.pkl")

    global train_loader
    global val_loader 
    global chal_loader

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
    )
    
    if add_challenge:
        chal_dataset = pickle.loads(open(f"challenge.pkl", "rb").read())
        chal_loader = torch.utils.data.DataLoader(
            chal_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )


#     raw_dataset = PTB_XL("raw")
#     train_dataset = PTB_XL("train")
#     val_dataset = PTB_XL("val")
#     if IID:
#         subsets = [len(raw_dataset) // num_clients] * num_clients
#         if sum(subsets) < len(raw_dataset):
#             subsets[0] += len(raw_dataset) - sum(subsets)
#         split = torch.utils.data.random_split(
#             raw_dataset, subsets, generator=torch.Generator().manual_seed(42)
#         )
#         train_dataset = split[client_num - 1]
#     if pretrain_this_client:
#         print("len raw dataset", len(raw_dataset))
#         pretrain_dataset, no_dataset = torch.utils.data.random_split(
#             raw_dataset,
#             [round(19267 * IID_percentage), round(19267 * (1 - IID_percentage))],
#             generator=torch.Generator().manual_seed(42),
#         )
#         print("pretrain_dataset length: ", len(pretrain_dataset))
#         global pretrain_loader
#         pretrain_loader = torch.utils.data.DataLoader(
#             pretrain_dataset, batch_size=batchsize, shuffle=True
#         )

#     if mixed_dataset:
#         print("len raw dataset", len(raw_dataset))
#         pretrain_dataset, no_dataset = torch.utils.data.random_split(
#             raw_dataset,
#             [round(19267 * IID_percentage), round(19267 * (1 - IID_percentage))],
#             generator=torch.Generator().manual_seed(42),
#         )
#         print("len train dataset", len(train_dataset))
#         train_dataset = torch.utils.data.ConcatDataset(
#             (pretrain_dataset, train_dataset)
#         )
#         print("len mixed-train dataset", len(train_dataset))
#     print("train_dataset length: ", len(train_dataset))
#     print("val_dataset length: ", len(train_dataset))
#     global train_loader
#     global val_loader

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batchsize, shuffle=True
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batchsize, shuffle=False
#     )


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


def handle_request(sock, getid, content):
    """
    executes the requested function, depending on the get id, and passes the recieved message

    :param sock: socket
    :param getid: id of the function, that should be executed if the message is recieved
    :param content: message content
    """
    # print("request mit id:", getid)
    switcher = {
        0: initialize_model,
        1: train_epoch,
        2: val_stage,
        7: chal_stage,
        3: test_stage,
        5: set_id,
        6: close_connection,
    }
    switcher.get(getid, "invalid request recieved")(sock, content)


def close_connection(s, content):
    global client_connected
    client_connected = False


def set_id(s, msg):
    global client_num, malicious
    client_num = msg["id"]
    malicious = msg["malicious"]
    
    print(f"INITIALIZED CLIENT {client_num}, MALICIUOS: {malicious}")


def serverHandler(conn):
    global client_connected
    while client_connected:
        recieve_request(conn)


def train_epoch(s, pretraining):
    # Initializaion of a bunch of variables
    global client
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    client.to(device)
    global data_send_per_epoch, data_recieved_per_epoch, data_send_per_epoch_total, data_recieved_per_epoch_total
    data_send_per_epoch, data_recieved_per_epoch = 0, 0
    correct_train, total_train, train_loss, loss_grad_total = 0, 0, 0, 0
    batches_aborted, total_train_nr, total_val_nr, total_test_nr = 0, 0, 0, 0
    hamming_epoch, precision_epoch, recall_epoch, f1_epoch, auc_train = 0, 0, 0, 0, 0
    global epoch
    global latent_space_image
    epoch += 1
    (
        flops_forward_epoch,
        flops_encoder_epoch,
        flops_backprop_epoch,
        flops_rest,
        flops_send,
    ) = (0, 0, 0, 0, 0)
    acc, f1, auc, auprc = 0, 0, 0, 0

    epoch_start_time = time.time()

    loader = pretrain_loader if pretraining else train_loader

    Communication.reset_tracker()  # Resets all communication trackers (MBs send/recieved...)

    test_accuracy = Accuracy(num_classes=5, average=average_setting)
    test_f1 = F1Score(num_classes=5, average=average_setting)
    test_auc = AUROC(num_classes=5, average=average_setting)
    test_auprc = AveragePrecision(num_classes=5, average=average_setting)

    for b, batch in enumerate(loader):
        flops_counter.read_counter("")  # Reset FLOPs counter

        forward_time = time.time()
        active_training_time_batch_client = 0
        start_time_batch_forward = time.time()

        # define labels and data per batch
        x_train, label_train = batch

        x_train = x_train.to(device)  # Place data on GPU
        label_train = label_train.double().to(
            device
        )  # Convert Labels to DouleTensors, to fit the model

        if len(x_train) != 64:  # Sorts out batches with less than 64 samples
            break

        flops_counter.read_counter("rest")

        optimizer = AdamW(client.parameters(), lr=lr)

        optimizer.zero_grad()  # sets gradients to 0 - start for backprop later
        client_output_backprop = client(x_train)
        client_output_send = client_output_backprop.detach().clone()

        flops_counter.read_counter("forward")  # Tracks forward propagation FLOPs

        client_output_train_not_encoded = 0
        if autoencoder:
            client_encoded = encode(client_output_send)  # Forward propagation encoder
            client_output_send = client_encoded.detach().clone()

        flops_counter.read_counter("encoder")  # Tracks encoder FLOPs

        global encoder_grad_server
        # Creates a message to the Server, containing model output and training information
        msg = {
            "client_output_train": client_output_send,
            "client_output_train_without_ae": client_output_train_not_encoded,
            "label_train": label_train,  # concat_labels,
            "batchsize": batchsize,
            "epoch": epoch,
            "stage": "train",
        }

        if record_latent_space:
            # Save pooled vector for analysis
            sample = {
                "client_output": utils.split_batch(client_output_send),
                "label": utils.split_batch(label_train),
                "client_output_pooled": utils.split_batch(
                    F.adaptive_avg_pool1d(client_output_send, 1).squeeze()
                ),
                "step": [b] * batchsize,
                "epoch": [epoch] * batchsize,
            }

        active_training_time_batch_client += time.time() - start_time_batch_forward
        Communication.send_msg(s, 0, msg)  # Send message to server
        flops_counter.read_counter("send")  # Tracks FLOPs needed to send the message
        msg = Communication.recieve_msg(s)  # Recieve message from server

        if pretraining == 0:
            Communication.send_msg(
                s,
                4,
                {
                    "action": "log",
                    "payload": {"dropout_threshold": msg["dropout_threshold"]},
                },
            )

        # decode grad:
        client_grad_without_encode = msg["client_grad_without_encode"]
        client_grad = msg["grad_client"]
        flops_counter.read_counter(
            "recieve"
        )  # Tracks FLOPs needed to recieve the message
        global scaler
        scaler = msg["scaler"]
        if msg["client_grad_abort"] == 0:
            client_grad_decode = client_grad.detach().clone()

        start_time_batch_backward = time.time()

        if client_grad == "abort":  # If the client update got aborted
            batches_aborted += 1

            # if record_latent_space:
            #    sample["grad_client"] = utils.split_batch(torch.zeros_like(client_grad))

        else:
            flops_counter.read_counter("rest")
            client_output_backprop.backward(client_grad_decode)  # Backpropagation

            # if record_latent_space:
            #    sample["grad_client"] = utils.split_batch(client_grad_decode)

            optimizer.step()
            flops_counter.read_counter("backprop")

        active_training_time_batch_client += time.time() - start_time_batch_backward

        # Evaluation of the current batch
        total_train_nr += 1
        train_loss += msg["train_loss"]
        output_train = msg["output_train"]

        if record_latent_space:
            sample["loss"] = [msg["train_loss"]] * batchsize
            sample["stage"] = ["train"] * batchsize
            sample["server_output"] = utils.split_batch(output_train)
            df_batch = pd.DataFrame(sample)
            latent_space_image = pd.concat(
                [latent_space_image, df_batch], ignore_index=True
            )

        active_training_time_batch_client += time.time() - start_time_batch_backward

        # Evaluation of the current batch
        acc += test_accuracy(
            output_train.detach().clone().cpu(),
            label_train.detach().clone().cpu().int(),
        ).numpy()
        f1 += test_f1(
            output_train.detach().clone().cpu(),
            label_train.detach().clone().cpu().int(),
        ).numpy()
        auc += test_auc(
            output_train.detach().clone().cpu(),
            label_train.detach().clone().cpu().int(),
        ).numpy()
        auprc += test_auprc(
            output_train.detach().clone().cpu(),
            label_train.detach().clone().cpu().int(),
        ).numpy()

    # Evaluation of Epoch
    epoch_endtime = time.time() - epoch_start_time
    epoch_evaluation(
        total_train_nr,
        s,
        train_loss,
        batches_aborted,
        epoch_endtime,
        test_auc,
        test_auprc,
        test_accuracy,
        test_f1,
        pretraining,
    )

    # Communication with server
    if not pretraining:
        Communication.send_msg(
            s, 2, client.state_dict()
        )  # Share weights with the server
        Communication.send_msg(
            s, 3, 0
        )  # Communicate that the current training epoch is finished


def epoch_evaluation(
    total_train_nr,
    s,
    train_loss,
    batches_aborted,
    epoch_endtime,
    test_auc,
    test_auprc,
    test_accuracy,
    test_f1,
    pretraining,
):
    """
    Evaluation function for the current training epoch
    """

    epoch_auc = test_auc.compute()
    epoch_accuracy = test_accuracy.compute()
    epoch_f1 = test_f1.compute()
    epoch_auprc = test_auprc.compute()
    status_train = "epoch: {}, auc: {:.4f}, auprc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}, train_loss: {:.4f} ".format(
        epoch,
        epoch_auc,
        epoch_auprc,
        epoch_accuracy,
        epoch_f1,
        epoch_endtime,
        batches_aborted / total_train_nr,
        train_loss / total_train_nr,
    )
    if pretraining:
        print("pretrain: ", status_train)

    if not pretraining:
        print("status training: ", status_train)
        global flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total, data_send_per_epoch_total, data_recieved_per_epoch_total
        flops_client_forward_total += flops_counter.flops_forward_epoch
        flops_client_encoder_total += flops_counter.flops_encoder_epoch
        flops_client_backprop_total += flops_counter.flops_backprop_epoch
        flops_client_send_total += flops_counter.flops_send
        flops_client_recieve_total += flops_counter.flops_recieve
        flops_client_rest_total += flops_counter.flops_rest

        print(
            "data_send_per_epoch: ",
            Communication.get_data_send_per_epoch() / 1000000,
            " MegaBytes",
        )
        print(
            "data_recieved_per_epoch: ",
            Communication.get_data_recieved_per_epoch() / 1000000,
            "MegaBytes",
        )
        data_send_per_epoch_total += Communication.get_data_send_per_epoch()
        data_recieved_per_epoch_total += Communication.get_data_recieved_per_epoch()

        if count_flops:
            print(
                "MegaFLOPS_forward_epoch", flops_counter.flops_forward_epoch / 1000000
            )
            print(
                "MegaFLOPS_encoder_epoch", flops_counter.flops_encoder_epoch / 1000000
            )
            print(
                "MegaFLOPS_backprop_epoch", flops_counter.flops_backprop_epoch / 1000000
            )
            print("MegaFLOPS_rest", flops_counter.flops_rest / 1000000)
            print("MegaFLOPS_send", flops_counter.flops_send / 1000000)
            print("MegaFLOPS_recieve", flops_counter.flops_recieve / 1000000)


        if count_flops:
            Communication.send_msg(
                s,
                4,
                {
                    "action": "log",
                    "payload": {
                        "Batches Abortrate": batches_aborted / total_train_nr,
                        "MegaFLOPS Client Encoder": flops_counter.flops_encoder_epoch
                        / 1000000,
                        "MegaFLOPS Client Forward": flops_counter.flops_forward_epoch
                        / 1000000,
                        "MegaFLOPS Client Backprop": flops_counter.flops_backprop_epoch
                        / 1000000,
                        "MegaFLOPS Send": flops_counter.flops_send / 1000000,
                        "MegaFLOPS Recieve": flops_counter.flops_recieve / 1000000,
                    },
                },
            )

        global auc_train_log, auprc_train_log, accuracy_train_log, f1_train_log, batches_abort_rate_total
        auc_train_log = epoch_auc
        auprc_train_log = epoch_auprc
        f1_train_log = epoch_f1
        accuracy_train_log = epoch_accuracy
        batches_abort_rate_total += batches_aborted / total_train_nr

def chal_stage(s, pretraining=0):
    """
    Validation cycle for one epoch, started by the server
    :param s: socket
    :param content:
    """
    global latent_space_image
    total_chal_nr, chal_loss_total = 0, 0
    (
        precision_epoch,
        recall_epoch,
        f1_epoch,
        auc_chal,
        accuracy_sklearn,
        accuracy_custom,
    ) = (0, 0, 0, 0, 0, 0)
    acc, f1, auc, auprc = 0, 0, 0, 0
    chal_accuracy = Accuracy(num_classes=5, average=average_setting)
    chal_f1 = F1Score(num_classes=5, average=average_setting)
    chal_auc = AUROC(num_classes=5, average=average_setting)
    chal_auprc = AveragePrecision(num_classes=5, average=average_setting)

    with torch.no_grad():  # No training involved, thus no gradient needed
        for b_t, batch_t in enumerate(chal_loader):
            x_chal, label_chal = batch_t

            x_chal, label_chal = x_chal.to(device), label_chal.double().to(device)
            # optimizer.zero_grad()
            output_chal = client(x_chal, drop=False)
            if autoencoder:
                output_chal = encode(output_chal)
            chal_batchsize = x_chal.shape[0]
            msg = {
                "client_output_val/test": output_chal,
                "label_val/test": label_chal,
                "epoch": epoch,
                "stage": "chal",
                "batchsize": chal_batchsize,
            }
            Communication.send_msg(s, 1, msg)
            msg = Communication.recieve_msg(s)
            output_chal_server = msg["output_val/test_server"]
            chal_loss_total += msg["val/test_loss"]
            total_chal_nr += 1

            if record_latent_space:
                sample = {
                    "server_output": utils.split_batch(output_chal_server),
                    "label": utils.split_batch(label_chal),
                    "client_output": utils.split_batch(output_chal),
                    "client_output_pooled": utils.split_batch(
                        F.adaptive_avg_pool1d(output_chal, 1).squeeze()
                    ),
                    "loss": [msg["val/test_loss"].detach().cpu().numpy()]
                    * chal_batchsize,
                    "epoch": [epoch] * chal_batchsize,
                    "step": [b_t] * chal_batchsize,
                    "stage": ["chal"] * chal_batchsize
                    # "grad_client": utils.split_batch(torch.zeros_like(output_val)),
                }
                df_batch = pd.DataFrame(sample)
                latent_space_image = pd.concat(
                    [latent_space_image, df_batch], ignore_index=True
                )

            # if b_t < 5:
            #    print("Label: ", label_val[b_t])
            #    print("Pred.: ", torch.round(output_val_server[b_t]))
            #    print("-------------------------------------------------------------------------")

            acc += chal_accuracy(
                output_chal_server.detach().clone().cpu(),
                label_chal.detach().clone().cpu().int(),
            ).numpy()
            f1 += chal_f1(
                output_chal_server.detach().clone().cpu(),
                label_chal.detach().clone().cpu().int(),
            ).numpy()
            auc += chal_auc(
                output_chal_server.detach().clone().cpu(),
                label_chal.detach().clone().cpu().int(),
            ).numpy()
            auprc += chal_auprc(
                output_chal_server.detach().clone().cpu(),
                label_chal.detach().clone().cpu().int(),
            ).numpy()

    epoch_auc, epoch_auprc, epoch_accuracy, epoch_f1 = (
        chal_auc.compute(),
        chal_auprc.compute(),
        chal_accuracy.compute(),
        chal_f1.compute(),
    )
    status_train = "auc: {:.4f}, auprc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}".format(
        epoch_auc, epoch_auprc, epoch_accuracy, epoch_f1
    )
    print("status_chal: ", status_train)

    if pretraining == 0:
        Communication.send_msg(
            s,
            4,
            {
                "action": "log",
                "payload": {
                    "Loss_chal": chal_loss_total / total_chal_nr,
                    "Accuracy_chal": epoch_accuracy,
                    "F1_chal": epoch_f1,
                    "AUC_chal": epoch_auc,
                    "AUPRC_chal": epoch_auprc,
                    "AUC_train": auc_train_log,
                    "AUPRC_train": auprc_train_log,
                    "Accuracy_train": accuracy_train_log,
                    "F1_train": f1_train_log,
                },
            },
        )

    if not pretraining:
        Communication.send_msg(s, 3, 0)

    # Save current latent space image
    if record_latent_space:
        latent_space_image.to_pickle(
            os.path.join(latent_space_dir, f"epoch_{epoch}.pickle")
        )
        latent_space_image = reset_latent_space_image(latent_space_image)

def val_stage(s, pretraining=0):
    """
    Validation cycle for one epoch, started by the server
    :param s: socket
    :param content:
    """
    global latent_space_image
    total_val_nr, val_loss_total = 0, 0
    (
        precision_epoch,
        recall_epoch,
        f1_epoch,
        auc_val,
        accuracy_sklearn,
        accuracy_custom,
    ) = (0, 0, 0, 0, 0, 0)
    acc, f1, auc, auprc = 0, 0, 0, 0
    val_accuracy = Accuracy(num_classes=5, average=average_setting)
    val_f1 = F1Score(num_classes=5, average=average_setting)
    val_auc = AUROC(num_classes=5, average=average_setting)
    val_auprc = AveragePrecision(num_classes=5, average=average_setting)

    with torch.no_grad():  # No training involved, thus no gradient needed
        for b_t, batch_t in enumerate(val_loader):
            x_val, label_val = batch_t

            x_val, label_val = x_val.to(device), label_val.double().to(device)
            # optimizer.zero_grad()
            output_val = client(x_val, drop=False)
            if autoencoder:
                output_val = encode(output_val)
            val_batchsize = x_val.shape[0]
            msg = {
                "client_output_val/test": output_val,
                "label_val/test": label_val,
                "epoch": epoch,
                "stage": "val",
                "batchsize": val_batchsize,
            }
            Communication.send_msg(s, 1, msg)
            msg = Communication.recieve_msg(s)
            output_val_server = msg["output_val/test_server"]
            val_loss_total += msg["val/test_loss"]
            total_val_nr += 1

            if record_latent_space:
                sample = {
                    "server_output": utils.split_batch(output_val_server),
                    "label": utils.split_batch(label_val),
                    "client_output": utils.split_batch(output_val),
                    "client_output_pooled": utils.split_batch(
                        F.adaptive_avg_pool1d(output_val, 1).squeeze()
                    ),
                    "loss": [msg["val/test_loss"].detach().cpu().numpy()]
                    * val_batchsize,
                    "epoch": [epoch] * val_batchsize,
                    "step": [b_t] * val_batchsize,
                    "stage": ["val"] * val_batchsize
                    # "grad_client": utils.split_batch(torch.zeros_like(output_val)),
                }
                df_batch = pd.DataFrame(sample)
                latent_space_image = pd.concat(
                    [latent_space_image, df_batch], ignore_index=True
                )

            # if b_t < 5:
            #    print("Label: ", label_val[b_t])
            #    print("Pred.: ", torch.round(output_val_server[b_t]))
            #    print("-------------------------------------------------------------------------")

            acc += val_accuracy(
                output_val_server.detach().clone().cpu(),
                label_val.detach().clone().cpu().int(),
            ).numpy()
            f1 += val_f1(
                output_val_server.detach().clone().cpu(),
                label_val.detach().clone().cpu().int(),
            ).numpy()
            auc += val_auc(
                output_val_server.detach().clone().cpu(),
                label_val.detach().clone().cpu().int(),
            ).numpy()
            auprc += val_auprc(
                output_val_server.detach().clone().cpu(),
                label_val.detach().clone().cpu().int(),
            ).numpy()

    epoch_auc, epoch_auprc, epoch_accuracy, epoch_f1 = (
        val_auc.compute(),
        val_auprc.compute(),
        val_accuracy.compute(),
        val_f1.compute(),
    )
    status_train = "auc: {:.4f}, auprc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}".format(
        epoch_auc, epoch_auprc, epoch_accuracy, epoch_f1
    )
    print("status_val: ", status_train)

    if pretraining == 0:
        Communication.send_msg(
            s,
            4,
            {
                "action": "log",
                "payload": {
                    "Loss_val": val_loss_total / total_val_nr,
                    "Accuracy_val": epoch_accuracy,
                    "F1_val": epoch_f1,
                    "AUC_val": epoch_auc,
                    "AUPRC_val": epoch_auprc,
                    "AUC_train": auc_train_log,
                    "AUPRC_train": auprc_train_log,
                    "Accuracy_train": accuracy_train_log,
                    "F1_train": f1_train_log,
                },
            },
        )

    if not pretraining:
        client.to("cpu")  # free up some gpu memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        Communication.send_msg(s, 3, 0)

    # Save current latent space image
    if record_latent_space:
        latent_space_image.to_pickle(
            os.path.join(latent_space_dir, f"epoch_{epoch}.pickle")
        )
        latent_space_image = reset_latent_space_image(latent_space_image)


def test_stage(s, epoch):
    """
    Test cycle for one epoch, started by the server
    :param s: socket
    :param epoch: current epoch
    """
    global latent_space_image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    client.to(device)
    encode.to(device)

    total_test_nr, test_loss_total = 0, 0
    (
        precision_epoch,
        recall_epoch,
        f1_epoch,
        auc_test,
        accuracy_sklearn,
        accuracy_custom,
    ) = (0, 0, 0, 0, 0, 0)
    acc, f1, auc, auprc = 0, 0, 0, 0
    test_accuracy = Accuracy(num_classes=5, average=average_setting)
    test_f1 = F1Score(num_classes=5, average=average_setting)
    test_auc = AUROC(num_classes=5, average=average_setting)
    test_auprc = AveragePrecision(num_classes=5, average=average_setting)

    with torch.no_grad():  # No training involved, thus no gradient needed
        for b_t, batch_t in enumerate(val_loader):
            x_test, label_test = batch_t

            x_test, label_test = x_test.to(device), label_test.double().to(device)
            # optimizer.zero_grad()
            output_test = client(x_test, drop=False)
            if autoencoder:
                output_test = encode(output_test)
            test_batchsize = x_test.shape[0]
            msg = {
                "client_output_val/test": output_test,
                "label_val/test": label_test,
                "epoch": epoch,
                "stage": "test",
                "batchsize": test_batchsize,
            }
            Communication.send_msg(s, 1, msg)
            msg = Communication.recieve_msg(s)
            output_test_server = msg["output_val/test_server"]
            test_loss_total += msg["val/test_loss"]
            total_test_nr += 1

            if record_latent_space:
                sample = {
                    "server_output": utils.split_batch(output_test_server),
                    "label": utils.split_batch(label_test),
                    "client_output": utils.split_batch(output_test),
                    "client_output_pooled": utils.split_batch(
                        F.adaptive_avg_pool1d(output_test, 1).squeeze()
                    ),
                    "loss": [msg["val/test_loss"].detach().cpu().numpy()]
                    * test_batchsize,
                    "epoch": [epoch] * test_batchsize,
                    "step": [b_t] * test_batchsize,
                    "stage": ["test"] * test_batchsize
                    # "grad_client": utils.split_batch(torch.zeros_like(output_val)),
                }
                df_batch = pd.DataFrame(sample)
                latent_space_image = pd.concat(
                    [latent_space_image, df_batch], ignore_index=True
                )

            # if b_t < 5:
            #    print("Label: ", label_val[b_t])
            #    print("Pred.: ", torch.round(output_val_server[b_t]))
            #    print("-------------------------------------------------------------------------")

            acc += test_accuracy(
                output_test_server.detach().clone().cpu(),
                label_test.detach().clone().cpu().int(),
            ).numpy()
            f1 += test_f1(
                output_test_server.detach().clone().cpu(),
                label_test.detach().clone().cpu().int(),
            ).numpy()
            auc += test_auc(
                output_test_server.detach().clone().cpu(),
                label_test.detach().clone().cpu().int(),
            ).numpy()
            auprc += test_auprc(
                output_test_server.detach().clone().cpu(),
                label_test.detach().clone().cpu().int(),
            ).numpy()

    epoch_auc, epoch_auprc, epoch_accuracy, epoch_f1 = (
        test_auc.compute(),
        test_auprc.compute(),
        test_accuracy.compute(),
        test_f1.compute(),
    )
    status_train = "auc: {:.4f}, auprc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}".format(
        epoch_auc, epoch_auprc, epoch_accuracy, epoch_f1
    )
    print("status_test: ", status_train)

    Communication.send_msg(
        s,
        4,
        {
            "action": "log",
            "payload": {
                "Loss_test": test_loss_total / total_test_nr,
                "Accuracy_test": epoch_accuracy,
                "F1_test": epoch_f1,
                "AUC_test": epoch_auc,
                "AUPRC_test": epoch_auprc,
            },
        },
    )

    global data_send_per_epoch_total, data_recieved_per_epoch_total, batches_abort_rate_total
    total_flops_model = (
        flops_client_forward_total
        + flops_client_encoder_total
        + flops_client_backprop_total
    )
    total_flops_all = (
        total_flops_model
        + flops_client_send_total
        + flops_client_recieve_total
        + flops_client_rest_total
    )
    data_transfer_per_epoch = data_send_per_epoch_total + data_recieved_per_epoch_total
    if count_flops:
        print("total FLOPs forward: ", flops_client_forward_total)
        print("total FLOPs encoder: ", flops_client_encoder_total)
        print("total FLOPs backprob: ", flops_client_backprop_total)
        print("total FLOPs Model: ", total_flops_model)
        print("total FLOPs: ", total_flops_all)
    print(
        "Average data transfer/epoch: ",
        data_transfer_per_epoch / epoch / 1000000,
        " MB",
    )
    print("Average dismissal rate: ", batches_abort_rate_total / epoch)

    Communication.send_msg(
        s,
        4,
        {
            "action": "config",
            "payload": {
                "Average data transfer/epoch (MB): ": data_transfer_per_epoch
                / epoch
                / 1000000,
                "Average dismissal rate: ": batches_abort_rate_total / epoch,
                "total_MegaFLOPS_forward": flops_client_forward_total / 1000000,
                "total_MegaFLOPS_encoder": flops_client_encoder_total / 1000000,
                "total_MegaFLOPS_backprob": flops_client_backprop_total / 1000000,
                "total_MegaFLOPS model": total_flops_model / 1000000,
                "total_MegaFLOPS": total_flops_all / 1000000,
            },
        },
    )
    
    Communication.send_msg(s, 3, 0)


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


def init_datasets(iid=True, add_challenge=False):
    """
    Data loading
    """
    sampling_frequency = 100
    datafolder = ptb_path
    task = "superdiagnostic"

    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(
        data, labels, task, min_samples=0, outputfolder=mlb_path
    )
    input_shape = data[0].shape
    print(input_shape)

    # 1-9 for training
    X_train = data[labels.strat_fold < 10]
    y_train = Y[labels.strat_fold < 10]
    # 10 for validation
    X_val = data[labels.strat_fold == 10]
    y_val = Y[labels.strat_fold == 10]

    num_classes = 5  # <=== number of classes in the finetuning dataset
    input_shape = [
        1000,
        12,
    ]  # <=== shape of samples, [None, 12] in case of different lengths

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    standard_scaler = pickle.load(open(scaler_path + "/standard_scaler.pkl", "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)

    train_dataset = PTB_XL(X_train, y_train, stage="train")
    val_dataset = PTB_XL(X_val, y_val, stage="val")
    
    if iid:
        # divides the numpy array X_train into num_clients subsets of equal size
        subsets = torch.utils.data.random_split(
            train_dataset,
            [1 / num_clients] * num_clients,
            generator=torch.Generator().manual_seed(42),
        )
        subsets = [(X_train[subset.indices], y_train[subset.indices]) for subset in subsets]
        
    else:
        point_scale = len(X_train) // 10
        client_scale = 2.5
        challenge_frac = 0.05
        
        subsets = []
        y_train_dec = np.sum(y_train * 2**np.arange(y_train.shape[1])[::-1], axis=1)
        df_train = pd.DataFrame({"y": y_train_dec})
        
        if add_challenge:
            challenge = df_train.sample(n=int(len(df_train) * challenge_frac), replace=False)
            df_train = df_train[~df_train.index.isin(challenge.index)]
        
        df_train = df_train.sort_values("y")


        p = np.random.normal(num_clients//2, client_scale, len(df_train))
        p = p.round().clip(0, num_clients-1).astype(int)
        clients, counts = np.unique(p, return_counts=True)

        for client_id in clients:
            loc = counts[:client_id].sum() + counts[client_id] // 2
            df_train[f"pdf_{client_id}"] = norm.pdf(df_train.index.sort_values().values, loc=loc, scale=point_scale)

        for client_id in clients:    
            sample = df_train.sample(n=counts[client_id], weights=f"pdf_{client_id}", replace=False)
            df_train = df_train[~df_train.index.isin(sample.index)]
            sample = pd.concat([sample, challenge], axis=0) if add_challenge else sample
            subsets.append((X_train[sample.index.values], y_train[sample.index.values]))

    for idx, subset in enumerate(subsets):
        pickle.dump(subset, open(f"train_dataset_{idx}.pkl", "wb"))
        pickle.dump(val_dataset, open(f"val_dataset_{idx}.pkl", "wb"))
        
    if add_challenge:
        chal_dataset = PTB_XL(X_train[challenge.index.values], y_train[challenge.index.values], stage="chal")
        pickle.dump(chal_dataset, open("challenge.pkl", "wb"))


def poison_data(X_train, y_train):
    print("Malicious data poisoning activated for client", client_num)
    # model poisoning
    point_mask = np.random.uniform(size=X_train.shape[0]) <= data_poisoning_prob

    if point_mask.sum() > 0:
        if data_poisoning_method == "blend":
            # Blend normal samples with abnormal samples, and vice versa
            print("Poisoning data with blending")
            data_poisoned = X_train[point_mask]
            data_poisoned.shape
            Y_poisoned = y_train[point_mask]

            normal_mask = (Y_poisoned[:, 3] == 1) & (Y_poisoned.sum(axis=1) == 1)
            abnormal_mask = ~normal_mask

            normal_data = data_poisoned[normal_mask]
            abnormal_data = data_poisoned[abnormal_mask]

            abnormal_samples = abnormal_data[
                np.random.choice(
                    abnormal_data.shape[0], normal_data.shape[0], replace=True
                )
            ]
            normal_samples = normal_data[
                np.random.choice(
                    normal_data.shape[0], abnormal_data.shape[0], replace=True
                )
            ]

            normal_data = (
                blending_factor * normal_data + (1 - blending_factor) * abnormal_samples
            )
            abnormal_data = (
                blending_factor * abnormal_data + (1 - blending_factor) * normal_samples
            )

            data_poisoned[normal_mask] = normal_data
            data_poisoned[abnormal_mask] = abnormal_data
            X_train[point_mask] = data_poisoned

        elif data_poisoning_method == "sinusoidal":
            # Blend normal samples with sinusioidal
            data_poisoned = X_train[point_mask]
            data_poisoned.shape
            Y_poisoned = y_train[point_mask]

            normal_mask = (Y_poisoned[:, 3] == 1) & (Y_poisoned.sum(axis=1) == 1)
            normal_data = data_poisoned[normal_mask]
            s = np.sin(np.arange(0, 10, 1 / 100) * blending_factor * 2 * np.pi)
            s = np.tile(s, (12, 1)).T

            normal_data = normal_data * s

            data_poisoned[normal_mask] = normal_data
            X_train[point_mask] = data_poisoned

        elif data_poisoning_method == "flat":
            # Blend abnormal samples with flat line
            data_poisoned = X_train[point_mask]
            data_poisoned.shape
            Y_poisoned = y_train[point_mask]

            normal_mask = (Y_poisoned[:, 3] == 1) & (Y_poisoned.sum(axis=1) == 1)
            abnormal_mask = ~normal_mask
            abnormal_data = data_poisoned[abnormal_mask]

            abnormal_data = abnormal_data * 0

            data_poisoned[abnormal_mask] = abnormal_data
            X_train[point_mask] = data_poisoned

        print("Point Mask Non-Zero for client", client_num)

    label_mask = np.random.uniform(size=y_train.shape[0]) <= label_flipping_prob
    if label_mask.sum() > 0:
        # Change labels that are abnormal to normal, and vice versa
        Y_poisoned = y_train[label_mask]
        normal_mask = (Y_poisoned[:, 3] == 1) & (Y_poisoned.sum(axis=1) == 1)
        normal_labels = Y_poisoned[normal_mask]
        abnormal_mask = ~normal_mask
        abnormal_labels = Y_poisoned[abnormal_mask]
        rand_labels = np.random.randint(0, 2, size=(normal_labels.shape[0], 4))
        Y_poisoned[abnormal_mask] = np.tile(
            np.array([0, 0, 0, 1, 0]), (abnormal_labels.shape[0], 1)
        )
        Y_poisoned[normal_mask] = np.concatenate(
            (
                rand_labels[:, :-1],
                np.zeros((normal_labels.shape[0], 1), dtype=normal_labels.dtype),
                rand_labels[:, -1:],
            ),
            axis=1,
        )
        y_train[label_mask] = Y_poisoned

        print("Label Mask Non-Zero for client", client_num)

    return X_train, y_train

def label_class(label, clas):
    """
    Append the label, matching the sample in the non-IID data case
    param label: label
    param clas: sample class
    """
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


def init_nn_parameters():
    """
    initialize device, client model, optimizer, loss and decoder
    """
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("training on gpu")
    print("training on,", device)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Initialize client, optimizer, error-function, potentially encoder and grad_encoder
    global client
    if model == "TCN":
        client = Models.Small_TCN_5(5, 12)
    if model == "CNN":
        client = Models.Client()
    client.double().to(device)

    global optimizer
    # optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)
    optimizer = AdamW(client.parameters(), lr=lr)

    global error
    # error = nn.CrossEntropyLoss()
    error = nn.BCELoss()

    global data_send_per_epoch
    global data_recieved_per_epoch
    data_send_per_epoch = 0
    data_recieved_per_epoch = 0

    if autoencoder:
        global encode
        if model == "CNN":
            encode = Models.Encode()
            if autoencoder_train == 0:
                encode.load_state_dict(torch.load("client/convencoder_medical.pth"))
        if model == "TCN":
            encode = Models.EncodeTCN()
            if autoencoder_train == 0:
                encode.load_state_dict(torch.load("client/convencoder_TCN.pth"))
        encode.eval()
        print("Start eval")
        encode.double().to(device)

        global optimizerencode
        optimizerencode = Adam(encode.parameters(), lr=lr)  ###


def reset_latent_space_image(df=None):
    df = pd.DataFrame(
        columns=[
            "client_output",
            "client_output_pooled",
            "label",
            "corrupted_point",
            "corrupted_label",
            "step",
            "epoch",
            "stage",
            # "grad_client",
            "loss",
            "server_output",
        ]
    )
    return df


def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """
    global label_sttc, label_hyp, label_mi, label_norm, label_cd
    label_sttc, label_hyp, label_mi, label_norm, label_cd = [], [], [], [], []
    global flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total
    (
        flops_client_forward_total,
        flops_client_encoder_total,
        flops_client_backprop_total,
        flops_client_send_total,
        flops_client_recieve_total,
        flops_client_rest_total,
    ) = (0, 0, 0, 0, 0, 0)
    global data_send_per_epoch_total, data_recieved_per_epoch_total, batches_abort_rate_total, encoder_grad_server, epoch
    (
        data_send_per_epoch_total,
        data_recieved_per_epoch_total,
        batches_abort_rate_total,
        encoder_grad_server,
        epoch,
    ) = (0, 0, 0, 0, 0)
    global flops_counter
    global client_connected
    global mlb_path, scaler_path, ptb_path, output_path
    global lr, batchsize, host, port, max_recv, autoencoder, count_flops, model, num_classes, data_poisoning_prob, blending_factor, label_flipping_prob, record_latent_space, autoencoder_train
    global average_setting, exp_name, latent_space_image, mixed_dataset, IID_percentage, IID, add_challenge, latent_space_dir, client_num, num_clients, pretrain_this_client, malicious, data_poisoning_method

    cwd = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.dirname(cwd)
    mlb_path = os.path.join(cwd, "mlb.pkl")
    scaler_path = os.path.join(cwd)
    ptb_path = os.path.join(
        cwd, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    )
    output_path = os.path.join(
        cwd,
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        "output/",
    )
    # model = 'TCN'

    parser = argparse.ArgumentParser(description="Client")
    parser.add_argument("--init_client", action="store_true")
    args = parser.parse_args()

    init_client = args.init_client
    client_num = -1

    f = open(
        "settings.json",
    )
    data = json.load(f)

    # set parameters fron json file
    # epoch = data["training_epochs"]
    lr = data["learningrate"]
    batchsize = data["batchsize"]
    host = data["host"]
    port = data["port"]
    max_recv = data["max_recv"]
    autoencoder = data["autoencoder"]
    count_flops = data["count_flops"]
    flops_counter = Flops.Flops(count_flops)
    model = data["Model"]
    num_classes = data["num_classes"]
    num_clients = data["nr_clients"]

    # latent space analysis variables & dir for files
    record_latent_space = data["record_latent_space"]
    exp_name = None if data["exp_name"] == "" else data["exp_name"]
    mixed_dataset = data["mixed_with_IID_data"]
    pretrain_epochs = data["pretrain_epochs"]
    IID = data["IID"]
    IID_percentage = data["IID_percentage"]
    add_challenge = data["add_challenge"]
    autoencoder_train = data["autoencoder_train"]
    average_setting = data["average_setting"]

    if not init_client:
        print_json()

        s = socket.socket()
        print("Start socket connect")
        s.connect((host, port))
        print("Socket connect success, to.", host, port)
        client_connected = True

        while client_num == -1:
            recieve_request(s)
            
        # model poisoning parameters
        data_poisoning_prob, blending_factor, label_flipping_prob = 0.0, 0.0, 0.0
        data_poisoning_method = ""
        
        if malicious:
            data_poisoning_prob = data["data_poisoning_prob"]
            blending_factor = data["blending_factor"]
            data_poisoning_method = data["data_poisoning_method"]
            label_flipping_prob = data["label_flipping_prob"]
            
            print(f"---- LFP: {label_flipping_prob}")
            print(f"---- DPP: {data_poisoning_prob}")
            print(f"---- ALPHA: {blending_factor}")
            print(f"---- METHOD: {data_poisoning_method}")

        if client_num == 0:
            pretrain_this_client = data["pretrain_active"]
        else:
            pretrain_this_client = 0

        if pretrain_this_client:
            print("Pretrain active")
            for a in range(pretrain_epochs):
                train_epoch(s, pretraining=1)
                val_stage(s, pretraining=1)
            initial_weights = client.state_dict()
            Communication.send_msg(s, 2, initial_weights)
            Communication.send_msg(s, 3, 0)
            epoch = 0

        if record_latent_space:
            if exp_name is None:
                exp_name = f"N={num_clients}_M={data['num_malicious']}_type=LF_p={data['label_flipping_prob']}"
                
            latent_space_dir = os.path.join(
                cwd, "latent_space", exp_name, "client_{}".format(client_num)
            )
            os.makedirs(latent_space_dir, exist_ok=True)
            latent_space_image = reset_latent_space_image()

            # Check if a file callet metadata.pickle exists in the latent_space_dir
            # If not, create a new file and write the metadata to it
            # Otherwise do nothing
            metadata_path = os.path.join(
                cwd, "latent_space", exp_name, "metadata.pickle"
            )
            if not os.path.isfile(metadata_path):
                metadata = {
                    "num_clients": data["nr_clients"],
                    "exp_name": exp_name,
                    "is_malicious": {client_num: malicious},
                    "batchsize": batchsize,
                    "data_poisoning_prob": data["data_poisoning_prob"],
                    "label_flipping_prob": data["label_flipping_prob"],
                }
                with open(metadata_path, "wb") as handle:
                    pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(metadata_path, "rb") as handle:
                    metadata = pickle.load(handle)
                    metadata["is_malicious"][client_num] = malicious
                    with open(metadata_path, "wb") as handle:
                        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        init_train_val_dataset()
        init_nn_parameters()

        serverHandler(s)
    else:
        init_datasets(iid=IID, add_challenge=add_challenge)


if __name__ == "__main__":
    main()