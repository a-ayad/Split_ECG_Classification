import argparse
import socket
import struct
import pickle

# from MeCab import Model
import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from threading import Thread
import torch.nn.functional as F
import time
from torch.autograd import Variable
import random
import sys
from sklearn.preprocessing import MinMaxScaler
import zlib
import os

import wandb
import ModelsServer
import pandas as pd
from security.analysis import *
import security.utils as utils
from tqdm import tqdm

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "10"
os.environ["PYDEVD_UNBLOCK_THREADS_TIMEOUT"] = "10"
os.environ["PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT"] = "10"
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "10"

# load data from json file
# f = open('server/parameter_server.json', )
f = open(
    "settings.json",
)
data = json.load(f)

# set parameters fron json file
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
lr = data["learningrate"]
update_treshold = data["update_threshold"]
numclients = data["nr_clients"]
autoencoder = data["autoencoder"]
num_epochs = data["epochs"]
mech = data["mechanism"]
pretrain_active = data["pretrain_active"]
update_mechanism = data["update_mechanism"]
model = data["Model"]
autoencoder_train = data["autoencoder_train"]
exp_name = data["exp_name"]
logging_active = True

data_send_per_epoch = 0
client_weights = 0
client_weights_available = 0

detect_anomalies = data["detect_anomalies"]
# global latent_space_image
# global detection_scores
# global detection_tau

if detect_anomalies:
    latent_space_image = utils.reset_latent_space_image()
    detection_scores = pd.DataFrame()
    detection_states = pd.DataFrame(columns=["epoch", "client_id", "state"])
    detection_tau = data["detection_tau"]
    detection_params = data["detection_params"]
    detection_similarity = data["detection_similarity"]
    detection_window = data["detection_window"]
    detection_start = data["detection_start"]
    detection_scheduler = data["detection_scheduler"]
    detection_tolerance = data["detection_tolerance"]
    blocked_list = [False] * numclients
    hold_list = [False] * numclients
    num_malicious = data["num_malicious"]


def update_detection_scores(detection_scores, epoch, stage="train"):
    global latent_space_image

    df_latent = latent_space_image[latent_space_image["stage"] == stage]

    # Per Client similarity scores
    df_clients = client_scores(
        num_clients=numclients,
        num_workers=multiprocessing.cpu_count(),
        df_base=df_latent,
        epochs=[epoch],
        similarities=[detection_similarity],
        **detection_params,
    )
    
    if len(df_clients) == 0:
        return detection_scores

    # Mean of all Labels
    df_clients = df_clients.groupby(["client_id", "epoch"]).mean()

    # Per Client loss contributions
    df_lc = loss_contributions(
        df_latent, num_clients=numclients, epochs=[epoch], moment="mean"
    )

    # Ratio of Similarity to Loss
    df_scores = df_clients.merge(df_lc, on=["epoch", "client_id"])
    df_scores = df_scores.groupby("epoch").apply(
        lambda x: normalize(x, [detection_similarity, "loss"])
    )
    df_scores[detection_similarity] = (1 / df_scores[detection_similarity]).multiply(
        df_scores["loss"], axis=0
    )

    # MAD under gaussian kernel
    df_scores = df_scores.groupby("epoch").apply(
        lambda x: medianAbsoluteDeviation(x, similarities=[detection_similarity])
    )
    df_scores.reset_index(inplace=True)

    # df_scores = pd.DataFrame()
    # for client_id in tqdm(range(1, numclients + 1), desc="Client Detection Scores"):
    #     client_score = per_epoch_scores(epoch, client_id, df=latent_space_image, similarities=[detection_similarity], **detection_params)
    #     client_loss = latent_space_image[latent_space_image["client_id"] == client_id]["loss"].sum()
    #     client_score[detection_similarity] = (1 / client_score[detection_similarity]).multiply(client_loss, axis=0)
    #     df_scores = pd.concat([df_scores, client_score], ignore_index=True)

    # Taking Softmax of Scores
    # df_scores["prob"] = df_scores.apply(lambda x: softmaxScheduler(x, similarities=[detection_similarity]))[detection_similarity]
    # df_scores = df_scores[["client_id", "epoch", "prob"] + [detection_similarity]]
    df_scores = df_scores[["client_id", "epoch"] + [detection_similarity]]

    print("Detection Scores for epoch: ", epoch)
    print(df_scores)

    detection_scores = pd.concat(
        [detection_scores, df_scores], axis=0, ignore_index=True
    )
    
    detection_scores["MA"] = detection_scores.groupby('client_id')[detection_similarity].transform(lambda x: x.rolling(window=detection_window, min_periods=0, center=False).mean())

    return detection_scores


def soft_threshold(client_id):
    if not detect_anomalies:
        return False
    else:
        return get_detection_score(client_id) < 2 * detection_tau


def hard_threshold(client_id):
    if not detect_anomalies:
        return False
    else:
        return get_detection_score(client_id) < detection_tau


def get_detection_score(client_id):
    client_scores = detection_scores[detection_scores["client_id"] == client_id]
    max_epoch = client_scores["epoch"].max()
    
    if len(client_scores) == 0 or max_epoch < detection_start:
        return 1.0
    else:
        score = client_scores.loc[(client_scores['epoch'] == max_epoch), 'MA'].values[0]
        return float(score)


def get_update_probability(client_id):
    if len(detection_scores) == 0:
        return 1.0

    max_epoch = detection_scores["epoch"].max()

    if not max_epoch > detection_start:
        return 1.0

    client_scores = detection_scores[detection_scores["client_id"] == client_id]
    client_scores = client_scores[
        client_scores["epoch"] >= max_epoch - detection_window
    ]

    return client_scores["prob"].mean()


def send_msg(sock, content):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = pickle.dumps(content)
    msg = struct.pack(">I", len(msg)) + msg  # add 4-byte length in netwwork byte order
    # print("send message with length: ", len(msg))
    sock.sendall(msg)


def send_request(sock, getid, content=None):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = [getid, content]
    msg = pickle.dumps(msg)
    msg = struct.pack(">I", len(msg)) + msg  # add 4-byte length in network byte order
    # print("communication overhead send: ", sys.getsizeof(msg), " bytes")
    global data_send_per_epoch
    data_send_per_epoch += sys.getsizeof(msg)
    sock.sendall(msg)


def recieve_msg(sock):
    """
    recieves the meassage with helper function, unpickles the message and separates
    the getid from the actual massage content
    calls the request handler
    :param
        sock: socket
    :return: none
    """
    msg = recv_msg(sock)  # receive client message from socket
    msg = pickle.loads(msg)
    getid = msg[0]
    content = msg[1]
    handle_request(sock, getid, content)


def recv_msg(sock):
    """
    gets the message length (which corresponds to the first for bytes of the recieved bytestream) with the recvall function

    :param sock: socket
    :return: returns the data retrieved from the recvall function
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    return recvall(sock, msglen)


def recvall(sock, n):
    """
    returns the data from a recieved bytestream, helper function to receive n bytes or return None if EOF is hit
    :param sock: socket
    :param n: length in bytes (number of bytes)
    :return: message
    """
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    # if len(data) > 4:
    # print("Length of Data: ", len(data))
    return data


def handle_request(sock, getid, content):
    """
    executes the requested function, depending on the get id, and passes the recieved message

    :param sock: socket
    :param getid: id of the function, that should be executed if the message is recieved
    :param content: message content
    """
    switcher = {
        0: calc_gradients,
        1: get_testacc,
        2: updateclientmodels,
        3: epoch_is_finished,
        4: log_wandb,
    }
    switcher.get(getid, "invalid request recieved")(sock, content)


def log_wandb(sock, msg):
    global logger_id

    if connectedclients.index(sock) == logger_id:
        if msg["action"] == "config":
            wandb.config.update(msg["payload"])
        elif msg["action"] == "log": 
            if detect_anomalies and "AUPRC_val" in msg["payload"]:                
                fp_detection = sum([(int(blocked_list[i]) - int(malicious_clients[i])) * int(blocked_list[i]) for i in range(numclients)])
                fn_detection = sum([(int(malicious_clients[i]) - int(blocked_list[i])) * int(malicious_clients[i]) for i in range(numclients)])
                tp_detection = sum(blocked_list) - fp_detection
                f1_detection = 0.0 if tp_detection == 0 else tp_detection / (tp_detection + 0.5 * (fp_detection + fn_detection))
                
                wandb.log(
                    {
                        "FP_det": fp_detection,
                        "FN_det": fn_detection,
                        "F1_det": f1_detection,
                    },
                    commit=False,
                )
            
            wandb.log(
                msg["payload"],
                commit=False,
            )

def get_testacc(conn, msg):
    """
    this method does the forward propagation with the recieved data, from to first layer of the decoder to the last layer
    of the model. It sends information about loss/accuracy back to the client.

    :param conn: connection
    :param msg: message
    """

    with torch.no_grad():
        client_output_test, label_test = (
            msg["client_output_val/test"],
            msg["label_val/test"],
        )
        client_output_test, label_test = client_output_test.to(device), label_test.to(
            device
        )
        if autoencoder:
            client_output_test = decode(client_output_test)  #
        client_output_test = client_output_test.clone().detach().requires_grad_(True)
        output_test = server(client_output_test, drop=False)
        loss_test = error(output_test, label_test)
        test_loss = loss_test.data
        correct_test = 0  # torch.sum(output_test.argmax(dim=1) == label_test).item()

        # Malicious Client Detection
        if detect_anomalies:
            global latent_space_image
            client_id = connectedclients.index(conn)
            stage = msg["stage"]
            batchsize = msg["batchsize"]
            sample = {
                "client_output": utils.split_batch(client_output_test),
                "label": utils.split_batch(label_test),
                "epoch": [epoch] * batchsize,
                "stage": [stage] * batchsize,
                "client_id": [client_id] * batchsize,
                "loss": [loss_test.item()] * batchsize,
            }
            latent_space_image = pd.concat(
                [latent_space_image, pd.DataFrame(sample)], ignore_index=True
            )

    msg = {
        "val/test_loss": test_loss,
        "correct_val/test": correct_test,
        "output_val/test_server": output_test,
    }
    send_msg(conn, msg)


def updateclientmodels(sock, updatedweights):
    """
    send the actual clientside weights to all connected clients,
    except from the clint that is currently training

    :param sock: the socket
    :param updatedweights: the client side weghts with actual status
    """
    update_time = time.time()
    client_id = connectedclients.index(sock)
    print("Weights Update for Client: ", client_id)
    if detect_anomalies:
        if hold_list[client_id] > 0:
            return
    # client.load_state_dict(updatedweights)
    global client_weights
    global client_weights_available
    client_weights = updatedweights
    client_weights_available = 1
    # for clientss in connectedclients:
    #    try:
    #        if clientss != sock:
    #            send_msg(clientss, updatedweights)
    #            print("weights updated")
    #    except:
    #        pass
    # print("update_time: ", time.time() - update_time)


def dropout_mechanisms(mechanism, epoch):
    """
    returns the predefined dropout function

    :mechanism: string to define dropout function
    :epoch: epoch number to adjust the dropout functions
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if mechanism == "linear":
        return (num_epochs - epoch) / num_epochs
    if mechanism == "none":
        return 1
    if mechanism == "sigmoid":
        return sigmoid((-epoch + num_epochs / 2) / 3)


def grad_preprocessing(grad):
    """
    Only relevant when the gradient is encoded, Apllys a scaling to transform
    the gradients to a range between 0 and 1

    :grad: gradient
    """
    grad_new = grad.numpy()
    for a in range(64):
        grad_new[a] = scaler.fit_transform(grad[a])
    grad_new = torch.DoubleTensor(grad_new).to(device)
    return grad_new


def calc_gradients(conn, msg):
    """
    this method does the forward propagation with the recieved data,
    from to first layer of the decoder to the last layer
    of the model. it calculates the loss, and does the backward propagation up to the
    cutlayer of the model.
    Depending on if the loss threshold is reached it sends the gradient of the back
    propagation at the cut layer and
    information about loss/accuracy/trainingtime back to the client.

    :param conn: the connected socket of the currently training client
    :param msg: the recieved data
    """
    global grad_available
    global latent_space_image
    start_time_training = time.time()
    with torch.no_grad():
        client_output_train, client_output_train_without_ae, label_train, batchsize = (
            msg["client_output_train"],
            msg["client_output_train_without_ae"],
            msg["label_train"],
            msg["batchsize"],
        )
        client_output_train, label_train = client_output_train.to(device), label_train
    if autoencoder:
        client_output_train = Variable(client_output_train, requires_grad=True)
        client_output_train_decode = decode(client_output_train)
    else:
        client_output_train_decode = client_output_train.detach().clone()
    # client_output_train = Variable(client_output_train, requires_grad=True)
    # client_output_train_decode = decode(client_output_train)
    # encoder_grad = 0

    optimizer.zero_grad()
    # splittensor = torch.split(client_output_train_decode, batchsize, dim=0)

    # tenss = client_output_train_decode#splittensor[dc]
    # tenss = tenss.requires_grad_(True)
    # tenss = tenss.to(device)
    client_output_train_decode = Variable(
        client_output_train_decode, requires_grad=True
    )
    output_train = server(client_output_train_decode)  # forward propagation
    with torch.no_grad():
        lbl_train = label_train.to(device)  # [dc].to(device)

    loss_train = error(output_train, lbl_train)  # calculates cross-entropy loss

    # Malicious Client Detection
    client_id = connectedclients.index(conn)
    if detect_anomalies:
        stage = msg["stage"]
        sample = {
            "client_output": utils.split_batch(client_output_train_decode),
            "label": utils.split_batch(label_train),
            "epoch": [epoch] * batchsize,
            "stage": [stage] * batchsize,
            "client_id": [client_id] * batchsize,
            "loss": [loss_train.item()] * batchsize,
        }
        latent_space_image = pd.concat(
            [latent_space_image, pd.DataFrame(sample)], ignore_index=True
        )

    # train_loss = loss_train.data
    # loss_train = loss_train.to(device)
    loss_train.backward()  # backward propagation
    client_grad_backprop = client_output_train_decode.grad  # .clone().detach()
    # print("client_grad_size: ", client_grad_backprop.size())
    client_grad = client_grad_backprop.detach().clone()

    if detect_anomalies:
        update_server = blocked_list[client_id] == 0
    else:
        update_server = True

    if update_server:
        optimizer.step()

    train_loss = loss_train.item()
    add_correct_train = 0  # torch.sum(output_train.argmax(dim=1) == lbl_train).item()
    add_total_train = len(lbl_train)
    total_training_time = time.time() - start_time_training

    random_number_between_0_and_1 = random.uniform(0, 1)

    client_grad_without_encode = 0
    update = False
    if update_mechanism == "static":
        if train_loss > update_treshold:
            update = True
    else:
        if random_number_between_0_and_1 < dropout_mechanisms(
            mechanism=mech, epoch=epoch
        ):
            update = True
    if update:
        client_grad_send = client_grad.detach().clone()
        client_grad_abort = 0
    else:
        client_grad_send = "abort"
        client_grad_abort = 1

    # print("client_grad_without_encode: ", client_grad_without_encode)

    msg = {
        "grad_client": client_grad_send,
        "client_grad_without_encode": client_grad_without_encode,
        "train_loss": train_loss,
        "add_correct_train": add_correct_train,
        "add_total_train": add_total_train,
        "active_trtime_batch_server": total_training_time,
        "output_train": output_train,
        "client_grad_abort": client_grad_abort,
        "dropout_threshold": dropout_mechanisms(mechanism=mech, epoch=epoch),
        "scaler": scaler,
    }
    # print("socket", conn)
    # print("msg: ", msg["train_loss"])
    send_msg(conn, msg)


def epoch_is_finished(conn, msg):
    """
    Sets the bool variable epoch_finished to True, to inform the Server, that the (training for one epoch / validation / testing) is finished
    :param conn: the connected socket of the currently active client
    """
    global epoch_finished
    epoch_finished = 1


def initialize_client(conn):
    """
    called when new client connect. if new connected client is not the first connected
    client, the send the initial weights to
    the new connected client
    :param conn: the connected socket of the currently active client
    """
    if len(connectedclients) == 1:
        msg = 0
    else:
        if client_weights_available:
            initial_weights = client_weights
            # initial_weights = client.state_dict()
            msg = initial_weights
            print("init_weights")
        else:
            msg = 0
    send_request(conn, 0, msg)


def clientHandler(conn, addr):
    """
    called when training on a client starts. The server communicates with the client, until the training epoch is finished
    :param conn: the connected socket of the currently active client
    """
    global epoch_finished
    while not epoch_finished:
        recieve_msg(conn)
    print("epoch finished")
    epoch_finished = 0


def train_client_for_one_epoch(conn):
    """
    Initiates training and validation cycle on a client
    :param conn: the connected socket of the currently active client
    """
    # training cycle for one epoch
    send_request(conn, 1, 0)
    global epoch_finished
    while not epoch_finished:
        recieve_msg(conn)
    print("epoch finished")
    epoch_finished = 0
    # validation cycle
    send_request(conn, 2, 0)
    while not epoch_finished:
        recieve_msg(conn)
    print("val finished")
    epoch_finished = 0


def test_client(conn, num_epochs):
    """
    Initiates testing cycle on a client
    :param conn: the connected socket of the currently active client
    """
    send_request(conn, 3, num_epochs)
    global epoch_finished
    while not epoch_finished:
        recieve_msg(conn)
    print("test cycle finished")
    epoch_finished = 0


connectedclients = []
trds = []


def main():
    """
    initialize device, server model, initial client model, optimizer, loss, decoder and accepts new clients
    """
    global grad_available, epoch_finished, device, epoch
    grad_available, epoch_finished, epoch = 0, 0, 0

    ###
    global prevloss_total, batches_total, average_loss_previous_epoch, lastepoch
    prevloss_total, batches_total, average_loss_previous_epoch, lastepoch = 0, 0, 999, 0
    ###

    print(torch.version.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("training on gpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global server
    if model == "TCN":
        server = ModelsServer.Small_TCN_5(5, 12).double().to(device)
    if model == "CNN":
        server = ModelsServer.Server().double().to(device)

    global optimizer
    # optimizer = SGD(server.parameters(), lr=lr, momentum=0.9)
    optimizer = AdamW(server.parameters(), lr=lr)

    global error
    # error = nn.CrossEntropyLoss()
    error = nn.BCELoss()
    # error = nn.BCEWithLogitsLoss()

    global error_autoencoder
    error_autoencoder = nn.MSELoss()

    global scaler
    scaler = MinMaxScaler()

    global logger_id
    logger_id = numclients - 1

    if autoencoder:
        global decode
        if model == "CNN":
            decode = ModelsServer.Decode()
        if model == "TCN":
            decode = ModelsServer.DecodeTCN()
        if autoencoder_train == 0:
            if model == "CNN":
                decode.load_state_dict(torch.load("server/convdecoder_medical.pth"))
            if model == "TCN":
                decode.load_state_dict(torch.load("server/convdecoder_TCN.pth"))
        decode.eval()
        decode.double().to(device)
        # print("Load decoder parameters complete.")

        global optimizerdecode
        optimizerdecode = Adam(decode.parameters(), lr=0.0001)
        
    global malicious_clients, all_clients

    s = socket.socket()
    s.bind(("0.0.0.0", port))
    s.listen(numclients)
    print("Listen to client reply.")
    
    all_clients = [i for i in range(numclients)]
    
    num_malicious = data["num_malicious"]
    # selects num_malicious numbers out of range(args.num_clients)
    malicious_clients = data["malicious_clients"] if "malicious_clients" in data else None
    if not malicious_clients:
        malicious_clients = list(np.random.choice(all_clients, num_malicious, replace=False))
        malicious_clients = [i in malicious_clients for i in all_clients]

    if pretrain_active:
        conn, addr = s.accept()
        connectedclients.append(conn)
        print("Conntected with", addr)
        # initialize_client(connectedclients[0])
        clientHandler(conn, addr)

        for i in all_clients[:-1]:
            conn, addr = s.accept()
            connectedclients.append(conn)
            print("Conntected with", addr)
    else:
        for i in all_clients:
            conn, addr = s.accept()
            connectedclients.append(conn)
            print("Conntected with", addr)
            
            msg = {"id": i, "malicious": malicious_clients[i]}
                    
            send_request(conn, 5, msg)
            print("Initialized with ID", i)

    # Initialize wandb
    if logging_active:
        wandb.init(
            project="SL_Security",
            entity="mohkoh",
            name=exp_name,
            config={
                "learning_rate": data["learningrate"],
                "batch_size": data["batchsize"],
                "autoencoder": data["autoencoder"],
                "learning_rate": lr,
                "PC: ": 2,
                "detect_anomalies": data["detect_anomalies"],
                "detection_scheduler": data["detection_scheduler"],
                "detection_tau": data["detection_tau"],
                "detection_tolerance": data["detection_tolerance"],
                "detection_window": data["detection_window"],
                "detection_start": data["detection_start"],
                "detection_similarity": data["detection_similarity"],
                "detection_params": data["detection_params"],
                "data_poisoning_prob": data["data_poisoning_prob"],
                "data_poisoning_method": data["data_poisoning_method"],
                "blending_factor": data["blending_factor"],
                "label_flipping_prob": data["label_flipping_prob"],
                "nr_clients": data["nr_clients"],
                "num_malicious": data["num_malicious"],
            },
        )

        wandb.define_metric("AUC_test", summary="max")
        wandb.define_metric("AUPRC_test", summary="max")
        wandb.define_metric("Accuracy_test", summary="max")
        wandb.define_metric("F1_test", summary="max")
        wandb.define_metric("AUC_val", summary="max")
        wandb.define_metric("AUPRC_val", summary="max")
        wandb.define_metric("Accuracy_val", summary="max")
        wandb.define_metric("F1_val", summary="max")
        
        if detect_anomalies:
            wandb.define_metric("FP_det", summary="min")
            wandb.define_metric("FN_det", summary="min")
            wandb.define_metric("F1_det", summary="max")

    print(connectedclients)
    global latent_space_image
    global detection_scores, blocked_list, hold_list
    
    for epoch in range(num_epochs):
        for c in all_clients:
            client = connectedclients[c]
            
            if detect_anomalies:
                if blocked_list[c]:
                    # send_request(
                    #     client,
                    #     6,
                    # )
                    all_clients.remove(c)
                    print("Detected Malicious Client: ", c, " - Closing Connection")
                    continue
            print("Started Training + Val for Client: ", c)
            print("init client: ", c)
            initialize_client(client)
            print("train_client: ", c)
            train_client_for_one_epoch(client)
            # print("test client: ", c + 1)
            # test_client(client, num_epochs)
        if detect_anomalies:
            detection_scores = update_detection_scores(detection_scores, epoch=epoch)
            latent_space_image = utils.reset_latent_space_image()
            hard_thresholds = [
                hard_threshold(client_id) for client_id in range(numclients)
            ]
            hold_list = [
                (hold_list[i] + hard_thresholds[i]) * hard_thresholds[i]
                for i in range(numclients)
            ]
            blocked_list = [
                hold_list[i] > detection_tolerance for i in range(numclients)
            ]
            
            for i in range(numclients):
                if blocked_list[i]:
                    state = -1 
                else:
                    if hold_list[i]:
                        state = 0
                    else:
                        state = 1
                detection_states.loc[len(detection_states)] = [epoch, i, state]
            
        if logging_active:
            # Choose logger
            logger_id = numclients - 1
            
            if detect_anomalies:
                logger_id = len(blocked_list) - blocked_list[::-1].index(False) - 1

                print("Logging Detection Scores and States")

                # Detection Score Plot
                xs = range(num_epochs)
                keys=[f"client_{client_id}" for client_id in range(numclients)]
                
                ys = (
                    detection_scores.sort_values(by="epoch")
                    .groupby("client_id")[detection_similarity]
                    .apply(list)
                    .to_list()
                )
                wandb.log(
                    {
                        "detection_scores": wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title="Detection Scores",
                            xname="Epoch",
                        )
                    },
                    commit=False
                )

                # Detection States Plot
                ys = (
                    detection_states.sort_values(by="epoch")
                    .groupby("client_id")["state"]
                    .apply(list)
                    .to_list()
                )
                wandb.log(
                    {
                        "detection_states": wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title="Detection States",
                            xname="Epoch",
                        )
                    },
                    commit=True
                )
            else:
                wandb.log(
                    {},
                    commit=True,
                )

    print("init client: ", all_clients[-1])
    initialize_client(connectedclients[all_clients[-1]])
    print("test client: ", all_clients[-1])
    test_client(connectedclients[all_clients[-1]], num_epochs)

    for c in all_clients:
        send_request(connectedclients[c], 6, 0)

    time.sleep(15)

    for c in all_clients:
        connectedclients[c].close()

    if logging_active:
        wandb.finish()


if __name__ == "__main__":
    main()
