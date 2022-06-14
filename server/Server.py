import socket
import struct
import pickle
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
import ModelsServer

# load data from json file
f = open('server/parameter_server.json', )
data = json.load(f)

# set parameters fron json file
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
lr = data["learningrate"]
update_treshold = data["update_threshold"]
max_numclients = data["max_nr_clients"]
autoencoder = data["autoencoder"]
#autoencoder_train = data["autoencoder_train"]
num_epochs = data["epochs"]
mech = data["mechanism"]
pretrain_active = data["pretrain_active"]
update_mechanism = data["update_mechanism"]

data_send_per_epoch = 0
client_weights = 0
client_weights_available = 0
autoencoder_train = 0


def send_msg(sock, content):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = pickle.dumps(content)
    msg = struct.pack('>I', len(msg)) + msg  # add 4-byte length in netwwork byte order
    #print("send message with length: ", len(msg))
    sock.sendall(msg)


def send_request(sock, getid, content=None):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = [getid, content]
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg  # add 4-byte length in network byte order
    #print("communication overhead send: ", sys.getsizeof(msg), " bytes")
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
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)


def recvall(sock, n):
    """
    returns the data from a recieved bytestream, helper function to receive n bytes or return None if EOF is hit
    :param sock: socket
    :param n: length in bytes (number of bytes)
    :return: message
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    #if len(data) > 4:
        #print("Length of Data: ", len(data))
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
    }
    switcher.get(getid, "invalid request recieved")(sock, content)


def get_testacc(conn, msg):
    """
    this method does the forward propagation with the recieved data, from to first layer of the decoder to the last layer
    of the model. It sends information about loss/accuracy back to the client.

    :param conn: connection
    :param msg: message
    """
    with torch.no_grad():
        client_output_test, label_test = msg['client_output_val/test'], msg['label_val/test']
        client_output_test, label_test = client_output_test.to(device), label_test.to(device)
        if autoencoder:
            client_output_test = decode(client_output_test)  #
        client_output_test = client_output_test.clone().detach().requires_grad_(True)
        output_test = server(client_output_test, drop=False)
        loss_test = error(output_test, label_test)
        test_loss = loss_test.data
        correct_test = 0#torch.sum(output_test.argmax(dim=1) == label_test).item()

    msg = {"val/test_loss": test_loss,
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
    #client.load_state_dict(updatedweights)
    global client_weights
    global client_weights_available
    client_weights = updatedweights
    client_weights_available = 1
    #for clientss in connectedclients:
    #    try:
    #        if clientss != sock:
    #            send_msg(clientss, updatedweights)
    #            print("weights updated")
    #    except:
    #        pass
    #print("update_time: ", time.time() - update_time)


def dropout_mechanisms(mechanism, epoch):
    """
    returns the predefined dropout function

    :mechanism: string to define dropout function
    :epoch: epoch number to adjust the dropout functions
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    if mechanism == 'linear':
        return (num_epochs - epoch) / num_epochs
    if mechanism == 'none':
        return 1
    if mechanism == 'sigmoid':
        return sigmoid((-epoch + num_epochs/2) / 3)


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
    start_time_training = time.time()
    with torch.no_grad():
        client_output_train, client_output_train_without_ae, label_train, batchsize, train_active, encoder_grad_server, train_grad_active, grad_encode = msg['client_output_train'], msg['client_output_train_without_ae'], msg['label_train'], msg[
            'batchsize'], msg['train_active'], msg['encoder_grad_server'], msg['train_grad_active'], msg['grad_encode']  # client output tensor
        client_output_train, label_train = client_output_train.to(device), label_train
    if autoencoder:
        if train_active:
            #print("train_active")
            optimizerdecode.zero_grad()
        client_output_train = Variable(client_output_train, requires_grad=True)
        client_output_train_decode = decode(client_output_train)
        if train_active:
            loss_autoencoder = error_autoencoder(client_output_train_without_ae, client_output_train_decode)
            loss_autoencoder.backward()
            encoder_grad = client_output_train.grad.detach().clone()#
            optimizerdecode.step()
            #print("loss_autoencoder: ", loss_autoencoder)
        else:
            encoder_grad = 0
    else:
        client_output_train_decode = client_output_train.detach().clone()
        encoder_grad = 0
    #client_output_train = Variable(client_output_train, requires_grad=True)
    #client_output_train_decode = decode(client_output_train)
    #encoder_grad = 0
    optimizer.zero_grad()
    #splittensor = torch.split(client_output_train_decode, batchsize, dim=0)

    # tenss = client_output_train_decode#splittensor[dc]
    # tenss = tenss.requires_grad_(True)
    # tenss = tenss.to(device)
    client_output_train_decode = Variable(client_output_train_decode, requires_grad=True)
    output_train = server(client_output_train_decode)  # forward propagation
    with torch.no_grad():
        lbl_train = label_train.to(device)  # [dc].to(device)

    loss_train = error(output_train, lbl_train)  # calculates cross-entropy loss
    # train_loss = loss_train.data
    # loss_train = loss_train.to(device)
    loss_train.backward()  # backward propagation
    client_grad_backprop = client_output_train_decode.grad  # .clone().detach()
    # print("client_grad_size: ", client_grad_backprop.size())
    client_grad = client_grad_backprop.detach().clone()
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
        if random_number_between_0_and_1 < dropout_mechanisms(mechanism=mech, epoch=epoch):
            update = True
    if update:
        if grad_encode:
            if train_grad_active:
                optimizer_grad_encoder.zero_grad()
            grad_encoded = grad_encoder(grad_preprocessing(client_grad.detach().clone().cpu()))
            client_grad_send = grad_encoded.detach().clone()
            if train_grad_active:
                client_grad_without_encode = grad_preprocessing(client_grad.detach().clone().cpu())
            client_grad_abort = 0
        else:
            client_grad_send = client_grad.detach().clone()
            client_grad_abort = 0
    else:
        client_grad_send = "abort"
        client_grad_abort = 1

    if train_grad_active:
        if grad_available == 1:
            grad_encoded.backward(encoder_grad_server)
            optimizer_grad_encoder.step()
            grad_available = 1

    # print("client_grad_without_encode: ", client_grad_without_encode)

    msg = {"grad_client": client_grad_send,
           "encoder_grad": encoder_grad,
           "client_grad_without_encode": client_grad_without_encode,
           "grad_encode": grad_encode,
           "train_loss": train_loss,
           "add_correct_train": add_correct_train,
           "add_total_train": add_total_train,
           "active_trtime_batch_server": total_training_time,
           "output_train": output_train,
           "client_grad_abort": client_grad_abort,
           "dropout_threshold": dropout_mechanisms(mechanism=mech, epoch=epoch),
           "scaler": scaler
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
            #initial_weights = client.state_dict()
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
    #training cycle for one epoch
    send_request(conn, 1, 0)
    global epoch_finished
    while not epoch_finished:
        recieve_msg(conn)
    print("epoch finished")
    epoch_finished = 0
    #validation cycle
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
    global grad_available, epoch_finished, device
    grad_available, epoch_finished = 0, 0

    print(torch.version.cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if (torch.cuda.is_available()):
        print("training on gpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global server
    server = ModelsServer.Server()
    #server = ModelsServer.Small_TCN_5(5, 12)
    server.double().to(device)

    global optimizer
    #optimizer = SGD(server.parameters(), lr=lr, momentum=0.9)
    optimizer = AdamW(server.parameters(), lr=lr)

    global error
    #error = nn.CrossEntropyLoss()
    error = nn.BCELoss()
    #error = nn.BCEWithLogitsLoss()

    global error_autoencoder
    error_autoencoder = nn.MSELoss()

    global scaler
    scaler = MinMaxScaler()

    if autoencoder:
        global decode
        decode = ModelsServer.Decode()
        if autoencoder_train == 0:
            decode.load_state_dict(torch.load("./convdecoder_medical.pth"))
            print("Decoder model loaded")
        decode.eval()
        decode.double().to(device)
        #print("Load decoder parameters complete.")

        global optimizerdecode
        optimizerdecode = Adam(decode.parameters(), lr=0.0001)

    global grad_encoder
    grad_encoder = ModelsServer.Grad_Encoder()
    #grad_encoder.load_state_dict(torch.load("./grad_encoder_medical.pth"))
    grad_encoder.double().to(device)
    print("Grad encoder model loaded")

    global optimizer_grad_encoder
    optimizer_grad_encoder = Adam(grad_encoder.parameters(), lr=lr)

    global epoch
    epoch = 0

    s = socket.socket()
    s.bind((host, port))
    s.listen(max_numclients)
    print("Listen to client reply.")

    if pretrain_active:
        conn, addr = s.accept()
        connectedclients.append(conn)
        print('Conntected with', addr)
        #initialize_client(connectedclients[0])
        clientHandler(conn, addr)

    for i in range(1):
        conn, addr = s.accept()
        connectedclients.append(conn)
        print('Conntected with', addr)

    print(connectedclients)
    for epoch in range(num_epochs):
        for c, client in enumerate(connectedclients):
            print("init client: ", c+1)
            initialize_client(client)
            print("train_client: ", c+1)
            train_client_for_one_epoch(client)
            #print("test client: ", c + 1)
            #test_client(client, num_epochs)

    for c, client in enumerate(connectedclients):
        print("test client: ", c + 1)
        test_client(client, num_epochs)
    time.sleep(15) #Waiting until Wandb sync is finished
        #t = Thread(target=clientHandler, args=(conn, addr))
        #print('Thread established')
        #trds.append(t)
        #t.start()
        #print('Thread start')

    #for t in trds:
    #    t.join()


if __name__ == '__main__':
    main()
