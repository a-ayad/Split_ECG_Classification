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

# load data from json file
f = open('parameter_server.json', )
data = json.load(f)

# set parameters fron json file
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
lr = data["learningrate"]
update_treshold = data["update_threshold"]
max_numclients = data["max_nr_clients"]
autoencoder = data["autoencoder"]
detailed_output = data["detailed_output"]
autoencoder_train = data["autoencoder_train"]

data_send_per_epoch = 0
client_weights = 0
client_weights_available = 0
num_epochs = 2



class Decode(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode, self).__init__()
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

class Grad_Encoder(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Grad_Encoder, self).__init__()
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
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
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

            #self.pool = nn.MaxPool1d(kernel_size=0)

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
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
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

"""
class initial_Client(nn.Module):
    def __init__(
            self,
            input_size=1,
            hid_size=64,
            kernel_size=5,
            num_classes=5,
    ):
        super().__init__()

        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )

    def forward(self, input):
        x = self.conv1(input)
        return x
"""

class Server(nn.Module):
    def __init__(
            self,
            input_size=128,
            hid_size=128,
            kernel_size=5,
            num_classes=5,
    ):
        super().__init__()

        self.conv2 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size // 2,
            kernel_size=kernel_size,
        )

        self.conv3 = ConvNormPool(
            input_size=hid_size // 2,
            hidden_size=hid_size // 4,
            kernel_size=kernel_size,
        )

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size // 4, out_features=num_classes)

    def forward(self, input):
        #print("input conv2: ", input.shape)
        x = self.conv2(input)
        #print("output conv2: ", x.shape)
        x = self.conv3(x)
        x = self.avgpool(x)
        # print(x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        x = torch.sigmoid(self.fc(x))
        return x



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
    #if detailed_output:
    #print("handle_request: ", content)
    handle_request(sock, getid, content)


def recv_msg(sock):
    """
    gets the message length (which corresponds to the first for bytes of the recieved bytestream) with the recvall function

    :param
        sock: socket
    :return: returns the data retrieved from the recvall function
    """
    raw_msglen = recvall(sock, 4)
    if detailed_output:
        print("RAW MSG LEN: ", raw_msglen)
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
        if detailed_output:
            print("n- lendata: ", n - len(data))
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    #if detailed_output:
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
        3: epoch_finished,
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
        output_test = server(client_output_test)
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
    if detailed_output:
        print("updateclientmodels")
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
        client_output_train, client_output_train_without_ae, label_train, batchsize, batch_concat, train_active, encoder_grad_server, train_grad_active, grad_encode = msg['client_output_train'], msg['client_output_train_without_ae'], msg['label_train'], msg[
            'batchsize'], msg['batch_concat'], msg['train_active'], msg['encoder_grad_server'], msg['train_grad_active'], msg['grad_encode']  # client output tensor
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

    dc = 0
    while dc < batch_concat:
        #tenss = client_output_train_decode#splittensor[dc]
        #tenss = tenss.requires_grad_(True)
        #tenss = tenss.to(device)
        client_output_train_decode = Variable(client_output_train_decode, requires_grad=True)
        output_train = server(client_output_train_decode)  # forward propagation
        with torch.no_grad():
            lbl_train = label_train.to(device)#[dc].to(device)

        loss_train = error(output_train, lbl_train)  # calculates cross-entropy loss
        #train_loss = loss_train.data
        #loss_train = loss_train.to(device)
        loss_train.backward()  # backward propagation
        client_grad_backprop = client_output_train_decode.grad#.clone().detach()
        #print("client_grad_size: ", client_grad_backprop.size())
        client_grad = client_grad_backprop.detach().clone()
        optimizer.step()
        train_loss = loss_train.item()
        add_correct_train = 0#torch.sum(output_train.argmax(dim=1) == lbl_train).item()
        add_total_train = len(lbl_train)
        total_training_time = time.time() - start_time_training
        if detailed_output:
            print("training: ", dc)
        #if train_loss > update_treshold:#.item()
        a = random.randint(6, 10)
        #print("a: ", a)
        if a > 5:
            if detailed_output:
                print("train_loss.item > update_treshold")
            #print("train_loss:", train_loss)
            pass
        else:
            client_grad_send = "abort"#
            print("abort")

        client_grad_without_encode = 0
        client_grad_abort = 0
        if grad_encode:
            if train_grad_active:
                optimizer_grad_encoder.zero_grad()
            grad_encoded = grad_encoder(client_grad)
            client_grad_send = grad_encoded.detach().clone()
            if train_grad_active:
                client_grad_without_encode = client_grad.detach().clone()
        else:
            #if train_loss > update_treshold:
            if a > 5:
                client_grad_send = client_grad.detach().clone()#
                client_grad_abort = 1

        if train_grad_active:
            if grad_available == 1:
                grad_encoded.backward(encoder_grad_server)
                optimizer_grad_encoder.step()
                grad_available = 1

        #print("client_grad_without_encode: ", client_grad_without_encode)

        msg = {"grad_client": client_grad_send,
               "encoder_grad": encoder_grad,
               "client_grad_without_encode": 0,#client_grad_without_encode,
               "grad_encode": grad_encode,
               "train_loss": train_loss,
               "add_correct_train": add_correct_train,
               "add_total_train": add_total_train,
               "active_trtime_batch_server": total_training_time,
               "output_train": output_train,
               "client_grad_abort": client_grad_abort,
               }
        #print("socket", conn)
        #print("msg: ", msg["train_loss"])
        send_msg(conn, msg)
        dc += 1


def epoch_finished(conn, msg):
    global epoch_unfinished
    epoch_unfinished = 1

def initialize_client(conn):
    """
    called when new client connect. if new connected client is not the first connected
    client, the send the initial weights to
    the new connected client
    :param conn:
    """
    if detailed_output:
        print("connected clients: ", len(connectedclients))
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
    initialize_client(conn)
    while True:
        try:
            recieve_msg(conn)
        except:
            #print("No message, wait!")
            pass


def train_client_for_one_epoch(conn):
    send_request(conn, 1, 0)
    global epoch_unfinished
    while True:
        try:
            recieve_msg(conn)
            #print("epoch_unfinished: ", epoch_unfinished)
            if epoch_unfinished:
                print("epoch finished")
                break
        except:
            #print("No message, wait!")
            pass
    epoch_unfinished = 0
    #val Phase
    send_request(conn, 2, 0)
    while True:
        try:
            recieve_msg(conn)
            # print("epoch_unfinished: ", epoch_unfinished)
            if epoch_unfinished:
                print("epoch finished")
                break
        except:
            # print("No message, wait!")
            pass
    epoch_unfinished = 0


def test_client(conn):
    send_request(conn, 3, 0)
    global epoch_unfinished
    while True:
        try:
            recieve_msg(conn)
            # print("epoch_unfinished: ", epoch_unfinished)
            if epoch_unfinished:
                print("epoch finished")
                break
        except:
            # print("No message, wait!")
            pass
    epoch_unfinished = 0


connectedclients = []
trds = []


def main():
    """
    initialize device, server model, initial client model, optimizer, loss, decoder and accepts new clients
    """
    global grad_available
    grad_available = 0

    global epoch_unfinished
    epoch_unfinished = 0

    print(torch.version.cuda)
    global device
    # device = 'cpu'#
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
    server = Server()
    server.double().to(device)

    """
    global client
    client = initial_Client()
    client.to(device)
    print("initial_Client complete.")
    """

    global optimizer
    #optimizer = SGD(server.parameters(), lr=lr, momentum=0.9)
    optimizer = AdamW(server.parameters(), lr=lr)


    global error
    #error = nn.CrossEntropyLoss()
    error = nn.BCELoss()
    print("Calculate CrossEntropyLoss complete.")


    global error_autoencoder
    error_autoencoder = nn.MSELoss()



    if autoencoder:
        global decode
        decode = Decode()
        if autoencoder_train == 0:
            decode.load_state_dict(torch.load("./convdecoder_medical.pth"))
            print("Decoder model loaded")
        decode.eval()
        decode.double().to(device)
        #print("Load decoder parameters complete.")

        global optimizerdecode
        optimizerdecode = Adam(decode.parameters(), lr=0.0001)

    global grad_encoder
    grad_encoder = Grad_Encoder()
    grad_encoder.to(device)

    global optimizer_grad_encoder
    optimizer_grad_encoder = Adam(grad_encoder.parameters(), lr=0.0001)

    s = socket.socket()
    s.bind((host, port))
    s.listen(max_numclients)
    print("Listen to client reply.")

    for i in range(2):
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

    for c, client in enumerate(connectedclients):
        print("test client: ", c + 1)
        test_client(client)

        #t = Thread(target=clientHandler, args=(conn, addr))
        #print('Thread established')
        #trds.append(t)
        #t.start()
        #print('Thread start')

    #for t in trds:
    #    t.join()


if __name__ == '__main__':
    main()
