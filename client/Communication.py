import struct
import socket
import pickle
import sys

global data_send_per_epoch, data_recieved_per_epoch, data_send_per_epoch_total, data_recieved_per_epoch_total
data_send_per_epoch, data_recieved_per_epoch = 0, 0

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
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    # print("Daten: ", data)
    return data


def get_data_send_per_epoch():
    return data_send_per_epoch

def get_data_recieved_per_epoch():
    return data_recieved_per_epoch

def reset_tracker():
    global data_recieved_per_epoch, data_send_per_epoch
    data_send_per_epoch = 0
    data_recieved_per_epoch = 0