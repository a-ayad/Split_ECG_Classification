import socket
import pickle
import json
from torch.optim import SGD, Adam, AdamW
import time
import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import Metrics
import os.path
import utils
import Models
import Flops
import wandb
import Communication
import warnings
warnings.simplefilter("ignore", UserWarning)
from torchmetrics.classification import Accuracy, F1Score, AUROC
#np.set_printoptions(threshold=np.inf)
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(cwd)
mlb_path = os.path.join(cwd, "mlb.pkl")
scaler_path = os.path.join(cwd)
ptb_path = os.path.join(cwd, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/")
output_path = os.path.join(cwd, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "output/")
#model = 'TCN'

client_num = 5
IID = 0 #to use IID data like in the single client experiment
average_setting = 'micro'
weights_and_biases = 1

f = open('settings.json', )
data = json.load(f)

# set parameters fron json file
#epoch = data["training_epochs"]
lr = data["learningrate"]
batchsize = data["batchsize"]
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
autoencoder = data["autoencoder"]
count_flops = data["count_flops"]
model = data["Model"]
num_classes = data["num_classes"]
if client_num == 1:
    pretrain_this_client = data["pretrain_active"]
else:
    pretrain_this_client = 0
mixed_dataset = data["mixed_with_IID_data"]
pretrain_epochs = data["pretrain_epochs"]

if weights_and_biases:
    wandb.init(project="TCN new Metric", entity="mfrei")
    wandb.init(config={
        "learning_rate": lr,
        "batch_size": batchsize,
        "autoencoder": autoencoder
    })
    wandb.config.update({"learning_rate": lr, "PC: ": 2})


def print_json():
    print("learningrate: ", lr)
    print("Getting the metadata host: ", host)
    print("Getting the metadata port: ", port)
    print("Getting the metadata batchsize: ", batchsize)
    print("Autoencoder: ", autoencoder)
    print("count_flops: ", count_flops)

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




def init_train_val_dataset():
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

    if mixed_dataset:
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
    #Initializaion of a bunch of variables
    global client
    torch.cuda.empty_cache()
    client.to('cuda:0')
    global data_send_per_epoch, data_recieved_per_epoch, data_send_per_epoch_total, data_recieved_per_epoch_total
    data_send_per_epoch, data_recieved_per_epoch = 0, 0
    correct_train, total_train, train_loss, loss_grad_total = 0, 0, 0, 0
    batches_aborted, total_train_nr, total_val_nr, total_test_nr = 0, 0, 0, 0
    hamming_epoch, precision_epoch, recall_epoch, f1_epoch, auc_train = 0, 0, 0, 0, 0
    global epoch
    epoch += 1
    flops_forward_epoch, flops_encoder_epoch, flops_backprop_epoch, flops_rest, flops_send = 0,0,0,0,0
    acc, f1, auc = 0,0,0


    epoch_start_time = time.time()

    loader = pretrain_loader if pretraining else train_loader

    Communication.reset_tracker()#Resets all communication trackers (MBs send/recieved...)

    test_accuracy = Accuracy(num_classes=5, average=average_setting)
    test_f1 = F1Score(num_classes=5, average=average_setting)
    test_auc = AUROC(num_classes=5, average=average_setting)

    for b, batch in enumerate(loader):
        flops_counter.read_counter("") #Reset FLOPs counter

        forward_time = time.time()
        active_training_time_batch_client = 0
        start_time_batch_forward = time.time()

        # define labels and data per batch
        x_train, label_train = batch
        x_train = x_train.to(device) #Place data on GPU
        label_train = label_train.double().to(device) #Convert Labels to DouleTensors, to fit the model

        if len(x_train) != 64: #Sorts out batches with less than 64 samples
            break

        flops_counter.read_counter("rest")

        optimizer.zero_grad()  # sets gradients to 0 - start for backprop later
        client_output_backprop = client(x_train)
        client_output_send = client_output_backprop.detach().clone()

        flops_counter.read_counter("forward") #Tracks forward propagation FLOPs

        client_output_train_not_encoded = 0
        if autoencoder:
            client_encoded = encode(client_output_send) #Forward propagation encoder
            client_output_send = client_encoded.detach().clone()

        flops_counter.read_counter("encoder") #Tracks encoder FLOPs


        global encoder_grad_server
        #Creates a message to the Server, containing model output and training information
        msg = {
            'client_output_train': client_output_send,
            'client_output_train_without_ae': client_output_train_not_encoded,
            'label_train': label_train,  # concat_labels,
            'batchsize': batchsize,
        }
        active_training_time_batch_client += time.time() - start_time_batch_forward
        Communication.send_msg(s, 0, msg) #Send message to server
        flops_counter.read_counter("send") #Tracks FLOPs needed to send the message
        msg = Communication.recieve_msg(s) #Recieve message from server

        if pretraining == 0:
            if weights_and_biases:
                wandb.log({"dropout_threshold": msg["dropout_threshold"]}, commit=False)

        # decode grad:
        client_grad_without_encode = msg["client_grad_without_encode"]
        client_grad = msg["grad_client"]
        flops_counter.read_counter("recieve") #Tracks FLOPs needed to recieve the message
        global scaler
        scaler = msg["scaler"]
        if msg["client_grad_abort"] == 0:
            client_grad_decode = client_grad.detach().clone()

        start_time_batch_backward = time.time()

        if client_grad == "abort": #If the client update got aborted
            batches_aborted += 1
        else:
            flops_counter.read_counter("rest")
            client_output_backprop.backward(client_grad_decode) #Backpropagation
            optimizer.step()
            flops_counter.read_counter("backprop")  

        active_training_time_batch_client += time.time() - start_time_batch_backward

        #Evaluation of the current batch    
        total_train_nr += 1
        train_loss += msg["train_loss"]
        output_train = msg["output_train"]

        active_training_time_batch_client += time.time() - start_time_batch_backward

        #Evaluation of the current batch
        acc +=test_accuracy(output_train.detach().clone().cpu(), label_train.detach().clone().cpu().int()).numpy()
        f1 += test_f1(output_train.detach().clone().cpu(), label_train.detach().clone().cpu().int()).numpy()
        auc += test_auc(output_train.detach().clone().cpu(), label_train.detach().clone().cpu().int()).numpy()

    #Evaluation of Epoch
    epoch_endtime = time.time() - epoch_start_time
    epoch_evaluation(total_train_nr, train_loss, batches_aborted, epoch_endtime, test_auc, test_accuracy, test_f1, pretraining)

    #Communication with server
    if not pretraining:
        Communication.send_msg(s, 2, client.state_dict()) #Share weights with the server
        Communication.send_msg(s, 3, 0) #Communicate that the current training epoch is finished


def epoch_evaluation(total_train_nr, train_loss, batches_aborted, epoch_endtime, test_auc, test_accuracy, test_f1, pretraining):
    """
        Evaluation function for the current training epoch
    """

    epoch_auc = test_auc.compute()
    epoch_accuracy = test_accuracy.compute()
    epoch_f1 = test_f1.compute()
    status_train = "epoch: {}, auc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}, train_loss: {:.4f} ".format(
        epoch, epoch_auc, epoch_accuracy, epoch_f1, epoch_endtime, batches_aborted / total_train_nr, train_loss / total_train_nr)
    if pretraining: print("pretrain: ", status_train)

    if not pretraining:
        print("status training: ", status_train)
        global flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total, data_send_per_epoch_total, data_recieved_per_epoch_total
        flops_client_forward_total += flops_counter.flops_forward_epoch
        flops_client_encoder_total += flops_counter.flops_encoder_epoch
        flops_client_backprop_total+=flops_counter.flops_backprop_epoch
        flops_client_send_total+=flops_counter.flops_send
        flops_client_recieve_total+=flops_counter.flops_recieve
        flops_client_rest_total+=flops_counter.flops_rest

        print("data_send_per_epoch: ", Communication.get_data_send_per_epoch() / 1000000, " MegaBytes")
        print("data_recieved_per_epoch: ", Communication.get_data_recieved_per_epoch() / 1000000, "MegaBytes")
        data_send_per_epoch_total+=Communication.get_data_send_per_epoch()
        data_recieved_per_epoch_total+=Communication.get_data_recieved_per_epoch()

        if count_flops:
            print("MegaFLOPS_forward_epoch", flops_counter.flops_forward_epoch/1000000)
            print("MegaFLOPS_encoder_epoch", flops_counter.flops_encoder_epoch/1000000)
            print("MegaFLOPS_backprop_epoch", flops_counter.flops_backprop_epoch/1000000)
            print("MegaFLOPS_rest", flops_counter.flops_rest/1000000)
            print("MegaFLOPS_send", flops_counter.flops_send/1000000)
            print("MegaFLOPS_recieve", flops_counter.flops_recieve/1000000)

        if weights_and_biases and count_flops:
            wandb.log({"Batches Abortrate": batches_aborted / total_train_nr, "MegaFLOPS Client Encoder": flops_counter.flops_encoder_epoch/1000000,
            "MegaFLOPS Client Forward": flops_counter.flops_forward_epoch / 1000000,
            "MegaFLOPS Client Backprop": flops_counter.flops_backprop_epoch / 1000000, "MegaFLOPS Send": flops_counter.flops_send / 1000000,
            "MegaFLOPS Recieve": flops_counter.flops_recieve / 1000000},
            commit=False)

        global auc_train_log, accuracy_train_log, batches_abort_rate_total
        auc_train_log = epoch_auc
        accuracy_train_log = epoch_accuracy
        batches_abort_rate_total += batches_aborted / total_train_nr


def val_stage(s, pretraining=0):
    """
    Validation cycle for one epoch, started by the server
    :param s: socket
    :param content:
    """
    total_val_nr, val_loss_total = 0, 0
    precision_epoch, recall_epoch, f1_epoch, auc_val, accuracy_sklearn,  accuracy_custom = 0, 0, 0, 0, 0, 0
    acc, f1, auc = 0,0,0
    val_accuracy = Accuracy(num_classes=5, average=average_setting)
    val_f1 = F1Score(num_classes=5, average=average_setting)
    val_auc = AUROC(num_classes=5, average=average_setting)

    with torch.no_grad(): #No training involved, thus no gradient needed
        for b_t, batch_t in enumerate(val_loader):
            x_val, label_val = batch_t
            x_val, label_val = x_val.to(device), label_val.double().to(device)
            #optimizer.zero_grad()
            output_val = client(x_val, drop=False)
            if autoencoder:
                output_val = encode(output_val)

            msg = {'client_output_val/test': output_val,
                   'label_val/test': label_val,}
            Communication.send_msg(s, 1, msg)
            msg = Communication.recieve_msg(s)
            output_val_server = msg["output_val/test_server"]
            val_loss_total += msg["val/test_loss"]
            total_val_nr += 1

            #if b_t < 5:
            #    print("Label: ", label_val[b_t])
            #    print("Pred.: ", torch.round(output_val_server[b_t]))
            #    print("-------------------------------------------------------------------------")

            acc +=val_accuracy(output_val_server.detach().clone().cpu(), label_val.detach().clone().cpu().int()).numpy()
            f1 += val_f1(output_val_server.detach().clone().cpu(), label_val.detach().clone().cpu().int()).numpy()
            auc += val_auc(output_val_server.detach().clone().cpu(), label_val.detach().clone().cpu().int()).numpy()

    epoch_auc, epoch_accuracy, epoch_f1 = val_auc.compute(), val_accuracy.compute(), val_f1.compute()
    status_train = "auc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}".format(epoch_auc, epoch_accuracy, epoch_f1)
    print("status_val: ", status_train)

    if pretraining == 0 and weights_and_biases:
        wandb.define_metric("AUC_val", summary="max")
        wandb.define_metric("Accuracy_val", summary="max")
        #wandb.log({"AUC_val_max": epoch_auc, "Accuracy_val_max": epoch_accuracy}, commit=False)
        wandb.log({"Loss_val": val_loss_total / total_val_nr,
               "Accuracy_val": epoch_accuracy,
               "F1_val": epoch_f1,
               "AUC_val": epoch_auc,
               "AUC_train": auc_train_log,
               "Accuracy_train": accuracy_train_log})

    
    if not pretraining:
        client.to('cpu') #free up some gpu memory
        Communication.send_msg(s, 3, 0)


def test_stage(s, epoch):
    """
    Test cycle for one epoch, started by the server
    :param s: socket
    :param epoch: current epoch
    """
    loss_test = 0.0
    correct_test, total_test = 0, 0

    total_test_nr = 0
    test_accuracy = Accuracy(num_classes=5, average=average_setting)
    test_f1 = F1Score(num_classes=5, average=average_setting)
    test_auc = AUROC(num_classes=5, average=average_setting)
    with torch.no_grad():
        for b_t, batch_t in enumerate(val_loader):
            x_test, label_test = batch_t
            x_test, label_test = x_test.to(device), label_test.double().to(device)
            #optimizer.zero_grad()
            output_test = client(x_test, drop=False)
            if autoencoder:
                output_test = encode(output_test)

            msg = {'client_output_val/test': output_test,
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

            acc +=test_accuracy(output_test_server.detach().clone().cpu(), label_test.detach().clone().cpu().int()).numpy()
            f1 += test_f1(output_test_server.detach().clone().cpu(), label_test.detach().clone().cpu().int()).numpy()
            auc += test_auc(output_test_server.detach().clone().cpu(), label_test.detach().clone().cpu().int()).numpy()

    epoch_auc, epoch_accuracy, epoch_f1 = test_auc.compute(), test_accuracy.compute(), test_f1.compute()
    status_test = "auc: {:.4f}, Accuracy: {:.4f}, f1: {:.4f}".format(epoch_auc, epoch_accuracy, epoch_f1)
    print("status_test: ", status_test)


    global data_send_per_epoch_total, data_recieved_per_epoch_total, batches_abort_rate_total
    total_flops_model = flops_client_forward_total + flops_client_encoder_total + flops_client_backprop_total
    total_flops_all = total_flops_model+flops_client_send_total+flops_client_recieve_total+flops_client_rest_total
    data_transfer_per_epoch = data_send_per_epoch_total+data_recieved_per_epoch_total
    if count_flops:
        print("total FLOPs forward: ", flops_client_forward_total)
        print("total FLOPs encoder: ", flops_client_encoder_total)
        print("total FLOPs backprob: ", flops_client_backprop_total)
        print("total FLOPs Model: ", total_flops_model)
        print("total FLOPs: ", total_flops_all)
    print("Average data transfer/epoch: ", data_transfer_per_epoch / epoch / 1000000, " MB")
    print("Average dismissal rate: ", batches_abort_rate_total / epoch)

    if weights_and_biases:
        wandb.config.update({"Average data transfer/epoch (MB): ": data_transfer_per_epoch / epoch / 1000000,
                         "Average dismissal rate: ": batches_abort_rate_total / epoch,
                         "total_MegaFLOPS_forward": flops_client_forward_total/1000000, "total_MegaFLOPS_encoder": flops_client_encoder_total/1000000,
                         "total_MegaFLOPS_backprob": flops_client_backprop_total/1000000,"total_MegaFLOPS model": total_flops_model/1000000 ,"total_MegaFLOPS": total_flops_all/1000000})

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


def initIID():
    global X_train, X_val, y_val, y_train, y_test, X_test
    sampling_frequency = 100
    datafolder = ptb_path
    task = 'superdiagnostic'

    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=mlb_path)
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

    standard_scaler = pickle.load(open(scaler_path + '/standard_scaler.pkl', "rb"))

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

def init_nn_parameters():
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

    #Initialize client, optimizer, error-function, potentially encoder and grad_encoder
    global client
    if model == 'TCN': client = Models.Small_TCN_5(5, 12)
    if model == 'CNN': client = Models.Client()
    client.double().to(device)

    global optimizer
    #optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)
    optimizer = AdamW(client.parameters(), lr=lr)

    global error
    #error = nn.CrossEntropyLoss()
    error = nn.BCELoss()

    global data_send_per_epoch
    global data_recieved_per_epoch
    data_send_per_epoch = 0
    data_recieved_per_epoch = 0

    if autoencoder:
        global encode
        encode = Models.Encode()
        print("Start Encoder")
        encode.eval()
        print("Start eval")
        encode.double().to(device)

        global optimizerencode
        optimizerencode = Adam(encode.parameters(), lr=lr)  ###


def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """
    global label_sttc, label_hyp, label_mi, label_norm, label_cd
    label_sttc, label_hyp, label_mi, label_norm, label_cd = [],[],[],[],[]
    global X_train, X_val, y_val, y_train, y_test, X_test
    global flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total
    flops_client_forward_total, flops_client_encoder_total, flops_client_backprop_total, flops_client_send_total, flops_client_recieve_total, flops_client_rest_total = 0,0,0,0,0,0
    global data_send_per_epoch_total, data_recieved_per_epoch_total, batches_abort_rate_total, encoder_grad_server, epoch
    data_send_per_epoch_total, data_recieved_per_epoch_total, batches_abort_rate_total, encoder_grad_server, epoch = 0,0,0,0,0
    global flops_counter
    flops_counter = Flops.Flops(count_flops)

    initIID()
    init_nonIID()
    print_json()
    init_train_val_dataset()
    init_nn_parameters()

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
        Communication.send_msg(s, 2, initial_weights)
        Communication.send_msg(s, 3, 0)
        epoch = 0

    serverHandler(s)



if __name__ == '__main__':
    main()
