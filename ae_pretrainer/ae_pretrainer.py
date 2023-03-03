import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import torchvision
from torchvision import transforms
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import time
import wfdb
import ast
import random
import os.path
import AE_Models as Models
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import sys
# Set path variables to load the PTB-XL dataset and its scaler
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(cwd)
mlb_path = os.path.join(cwd, "mlb.pkl")
scaler_path = os.path.join(cwd)
ptb_path = os.path.join(cwd, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/")
output_path = os.path.join(cwd, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3", "output/")
path_ecg_synthetic = os.path.join(cwd, "..", "ECG-Synthetic")
import utils

batchsize = 64
test_batches = 1500
epoch = 500
lr = 0.001
prev_loss = 999
diverge_tresh = 1.1
lr_adapt = 0.5
model = 'TCN' #Set Model to 'TCN' or 'CNN'
save_model = 0

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
            #if idx == 50:
            #    print("max: ",np.max(self.X_test[idx]))
            #    print("min: ",np.min(self.X_test[idx]))
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

def init():
    AE_test_dataset = PTB_XL('test')
    global AE_PTB_testdata
    AE_PTB_testdata = torch.utils.data.DataLoader(AE_test_dataset, batch_size=batchsize, shuffle=True)


class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = path_ecg_synthetic + '/attn.pth'
    lstm_state_path = path_ecg_synthetic + '/lstm.pth'
    cnn_state_path = path_ecg_synthetic + '/cnn.pth'

    attn_logs = path_ecg_synthetic + '/attn.csv'
    lstm_logs = path_ecg_synthetic + '/lstm.csv'
    cnn_logs = path_ecg_synthetic + '/cnn.csv'

    train_csv_path = path_ecg_synthetic + '/mitbih_with_syntetic_train.csv'
    test_csv_path = path_ecg_synthetic + '/mitbih_with_syntetic_test.csv'


def random_num():
    tensor = torch.tensor((), dtype=torch.float32)
    data = tensor.new_full((12, 1000), 0.1)
    for b in range(12):
        #for a in range(1000):
        data[b] = torch.tensor((np.random.normal(0, 6, 1000)), dtype=torch.float32)
    #data = data.expand(12, 1000)
    return data


def newshape(x):
    tensor = torch.tensor((), dtype=torch.float32)
    data = tensor.new_full((1, 1000), 0.1)#
    for a in range(65):
        data[0][a] = x[0][0]
    for a, value in enumerate(x[0]):
        data[0][65 + 5 * a] = value
        data[0][65 + 5 * a + 1] = value
        data[0][65 + 5 * a + 2] = value
        data[0][65 + 5 * a + 3] = value
        data[0][65 + 5 * a + 4] = value
    data = data.expand(12, 1000)
    return data

class ECGDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.data_columns].astype('float32')
        signal = torch.FloatTensor(np.array([signal.values]))

        #if idx == 0:
        #    print("ECG: ", signal)
        signal = newshape(signal) # reshape to match input of PTB-XL
        #signal = random_num()
        #if idx == 0:
        #    print("ECG: ", signal)
        target = torch.LongTensor(np.array(self.df.loc[idx, 'class']))
        return signal, target

    def __len__(self):
        return len(self.df)

def get_dataloader(phase: str, batch_size: int = 64) -> DataLoader:
    '''
    Dataset and DataLoader.
    Parameters:
        pahse: training or validation phase.
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    df = pd.read_csv(config.train_csv_path)
    print("Total Dataset size: ", df.shape)
    train_df1, ae_df = train_test_split(
        df, test_size=0.1, random_state=config.seed, stratify=df['label']
    )
    print(train_df1.shape)
    train_df1 = train_df1.reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_df1, test_size=0.15, random_state=config.seed, stratify=train_df1['label']
    )
    #print(df.shape)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    ae_df = ae_df.reset_index(drop=True)
    if phase == 'ae':
        df = ae_df
        print("Dataset AE size: ", df.shape)
    else:
        df = train_df if phase == 'train' else val_df
    dataset = ECGDataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
    return dataloader


def test_PTB(loss_last_10_ptb, e):
    batch_nr = 0
    loss_ptb_epoch = 0
    with torch.no_grad():
        for b, batch in enumerate(AE_PTB_testdata):
            x_train, label_train = batch
            x_train, label_train = x_train.double().to(device), label_train.double().to(device)
            x_train_2 = client_ptb(x_train)
            output_train_enc = encode(x_train_2)
            output_train = decode(output_train_enc)
            loss_train = error_AE(x_train_2, output_train)  # calculates cross-entropy loss
            loss_ptb_epoch += loss_train
            batch_nr += 1
            loss_last_10_ptb.append(loss_train)

        if e == 0:
            print("x_train_ptb: ", x_train.shape)
        loss_10 = 0
        if e % 10 == 0:
            for loss in loss_last_10_ptb:
                loss_10 += loss / batch_nr
            #print("epoch: {}, train-loss_10_ptb: {:.6f}".format(e, loss_10 / 10))
            final_loss = loss_10 / 10
            loss_last_10_ptb.clear()

        print("epoch: {}, train-loss_PTB: {:.6f}".format(e, loss_ptb_epoch / batch_nr))
        data = loss_ptb_epoch / batch_nr
        return loss_last_10_ptb, data


def start_training():
    """
    function, which does the training process for the autoencoder and seves the weights at the end of every epoch,
    if no divergenvce occurs

    --> adaptive learning rate, depending on divergences --> in case of divergence, results of the epoch are dismissed,
    and the training of this epoch restarts with a smaller learning rate

    """
    prevloss = prev_loss
    learnrate = lr###
    optimizerencode = Adam(encode.parameters(), lr=learnrate)###
    optimizerdecode = Adam(decode.parameters(), lr=learnrate)###
    loss_last_10 = []
    loss_last_10_ptb = []
    total_time = time.time()
    data = []
    datalist= []
    """
    phases = ['train', 'val', 'ae']
    dataloaders = {
        phase: get_dataloader(phase, batchsize) for phase in phases
    }
    """

    time_train_epoch = time.time()
    for e in range(epoch):
        #x_test, label_test = x_tester, label_tester
        #x_test, label_test = x_test.to(device), label_test.to(device)
        loss_train_epoch = 0
        batch_nr = 0
        if e == 1:
            time_train_epoch = time.time() - time_train_epoch
            print("estimated_time_total: ", time_train_epoch*epoch/60, " min")
        for b, batch in enumerate(AE_PTB_testdata): #dataloaders['ae']):AE_PTB_testdata):#dataloaders['ae'])
                x_train, label_train = batch
                #x_train = datalist[b]
                #print("x_train: ", x_train)
                x_train, label_train = x_train.double().to(device), label_train.double().to(device)
                #print("x_train shape: ", x_train.shape)

                x_train_2 = client_ptb(x_train)
                #if b == 0:
                #    print("input AE mean ", torch.mean(x_train_2).item())
                #    print("input AE var: ", torch.var(x_train_2).item())
                #x_train = x_train.reshape(-1, 128)
                #print("output client: ",x_train_2.size())
                optimizerencode.zero_grad()
                optimizerdecode.zero_grad()
                if (e == 1 and b == 1):
                    print("shape: ", x_train_2.shape)
                output_train_enc = encode(x_train_2)
                if (e == 1 and b == 1):
                    print("encoded shape: ", output_train_enc.shape)
                #print(output_train_enc.shape)
                output_train = decode(output_train_enc)
                #if b == 0:
                #    print("output AE mean: ", torch.mean(output_train).item())
                #    print("output AE var: ", torch.var(output_train).item())
                #if b == 0:
                #    print(output_train)
                #print("output decoder: ",output_train.size())
                loss_train = error_AE(x_train_2, output_train)  # calculates cross-entropy loss
                #print("loss_train", loss_train)
                loss_train.backward()

                loss_train_epoch += loss_train

                optimizerencode.step()
                optimizerdecode.step()
                batch_nr += 1
                loss_last_10.append(loss_train)

        if e == 0:
            print("x_train: ", x_train.shape)

        print("epoch: {}, train-loss: {:.6f}".format(e, loss_train_epoch / batch_nr))

        if (e % 300 == 0):
            print("output encoder: ", output_train_enc.size())
        #if e == 99:
        #    print(loss_last_10)
        loss_10 = 0
        if e % 10 == 0:
            for loss in loss_last_10:
                loss_10 += loss / batch_nr
            #print("epoch: {}, train-loss_10: {:.6f}".format(e, loss_10 / 10))
            final_loss = loss_10 / 10
            loss_last_10.clear()

        #loss_last_10_ptb, loss_train_epoch_ptb = test_PTB(loss_last_10_ptb, e)


        #if loss_train_epoch_ptb < 0.3: #with same dataset : 0.174317
            #4 Layers: 0.217885 (24), 0.194269 (48), same dataset: 0.132442
            #3 Layers: 0.187612 (24), 0.182876 (48)
            #3Layers same Dataset: 0.128
            #Saved 51 epochs: 0.216848
            #print("saved")
        if model == 'TCN' and save_model == 1:
            torch.save(encode.state_dict(), "convencoder_TCN.pth")
            torch.save(decode.state_dict(), "convdecoder_TCN.pth")
        if model == 'CNN' and save_model == 1:
            torch.save(encode.state_dict(), "../client/convencoder_medical.pth")        
            torch.save(decode.state_dict(), "../server/convdecoder_medical.pth")

    print("total time:", time.time()-total_time)


def start_training3():
    """
    function, which does the training process for the autoencoder and seves the weights at the end of every epoch,
    if no divergenvce occurs

    --> adaptive learning rate, depending on divergences --> in case of divergence, results of the epoch are dismissed,
    and the training of this epoch restarts with a smaller learning rate

    """
    prevloss = prev_loss
    learnrate = lr###
    optimizerencode = Adam(encode.parameters(), lr=learnrate)###
    optimizerdecode = Adam(decode.parameters(), lr=learnrate)###
    loss_last_10 = []
    loss_last_10_ptb = []
    total_time = time.time()
    data = []
    datalist= []
    for a in range(1):
        for b in range(64):
            data.append(random_num())
        datalist.append(data)
    print(data[0])
    print(len(datalist))

    phases = ['train', 'val', 'ae']
    dataloaders = {
        phase: get_dataloader(phase, batchsize) for phase in phases
    }
    for b, batch in enumerate(AE_PTB_testdata):
        pass


    time_train_epoch = time.time()
    for e in range(epoch):
        #x_test, label_test = x_tester, label_tester
        #x_test, label_test = x_test.to(device), label_test.to(device)
        loss_train_epoch = 0
        batch_nr = 0
        if e == 1:
            time_train_epoch = time.time() - time_train_epoch
            print("estimated_time_total: ", time_train_epoch*epoch/60, " min")
        for b, batch in enumerate(AE_PTB_testdata):#dataloaders['ae']):
                x_train, label_train = batch
                #x_train = datalist[b]
                #print("x_train: ", x_train)
                x_train, label_train = x_train.double().to(device), label_train.double().to(device)
                if len(x_train) != 64:
                    break
                #print("x_train shape: ", x_train.shape)

                x_train_2 = get_grad(x_train, label_train)
                #x_train = x_train.reshape(-1, 128)
                #print("output client: ",x_train_2.size())
                optimizerencode.zero_grad()
                optimizerdecode.zero_grad()
                output_train_enc = encode(x_train_2)
                #print(output_train_enc.shape)
                output_train = decode(output_train_enc)
                #if b == 0:
                #    print(output_train)
                #print("output decoder: ",output_train.size())
                loss_train = error_AE(x_train_2, output_train)  # calculates cross-entropy loss
                #print("loss_train", loss_train)
                loss_train.backward()

                loss_train_epoch += loss_train

                optimizerencode.step()
                optimizerdecode.step()
                batch_nr += 1
                loss_last_10.append(loss_train)

        if e == 0:
            print("x_train: ", x_train.shape)

        print("epoch: {}, train-loss: {:.6f}".format(e, loss_train_epoch / batch_nr))

        if (e % 300 == 0):
            print("output encoder: ", output_train_enc.size())
        #if e == 99:
        #    print(loss_last_10)
        loss_10 = 0
        if e % 10 == 0:
            for loss in loss_last_10:
                loss_10 += loss / batch_nr
            #print("epoch: {}, train-loss_10: {:.6f}".format(e, loss_10 / 10))
            final_loss = loss_10 / 10
            loss_last_10.clear()

        loss_last_10_ptb, loss_train_epoch_ptb = test_PTB(loss_last_10_ptb, e)


        #if loss_train_epoch_ptb < 0.3: #with same dataset : 0.174317
            #4 Layers: 0.217885 (24), 0.194269 (48), same dataset: 0.132442
            #3 Layers: 0.187612 (24), 0.182876 (48)
            #3Layers same Dataset: 0.128
            #Saved 51 epochs: 0.216848
            #print("saved")
            #torch.save(encode.state_dict(), "../client/convencoder_medical.pth")
            #encode2 = Encode64()
            #encode2.load_state_dict(torch.load("./convencoder_medical.pth"))
            #encode2.eval()
            #torch.save(decode.state_dict(), "../server/convdecoder_medical.pth")
            #prevloss = loss_test

    print("total time:", time.time()-total_time)


def get_grad(x_train, label_train):
    #label_train = get_label_batch()
    #label_train = label_train.double().to(device)
    server_optimizer.zero_grad()

    # Client part
    activations = client_ptb(x_train)
    server_inputs = activations.detach().clone()

    # Server part
    server_inputs = Variable(server_inputs, requires_grad=True)
    outputs = server_ptb(server_inputs)
    # print("output shape", outputs.shape)
    # print("label", label_train.shape)
    loss = error(outputs, label_train)
    loss.backward()
    return server_inputs.grad


def get_label_batch():
    tensor = torch.tensor((), dtype=torch.int32)
    data = tensor.new_full((64, 5), 0)  #
    for a in range(64):
        for b in range(5):
            data[a][b] = torch.tensor((np.random.randint(0, 2)))
    #print(data)
    return data


def start_training_grad():
    """
    function, which does the training process for the autoencoder and seves the weights at the end of every epoch,
    if no divergenvce occurs

    --> adaptive learning rate, depending on divergences --> in case of divergence, results of the epoch are dismissed,
    and the training of this epoch restarts with a smaller learning rate

    """
    prevloss = prev_loss
    learnrate = lr###
    client_optimizer = Adam(client_ptb.parameters(), lr=learnrate)###
    server_optimizer = Adam(server_ptb.parameters(), lr=learnrate)###

    phases = ['train', 'val', 'ae']
    dataloaders = {
        phase: get_dataloader(phase, batchsize) for phase in phases
    }

    for e in range(epoch):
        #x_test, label_test = x_tester, label_tester
        #x_test, label_test = x_test.to(device), label_test.to(device)
        loss_epoch = 0
        loss_train = 0
        batch_nr = 0
        loss_minmax_total = 0
        for b, batch in enumerate(AE_PTB_testdata):#dataloaders['ae']):
                x_train, label_train = batch
                x_train = x_train.double().to(device)
                #label_train = get_label_batch()
                label_train = label_train.double().to(device)
                #print(x_train.shape)
                #print(len(x_train))
                if len(x_train) != 64:
                    break

                #x_train = x_train.reshape(-1, 128)
                #print("output client: ",x_train.size())

                #client_optimizer.zero_grad()
                server_optimizer.zero_grad()

                # Client part
                activations = client_ptb(x_train)
                server_inputs = activations.detach().clone()

                # Server part
                server_inputs = Variable(server_inputs, requires_grad=True)
                outputs = server_ptb(server_inputs)
                #print("output shape", outputs.shape)
                #print("label", label_train.shape)
                loss = error(outputs, label_train)
                loss_epoch += loss
                loss.backward()
                server_inputs_AE = server_inputs.grad
                minmax_inputs = server_inputs_AE.detach().clone().cpu()
                server_optimizer.step()
                #if b == 0:
                    #print("pre AE: ", server_inputs_AE[5])

                loss_train_inc, grad_post = AE_grad(grad_preprocessing(server_inputs_AE.detach().clone().cpu()), b)
                loss_train += loss_train_inc
                #print("post:",grad_post.shape)
                #print("pre:",minmax_inputs.shape)
                loss_minmax = error_MinMax(minmax_inputs, grad_post)
                loss_minmax_total += loss_minmax
                batch_nr +=1

                #running_loss += loss.item()
                #total_samples += labels.shape[0]
                #print("grad: ", server_inputs.grad)

                # Client part
                #activations.backward(output_decode_grad)
                #client_optimizer.step()
                #output_train_enc = encode(x_train)
                #output_train = decode(output_train_enc)
                #print("output decoder: ",output_train.size())
                #loss_train = error(x_train, output_train)  # calculates cross-entropy loss
                #loss_train.backward()

                #optimizerencode.step()
                #optimizerdecode.step()

        print("epoch: {}, train-loss: {:.6f}".format(e + 1, loss_epoch/batch_nr))
        print("epoch: {}, train-loss_AE: {:.6f}".format(e + 1, loss_train/batch_nr))
        print("epoch: {}, train-minmax_AE: {:.6f}".format(e + 1, loss_minmax_total/batch_nr*1000000))

       # if loss_test <= prevloss*diverge_tresh:
        torch.save(encode_grad.state_dict(), "../server/grad_encoder_medical.pth")
            #encode2 = Encode64()
            #encode2.load_state_dict(torch.load("./convencoder_medical.pth"))
            #encode2.eval()

        torch.save(decode_grad.state_dict(), "../client/grad_decoder_medical.pth")
            #prevloss = loss_test


def grad_preprocessing(grad):
    grad_new = grad.numpy()
    for a in range(64):
        grad_new[a] = scaler.fit_transform(grad[a])
    return grad_new

def grad_postprocessing(grad):
    grad_new = grad.numpy()
    for a in range(64):
        #scaler.fit(grad[a])
        grad_new[a] = scaler.inverse_transform(grad[a])
    grad_new = torch.DoubleTensor(grad_new)
    return grad_new


def AE_grad(server_inputs, b):
    optimizerencode_grad.zero_grad()
    optimizerdecode_grad.zero_grad()

    server_inputs = torch.DoubleTensor(server_inputs).to(device)

    output_encode_grad = encode_grad(server_inputs)
    output_decode_grad = decode_grad(output_encode_grad)

    grad_post = grad_postprocessing(output_decode_grad.detach().clone().cpu())

    loss_train = error_AE(server_inputs, output_decode_grad)  # calculates cross-entropy loss
    loss_train.backward()

    optimizerencode_grad.step()
    optimizerdecode_grad.step()

    return loss_train, grad_post


def main():
    """
    initialize device, client model, encoder and decoder and starts the training process
    """
    global config
    config = Config()


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

    X_test = data[labels.strat_fold == 10]
    y_test = Y[labels.strat_fold == 10]

    num_classes = 5  # <=== number of classes in the finetuning dataset
    input_shape = [1000, 12]  # <=== shape of samples, [None, 12] in case of different lengths

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    import pickle

    standard_scaler = pickle.load(open(scaler_path + '/standard_scaler.pkl', "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)
    X_test = utils.apply_standardizer(X_test, standard_scaler)


    init()

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda:0
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global error
    #error = nn.MSELoss()
    error = nn.BCELoss()

    global error_AE
    error_AE = nn.MSELoss()

    global error_MinMax
    error_MinMax = nn.MSELoss()
    #error_AE = nn.CrossEntropyLoss()

    global client_ptb, server_ptb, encode, decode, decode_grad, encode_grad
    if model == 'TCN':
        client_ptb = Models.Small_TCN_5_Client(5, 12).double().to(device)
        server_ptb = Models.Small_TCN_5_Server(5, 12).double().to(device)
        encode = Models.EncodeTCN().double().to(device)
        decode = Models.DecodeTCN().double().to(device)
        encode_grad = Models.Grad_Encoder().double().to(device)
        decode_grad = Models.Grad_Decoder().double().to(device)
    else:
        client_ptb = Models.Client_PTB().double().to(device)
        server_ptb = Models.Server_PTB().double().to(device)
        encode = Models.Encode192().double().to(device)
        decode = Models.Decode192().double().to(device)
        encode_grad = Models.Grad_Encoder().double().to(device)
        decode_grad = Models.Grad_Decoder().double().to(device)


    global scaler
    scaler = MinMaxScaler()

    global optimizerencode_grad
    optimizerencode_grad = Adam(encode_grad.parameters(), lr=0.001)###
    global optimizerdecode_grad
    optimizerdecode_grad = Adam(decode_grad.parameters(), lr=0.001)###
    global server_optimizer
    server_optimizer = Adam(server_ptb.parameters(), lr=0.001)  ###

    start_training()
    #start_training_grad()


if __name__ == '__main__':
    main()
