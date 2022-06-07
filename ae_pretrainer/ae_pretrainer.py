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
import utils
import ast
import random
import os.path
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
cwd = os.path.dirname(os.path.abspath(__file__))
mlb_path = os.path.join(cwd,  "..", "Benchmark", "output", "mlb.pkl")
scaler_path = os.path.join(cwd,  "..", "Benchmark", "output", "standard_scaler.pkl")
ptb_path = os.path.join(cwd,  "..", "server", "PTB-XL", "ptb-xl/")
path_ecg_synthetic = os.path.join(cwd, "..", "ECG-Synthetic")

batchsize = 64
test_batches = 1500
epoch = 500
lr = 0.001
prev_loss = 999
diverge_tresh = 1.1
lr_adapt = 0.5

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


class Encode192(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode192, self).__init__()
        self.conva = nn.Conv1d(192, 144, 2, stride=2,  padding=1)
        self.convb = nn.Conv1d(144, 96, 2, stride=2, padding=0)
        self.convc = nn.Conv1d(96, 48, 2, stride=2,  padding=0)
        self.convd = nn.Conv1d(48, 24, 2, stride=2, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        x = self.convd(x)
        #print("encode 4 Layer: ", x.size())
        #print("encode 5 Layer: ", x.size())
        return x

class Encode64(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode64, self).__init__()
        self.conva = nn.Conv1d(64, 32, 2, stride=2,  padding=0)
        self.convb = nn.Conv1d(32, 16, 2, stride=2, padding=0)
        self.convc = nn.Conv1d(16, 8, 2, stride=2,  padding=0)
        self.convd = nn.Conv1d(8, 4, 2, stride=1, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        x = self.convd(x)
        #print("encode 4 Layer: ", x.size())
        return x

class Encode32small(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode32small, self).__init__()
        self.conva = nn.Conv1d(32, 16, 2, stride=2,  padding=0)
        self.convb = nn.Conv1d(16, 8, 2, stride=2, padding=1)
        self.convc = nn.Conv1d(8, 4, 2, stride=2,  padding=1)

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        return x

class Encode32(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode32, self).__init__()
        self.conva = nn.Conv1d(32, 16, 2, stride=2, padding=0)
        self.convb = nn.Conv1d(16, 8, 2, stride=2,  padding=1)
        self.convc = nn.Conv1d(8, 8, 2, stride=2, padding=1)##
        self.convd = nn.Conv1d(8, 4, 2, stride=2, padding=0)  ##
        self.conve = nn.Conv1d(4, 4, 2, stride=2, padding=0)

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        x = self.convd(x)
        #print("encode 4 Layer: ", x.size())
        x = self.conve(x)
        #print("encode 5 Layer: ", x.size())
        return x


class Decode192(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode192, self).__init__()
        self.t_convb = nn.ConvTranspose1d(24, 48, 2, stride=2, padding=0)
        self.t_convc = nn.ConvTranspose1d(48, 96, 2, stride=2, padding=0)
        self.t_convd = nn.ConvTranspose1d(96, 144, 2, stride=2, padding=0)
        self.t_conve = nn.ConvTranspose1d(144, 192, 2, stride=2, padding=1)

    def forward(self, x):
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        x = self.t_convd(x)
        #print("decode 4 Layer: ", x.size())
        x = self.t_conve(x)
        #print("decode 4 Layer: ", x.size())
        return x

class Decode64(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode64, self).__init__()
        self.t_conva = nn.ConvTranspose1d(4, 8, 2, stride=1)
        self.t_convb = nn.ConvTranspose1d(8, 16, 2, stride=2)
        self.t_convc = nn.ConvTranspose1d(16, 32, 2, stride=2)
        self.t_convd = nn.ConvTranspose1d(32, 64, 2, stride=2)

    def forward(self, x):
        x = self.t_conva(x)
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        x = self.t_convd(x)
        #print("decode 4 Layer: ", x.size())
        return x

class Decode32small(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode32small, self).__init__()
        self.t_conva = nn.ConvTranspose1d(4, 8, 2, stride=2, padding=1)
        self.t_convb = nn.ConvTranspose1d(8, 16, 2, stride=2, padding=1)
        self.t_convc = nn.ConvTranspose1d(16, 32, 2, stride=2, padding=0)

    def forward(self, x):
        x = self.t_conva(x)
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        return x

class Decode32(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode32, self).__init__()
        self.t_conva = nn.ConvTranspose1d(4, 4, 2, stride=2, padding=0)
        self.t_convb = nn.ConvTranspose1d(4, 8, 2, stride=2, padding=0)
        self.t_convc = nn.ConvTranspose1d(8, 8, 2, stride=2, padding=1)
        self.t_convd = nn.ConvTranspose1d(8, 16, 2, stride=2,  padding=1)
        self.t_conve = nn.ConvTranspose1d(16, 32, 2, stride=2, padding=0)

    def forward(self, x):
        x = self.t_conva(x)
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        x = self.t_convd(x)
        #print("decode 4 Layer: ", x.size())
        x = self.t_conve(x)
        #print("decode 5 Layer: ", x.size())
        return x

class Grad_Encoder(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Grad_Encoder, self).__init__()
        self.conva = nn.Conv1d(192, 144, 2, stride=2,  padding=1)
        self.convb = nn.Conv1d(144, 96, 2, stride=2, padding=0)
        self.convc = nn.Conv1d(96, 48, 2, stride=2,  padding=0)
        self.convd = nn.Conv1d(48, 24, 2, stride=2, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        x = self.convd(x)
        #print("encode 4 Layer: ", x.size())
        #print("encode 5 Layer: ", x.size())
        return x

class Grad_Decoder(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Grad_Decoder, self).__init__()
        self.t_convb = nn.ConvTranspose1d(24, 48, 2, stride=2, padding=0)
        self.t_convc = nn.ConvTranspose1d(48, 96, 2, stride=2, padding=0)
        self.t_convd = nn.ConvTranspose1d(96, 144, 2, stride=2, padding=0)
        self.t_conve = nn.ConvTranspose1d(144, 192, 2, stride=2, padding=1)

    def forward(self, x):
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        x = self.t_convd(x)
        #print("decode 4 Layer: ", x.size())
        x = self.t_conve(x)
        #print("decode 4 Layer: ", x.size())
        return x

# just the part of the clientmidel before the autoencoder

class Client_PTB(nn.Module):
    """
    client model
    """
    def __init__(self):
        super(Client_PTB, self).__init__()
        self.conv1 = nn.Conv1d(12, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(192, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        #print("Client_PTB output", x.shape)
        return x


class Server_PTB(nn.Module):
    """
    client model
    """
    def __init__(self):
        super(Server_PTB, self).__init__()
        self.drop2 = nn.Dropout(0.4)
        self.conv3 = nn.Conv1d(192, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.drop3 = nn.Dropout(0.4)
        self.conv4 = nn.Conv1d(192, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
        self.relu4 = nn.ReLU()
        #self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.drop4 = nn.Dropout(0.4)
        self.conv5 = nn.Conv1d(192, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        self.relu5 = nn.ReLU()
        #self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.drop5 = nn.Dropout(0.4)
        self.conv6 = nn.Conv1d(192, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=2)
        #self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2)
        #self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.flatt = nn.Flatten(start_dim=1)
        self.linear2 = nn.Linear(in_features=192, out_features=5, bias=True)
    def forward(self, x):
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.pool3(x)
        x = self.drop3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        #x = self.pool4(x)
        x = self.drop4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        #x = self.pool5(x)
        x = self.drop5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.flatt(x)
        x = torch.sigmoid(self.linear2(x))
        return x


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
        for b, batch in enumerate(dataloaders['ae']):#AE_PTB_testdata):#dataloaders['ae']):
                x_train, label_train = batch
                #x_train = datalist[b]
                #print("x_train: ", x_train)
                x_train, label_train = x_train.double().to(device), label_train.double().to(device)
                #print("x_train shape: ", x_train.shape)

                x_train_2 = client_ptb(x_train)
                if b == 0:
                    print("input AE mean ", torch.mean(x_train_2).item())
                    print("input AE var: ", torch.var(x_train_2).item())
                #x_train = x_train.reshape(-1, 128)
                #print("output client: ",x_train_2.size())
                optimizerencode.zero_grad()
                optimizerdecode.zero_grad()
                output_train_enc = encode(x_train_2)
                #print(output_train_enc.shape)
                output_train = decode(output_train_enc)
                if b == 0:
                    print("output AE mean: ", torch.mean(output_train).item())
                    print("output AE var: ", torch.var(output_train).item())
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


    global X_train
    global X_val

    global y_val
    global y_train
    global y_test
    global X_test
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

    standard_scaler = pickle.load(open(scaler_path, "rb"))

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

    global client_ptb
    client_ptb = Client_PTB()
    client_ptb.double().to(device)

    global server_ptb
    server_ptb = Server_PTB()
    server_ptb.double().to(device)

    global encode
    encode = Encode192()
    encode.double().to(device)

    global decode
    decode = Decode192()
    decode.double().to(device)

    global encode_grad
    encode_grad = Grad_Encoder()
    encode_grad.double().to(device)

    global decode_grad
    decode_grad = Grad_Decoder()
    decode_grad.double().to(device)

    global scaler
    scaler = MinMaxScaler()

    global optimizerencode_grad
    optimizerencode_grad = Adam(encode_grad.parameters(), lr=0.001)###
    global optimizerdecode_grad
    optimizerdecode_grad = Adam(decode_grad.parameters(), lr=0.001)###
    global server_optimizer
    server_optimizer = Adam(server_ptb.parameters(), lr=0.001)  ###

    #start_training()
    start_training_grad()


if __name__ == '__main__':
    main()
