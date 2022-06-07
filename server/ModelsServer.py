import torch.nn as nn
import torch

class Decode(nn.Module):
    """
    Decoder model
    """
    def __init__(self):
        super(Decode, self).__init__()
        self.t_convb = nn.ConvTranspose1d(24, 48, 2, stride=2, padding=0)
        self.t_convc = nn.ConvTranspose1d(48, 96, 2, stride=2, padding=0)
        self.t_convd = nn.ConvTranspose1d(96, 144, 2, stride=2, padding=0)
        self.t_conve = nn.ConvTranspose1d(144, 192, 2, stride=2, padding=1)

    def forward(self, x):
        x = self.t_convb(x)
        x = self.t_convc(x)
        x = self.t_convd(x)
        x = self.t_conve(x)
        return x


class Grad_Encoder(nn.Module):
    """
    Encoder model
    """
    def __init__(self):
        super(Grad_Encoder, self).__init__()
        self.conva = nn.Conv1d(192, 144, 2, stride=2,  padding=1)
        self.convb = nn.Conv1d(144, 96, 2, stride=2, padding=0)
        self.convc = nn.Conv1d(96, 48, 2, stride=2,  padding=0)
        self.convd = nn.Conv1d(48, 24, 2, stride=2, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        x = self.convb(x)
        x = self.convc(x)
        x = self.convd(x)
        return x


class Server(nn.Module):
    """
    Server model
    """
    def __init__(self):
        super(Server, self).__init__()
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
    def forward(self, x, drop = True):
        if drop == True: x = self.drop2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.pool3(x)
        if drop == True: x = self.drop3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        #x = self.pool4(x)
        if drop == True: x = self.drop4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        #x = self.pool5(x)
        if drop == True: x = self.drop5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.flatt(x)
        x = torch.sigmoid(self.linear2(x))
        return x