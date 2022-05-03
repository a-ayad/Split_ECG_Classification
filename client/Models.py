import torch.nn as nn

class Client(nn.Module):
    """
    Client-Model:
    """
    def __init__(self, training=True):
        super(Client, self).__init__()
        self.conv1 = nn.Conv1d(12, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.4, training)
        self.conv2 = nn.Conv1d(192, 192, kernel_size=3, stride=2, dilation=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x, drop=True):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        if drop == True: x = self.drop1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x


class Encode(nn.Module):
    """
    Encoder-Model:
    """
    def __init__(self):
        super(Encode, self).__init__()
        self.conva = nn.Conv1d(192, 144, 2, stride=2,  padding=1)
        self.convb = nn.Conv1d(144, 96, 2, stride=2, padding=0)
        self.convc = nn.Conv1d(96, 48, 2, stride=2,  padding=0)
        self.convd = nn.Conv1d(48, 24, 2, stride=2, padding=0)

    def forward(self, x):
        x = self.conva(x)
        x = self.convb(x)
        x = self.convc(x)
        x = self.convd(x)
        return x


class Grad_Decoder(nn.Module):
    """
    Decoder-Model:
    """
    def __init__(self):
        super(Grad_Decoder, self).__init__()
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