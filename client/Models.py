import torch.nn as nn
import nemo

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


class Small_TCN_5(nn.Module):
    def __init__(self, classes, n_inputs):
        super(Small_TCN_5, self).__init__()
        # Hyperparameters for TCN
        Kt = 19
        pt = 0.3
        Ft = 11

        self.pad0 = nn.ConstantPad1d(padding=(Kt - 1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 1, kernel_size=19, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 1)

        # First block
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=1, bias=False)
        self.upsamplerelu = nn.ReLU()
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.pad1 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        self.add1 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd1 = nn.ReLU()

        # Second block
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        self.add2 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd2 = nn.ReLU()

    def forward(self, x, drop=True):
        # Now we propagate through the network correctly
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)

        # TCN
        # First block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        if drop == True: res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        if drop == True: res = self.dropout2(res)

        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)

        x = self.add1(x, res)
        x = self.reluadd1(x)

        # Second block
        res = self.pad3(x)
        # res = self.pad3(res)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        if drop == True: res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        if drop == True: res = self.dropout4(res)
        x = self.add2(x, res)
        x = self.reluadd2(x)
        return x

