import torch.nn as nn
import torch
import nemo

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


class Small_TCN_5(nn.Module):
    def __init__(self, classes, n_inputs ):
        super(Small_TCN_5, self).__init__()
        # Hyperparameters for TCN
        Kt = 19
        pt = 0.3
        Ft = 11

        # Third block
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)
        self.add3 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd3 = nn.ReLU()

        # fourth block
        dilation = 8
        self.pad7 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv7 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm7 = nn.BatchNorm1d(num_features=Ft)
        self.act7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=pt)
        self.pad8 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv8 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm8 = nn.BatchNorm1d(num_features=Ft)
        self.act8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=pt)
        self.add4 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd4 = nn.ReLU()

        # fifth block
        dilation = 16
        self.pad9 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv9 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm9 = nn.BatchNorm1d(num_features=Ft)
        self.act9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=pt)
        self.pad10 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv10 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm10 = nn.BatchNorm1d(num_features=Ft)
        self.act10 = nn.ReLU()
        self.dropout10 = nn.Dropout(p=pt)
        self.add5 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd5 = nn.ReLU()

        # Last layer
        self.linear = nn.Linear(in_features=Ft*1000, out_features=classes, bias=False) #Ft * 250

    def forward(self, x, drop=True):
        # Now we propagate through the network correctly

        # Third block
        res = self.pad5(x)
        # res = self.pad5(res)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        if drop == True: res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        if drop == True: res = self.dropout6(res)
        x = self.add3(x, res)
        x = self.reluadd3(x)

        # Fourth block
        res = self.pad7(x)
        # res = self.pad5(res)
        res = self.conv7(res)
        res = self.batchnorm7(res)
        res = self.act7(res)
        if drop == True: res = self.dropout7(res)
        res = self.pad8(res)
        res = self.conv8(res)
        res = self.batchnorm8(res)
        res = self.act8(res)
        if drop == True: res = self.dropout8(res)
        x = self.add4(x, res)
        x = self.reluadd4(x)

        """
        # Fifth block
        res = self.pad9(x)
        # res = self.pad5(res)
        res = self.conv9(res)
        res = self.batchnorm9(res)
        res = self.act9(res)
        res = self.dropout9(res)
        res = self.pad10(res)
        res = self.conv10(res)
        res = self.batchnorm10(res)
        res = self.act10(res)
        res = self.dropout10(res)
        x = self.add5(x, res)
        x = self.reluadd5(x)
        """

        # Linear layer to classify
        x = x.flatten(1)
        o = self.linear(x)
        o = torch.sigmoid(o)
        return o  # Return directly without softmax

