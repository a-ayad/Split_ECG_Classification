import torch.nn as nn
import torch
import nemo

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


class Small_TCN_5_Client(nn.Module):
    def __init__(self, classes, n_inputs):
        super(Small_TCN_5_Client, self).__init__()
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


class Small_TCN_5_Server(nn.Module):
    def __init__(self, classes, n_inputs ):
        super(Small_TCN_5_Server, self).__init__()
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


class EncodeTCN(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(EncodeTCN, self).__init__()
        self.conva = nn.Conv1d(11, 11, 4, stride=2,  padding=1)
        self.convb = nn.Conv1d(11, 8, 4, stride=2, padding=1)
        self.convc = nn.Conv1d(8, 5, 4, stride=2,  padding=1)
        #self.convd = nn.Conv1d(48, 24, 2, stride=2, padding=0)##

    def forward(self, x):
        x = self.conva(x)
        #print("encode 1 Layer: ", x.size())
        x = self.convb(x)
        #print("encode 2 Layer: ", x.size())
        x = self.convc(x)
        #print("encode 3 Layer: ", x.size())
        #x = self.convd(x)
        #print("encode 4 Layer: ", x.size())
        #print("encode 5 Layer: ", x.size())
        return x


class DecodeTCN(nn.Module):
    """
    decoder model
    """ 
    def __init__(self):
        super(DecodeTCN, self).__init__()
        self.t_convb = nn.ConvTranspose1d(5, 8, 4, stride=2, padding=1)
        self.t_convc = nn.ConvTranspose1d(8, 11, 4, stride=2, padding=1)
        self.t_convd = nn.ConvTranspose1d(11, 11, 4, stride=2, padding=1)
        #self.t_conve = nn.ConvTranspose1d(144, 192, 2, stride=2, padding=1)

    def forward(self, x):
        #print("decode 1 Layer: ", x.size())
        x = self.t_convb(x)
        #print("decode 2 Layer: ", x.size())
        x = self.t_convc(x)
        #print("decode 3 Layer: ", x.size())
        x = self.t_convd(x)
        #print("decode 4 Layer: ", x.size())
        #x = self.t_conve(x)
        #print("decode 4 Layer: ", x.size())
        return x