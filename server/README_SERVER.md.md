# Advanced server side configurations

## Training parameter configuration
By default, Training and tcp paramters (learningrate, update threshold, update_mechanism, mechanism) are read from the parameter_server.json file.

Further, device, optimizer and lossfunction can be set int the main method of the server.py file.

```python
global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

global optimizer
optimizer = AdamW(server.parameters(), lr=lr)

global error
error = nn.BCELoss()
```
## Model
The model of the client can also be modified in the server.py file in the Server Class. 
Attention: the input shape has to match the dataset and the outputshape the decoder/client side model!
```python
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
```
## Costumize Autoencoder
Customizing the autoencoder is explaines in the advanced_settings file.

