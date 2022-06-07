# Advanced client side configurations

## Dataset
The dataset is divided in 10 parts. As recommended on the official website, the first 9 parts are used for training, and the 10th part is used for testing and validation.

```python
# 1-9 for training
X_train = data[labels.strat_fold < 10]
global y_train
y_train = Y[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]
```

Furthermore a scaler is used on the samples, to create a uniform input shape for the neural network.

```python
X_train = utils.apply_standardizer(X_train, standard_scaler)
X_val = utils.apply_standardizer(X_val, standard_scaler)
```


# Create Datasets
```python
def init():
    train_dataset = PTB_XL('train')
    val_dataset = PTB_XL('val')
```

## Host configureation
By default, Host IP and Port are read from the parameter_client.json file. The paramter can also be set directly in the client.py file. 

```python
host = data["host"]
port = data["port"]
```

## Training parameter configuration
By default, Training and tcp paramters (learningratem batchsize, max recv) are read from the parameter_client.json file. The paramter can also be set directly in the client.py file, by changing the following variables.
```python
epoch = data["trainingepochs"]
lr = data["learningrate"]
batchsize = data["batchsize"]
batch_concat = data["batch_concat"]
max_recv = data["max_recv"]
```
Further, device, optimizer and lossfunction can be set int the main method of the client.py file.

```python
global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

global optimizer
optimizer = AdamW(client.parameters(), lr=lr)

global error
error = nn.BCELoss()
```
## Model
The model of the client can also be modified in the cleint.py file in the Client Class. 
Attention: the input shape has to match the dataset and the outputshape the encoder/ server side model!
```python
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
        print("Input: ", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        if drop == True: x = self.drop1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        print("Output: ", x.shape)
        return x
```

## Costumize Autoencoder
Customizing the autoencoder is explaines in the advanced_settings file.

