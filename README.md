# Efficient and Private ECG Classification on the Edge Using a Modified Split Learning Mechanism
## Split ECG (S-ECG)
This is the repository containing the code for this paper "Efficient and Private ECG Classification on the Edge Using a Modified Split Learning Mechanism"


## Requirements

### Server
* Python
#### Packages
* socket
* struct
* pickle
* numpy
* json
* torch
* threading
* time

### Client
* Python
#### Packages
* socket
* struct
* pickle
* time
* json
* numpy
* matplotlib
* torch
* torchvision
* sys


## Simple usage

Here's a brief overview of how you can use this project to run split learning on a server and a client.

### Download the dataset
The PTB=XL dataset can be downloaded from 
[link](https://physionet.org/content/ptb-xl/1.0.1/). download and unpack the zip file. Copy the folders "testset" and "trainingset" into the ./server/ directory in the project.

### Set client side configurations

To set the client side configuraitons, the file parameter_client.json can be edited.  

The initial settings are:

```json
{
    "host": "169.254.123.80",
    "port": 10087,
    "max_recv": 4096,

    "batchsize": 64,
    "learningrate": 0.0005,

    "count_flops": 0,

    "autoencoder": 0,
}

```
To configure the connection, "host" needs to be set to the ip adderess of the server and the port, to the port that will be configured at the server.

The max_recv property is the bit rate of the tcp port.


To adjust the training parameter, the bacthsize can be set as integer values and the learningrate as a float value.
Furthermore at the client, the computed FLOPs can be measured.

To reproduce the different experiments, here the autoencoder can be activated or deactivated (settings: (1 or 0)).


### Set serverside configurations

To set the server side configuraitons, the file parameter_server.json can be edited.  

The initial settings are:
```json
{
    "max_nr_clients": 5,
    "host": "0.0.0.0",
    "port": 10087,
    "max_recv": 4096,

    "learningrate": 0.0005,
    "epochs": 30,

    "autoencoder": 0,
    "update_mechanism": "static",
    "update_threshold": 0,
    "mechanism": "none",
}
```

max_nr_clients is the maximal number of clients, that can be connected to the server at the same time.
To configer the server, the host is initially set to "0.0.0.0" (localhost).
Also the Port has to be set.
The max_recv property is the bit rate of the tcp port.

One can set the server side learningrate manually (in all the configurations of the corresponding paper, the same learning rate for server and client was chosen).


Also, the update_mechanism can be configured (static or adaptive):
In the static setting: 
If the loss of a batch exceeds this threshold, the gradients are not sent back to the client (client side update skipped).
In the adaptive setting:
Change the mechanism to "linear", "sigmoid" or "none":
The adaptive threshold changes over the course of training and according to the value of the mechanism, a percentage of updates per epoch are skipped.

### Running the model

To run the model, the server.py script has to be run the server. Afterwards, the client.py script needs to be run at the client




## Authors
Ahmad Ayad, Marian Frei, Melvin Renner, Zhenghang Zhong
