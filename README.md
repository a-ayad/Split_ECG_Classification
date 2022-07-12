# Efficient and Private ECG Classification on the Edge Using a Modified Split Learning Mechanism
## Split ECG (S-ECG)
This is the repository containing the code for this paper "Efficient and Private ECG Classification on the Edge Using a Modified Split Learning Mechanism"


## Simple usage

Here's a brief overview of how you can use this project to run split learning on a server and a client.

### Download the dataset
The PTB=XL dataset can be downloaded from 
[link](https://physionet.org/content/ptb-xl/1.0.1/). download and unpack the zip file. Place it with its original name "ptb-xl-a-large-publicly..." in the working directory

### Set configurations

To set the client side configuraitons, the file settings.json can be edited.  

The initial settings are:

```json
{
    "host": "169.254.123.156",
    "port": 10089,
    "max_recv": 4096,

    "batchsize": 64,
    "learningrate": 0.001, 
    "(Comment): Learnrate for CNN": 0.0005,

    "Model": "CNN",
    "(Comment): For Model, TCN and CNN are viable": 0,

    "autoencoder": 0,
    "count_flops": 0,

    "epochs": 30,
    "update_mechanism": "static",
    "update_mechanism: static to use the update_threshold without a mechanism": 0,
    "update_threshold": 0,
    "mechanism": "sigmoid",

    
    "(Comment): To use multiple clients increase the nr_clients to more than 1": 0,
    "nr_clients": 1,
    "(Comment): Settings for non-IID & multiple clients:": 0,
    "num_classes": 2,
    "pretrain_active": 0,
    "pretrain_epochs": 30,
    "mixed_with_IID_data": 0
}

```
To configure the connection, "host" needs to be set to the ip adderess of the server and the port, to the port that will be configured at the server.

The max_recv property is the bit rate of the tcp port.


To adjust the training parameter, the batchsize can be set as integer values and the learningrate as a float value.
Furthermore at the client, the computed FLOPs can be measured.

To reproduce the different experiments, here the autoencoder can be activated or deactivated (settings: (1 or 0)).


Also, the update_mechanism can be configured (static or adaptive):
In the static setting: 
If the loss of a batch exceeds this threshold, the gradients are not sent back to the client (client side update skipped).
In the adaptive setting:
Change the mechanism to "linear", "sigmoid" or "none":
The adaptive threshold changes over the course of training and according to the value of the mechanism, a percentage of updates per epoch are skipped.

### Running the model

To run the model, the server.py script has to be started on the server. Afterwards, the client.py script needs to be started on the client.

For multiplee clients, use the server.py and the client1-5.py.
In the settings.json, the number of clients needs to be set accordingly.




## Authors
Ahmad Ayad, Marian Frei, Melvin Renner, Zhenghang Zhong
