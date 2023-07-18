import argparse
from contextlib import closing
import json
import socket
from subprocess import Popen
import time
import os
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client")
    # Load the JSON file
    with open('default_settings.json') as file:
        config = json.load(file)

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments for each attribute in the JSON file
    parser.add_argument('--host', type=str, default=config['host'], help='Hostname')
    parser.add_argument('--port', type=int, default=config['port'], help='Port number')
    parser.add_argument('--max_recv', type=int, default=config['max_recv'], help='Maximum receive size')
    parser.add_argument('--batchsize', type=int, default=config['batchsize'], help='Batch size')
    parser.add_argument('--learningrate', type=float, default=config['learningrate'], help='Learning rate for general use')
    parser.add_argument('--learnrate_cnn', type=float, default=config['(Comment): Learnrate for CNN'], help='Learning rate for CNN')  # Commented attribute
    parser.add_argument('--model', type=str, choices=['CNN', 'TCN'], default=config['Model'], help='Model type (CNN or TCN)')  # Commented attribute
    parser.add_argument('--autoencoder', type=int, default=config['autoencoder'], help='Autoencoder setting')
    parser.add_argument('--autoencoder_train', type=int, default=config['autoencoder_train'], help='Autoencoder training setting')
    parser.add_argument('--count_flops', type=int, default=config['count_flops'], help='Count FLOPs setting')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--update_mechanism', type=str, choices=['static', 'other'], default=config['update_mechanism'], help='Update mechanism')  # Commented attribute
    parser.add_argument('--update_threshold', type=int, default=config['update_threshold'], help='Update threshold')
    parser.add_argument('--mechanism', type=str, default=config['mechanism'], help='Mechanism type')
    parser.add_argument('--nr_clients', type=int, default=config['nr_clients'], help='Number of clients')
    parser.add_argument('--num_classes', type=int, default=config['num_classes'], help='Number of classes')
    parser.add_argument('--pretrain_active', type=int, default=config['pretrain_active'], help='Pretrain active setting')
    parser.add_argument('--pretrain_epochs', type=int, default=config['pretrain_epochs'], help='Number of pretrain epochs')
    parser.add_argument('--mixed_with_IID_data', type=int, default=config['mixed_with_IID_data'], help='Mixed with IID data setting')
    parser.add_argument('--IID_percentage', type=float, default=config['IID_percentage'], help='IID percentage')
    parser.add_argument('--record_latent_space', action="store_true" ,default=config['record_latent_space'], help='Record latent space setting')
    parser.add_argument('--exp_name', type=str, default=config['exp_name'], help='Experiment name')
    parser.add_argument('--num_malicious', type=int, default=config['num_malicious'], help='Number of malicious instances')
    parser.add_argument('--data_poisoning_prob', type=float, default=config['data_poisoning_prob'], help='Data poisoning probability')
    parser.add_argument('--data_poisoning_method', type=str, default=config['data_poisoning_method'], help='Data poisoning method')
    parser.add_argument('--blending_factor', type=float, default=config['blending_factor'], help='Blending factor')
    parser.add_argument('--label_flipping_prob', type=float, default=config['label_flipping_prob'], help='Label flipping probability')
    parser.add_argument('--detect_anomalies', action="store_true" ,default=config['detect_anomalies'], help='Detect anomalies setting')
    parser.add_argument('--detection_scheduler', action="store_true" ,default=config['detection_scheduler'], help='Detection scheduler setting')
    parser.add_argument('--detection_tau', type=float, default=config['detection_tau'], help='Detection tau')
    parser.add_argument('--detection_tolerance', type=int, default=config['detection_tolerance'], help='Detection tolerance')
    parser.add_argument('--detection_window', type=int, default=config['detection_window'], help='Detection window')
    parser.add_argument('--detection_start', type=int, default=config['detection_start'], help='Detection start')
    parser.add_argument('--detection_similarity', type=str, default=config['detection_similarity'], help='Detection similarity')
    parser.add_argument('--detection_params', type=str, default=json.dumps(config['detection_params']), help='Detection params')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    
    for key in vars(args):
        config[key] = getattr(args, key)
    config["detection_params"] = json.loads(args.detection_params)
    
    # check if port is free, otherwise increment until free port is found
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        config["port"] = s.getsockname()[1]
        s.close()
        
    json.dump(config, open("settings.json", "w"), indent=4)
    
    # launches the server and clients in parallel in two different terminals
    cmd_server = "python3 ./server/Server.py"
    cmd_client = "python3 ./client/ClientMulti.py"
    Popen(cmd_server, shell=True)	
    time.sleep(5)	
    os.system(cmd_client)