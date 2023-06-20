import argparse
import json
import time

import numpy as np
import Client
import os
#np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Client Launcher')
    parser.add_argument('--malicious_ids',type=int, nargs='+', help='Ids of malicious clients')
    args = parser.parse_args()
    
    f = open(
        "settings.json",
    )
    data = json.load(f)
    num_malicious = data["num_malicious"]
    num_clients = data["nr_clients"]
    
    # selects num_malicious numbers out of range(args.num_clients)
    malicious_ids = args.malicious_ids
    if not malicious_ids:
        malicious_ids = list(np.random.choice(range(num_clients), num_malicious, replace=False))
    
    # Initialize clients datasets
    os.system("python3 client/Client.py --init_client --IID ")
    
    cmd = ""
    
    for i in range(num_clients):
        if i > 0 :
            cmd += " & "
        cmd += "python3 client/Client.py " + " --IID --weights_and_biases --average_setting micro "
        
        if i in malicious_ids:
            cmd += " --malicious"
            
    os.system(cmd)
    