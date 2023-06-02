import argparse
import json
import time

import numpy as np
import Client
import os
#np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Client Launcher')
    parser.add_argument('--num_clients', type=int, default=1, help='Number of clients to launch')
    args = parser.parse_args()
    
    f = open(
        "settings.json",
    )
    data = json.load(f)
    num_malicious = data["num_malicious"]
    
    # selects num_malicious numbers out of range(1, args.num_clients + 1)
    malicious_ids = list(np.random.choice(range(1, args.num_clients + 1), num_malicious, replace=False))
    
    # Initialize clients datasets
    os.system("python3 client/Client.py --client_num 0 " + " --num_clients " + str(args.num_clients))
    
    cmd = ""
    
    for i in range(1, args.num_clients + 1):
        if i > 1 :
            cmd += " & "
        cmd += "python3 client/Client.py --client_num " + str(i) + " --num_clients " + str(args.num_clients)
        
        if i in malicious_ids:
            cmd += " --malicious"
            
    os.system(cmd)
    