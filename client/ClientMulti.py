import argparse
import json
import time

import numpy as np
import Client
import os
#np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':    
    f = open(
        "settings.json",
    )
    data = json.load(f)
    num_clients = data["nr_clients"]
    
    # Initialize clients datasets
    os.system("python3 client/Client.py --init_client")
    
    cmd = ""
    
    for i in range(num_clients):
        if i > 0 :
            cmd += " & "
        cmd += "python3 client/Client.py "
            
    os.system(cmd)
    