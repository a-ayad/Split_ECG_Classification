import argparse
import Client
import os
#np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Client Launcher')
    parser.add_argument('--num_clients', type=int, default=1, help='Number of clients to launch')
    args = parser.parse_args()
    
    cmd = "python3 client/Client.py --client_num 1"
    for i in range(2, args.num_clients + 1):
        cmd += " & "
        cmd += "python3 client/Client.py --client_num " + str(i)
        
    os.system(cmd)
    