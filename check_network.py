from Embedd_Network_model import *
import torch
import argparse
parser = argparse.ArgumentParser(description='ニューラルネットワーク確認コード')
parser.add_argument('--model_name', help='モデル名')
args = parser.parse_args()
print(args)
#net = Net(10173,10,1)
net = New_Dual_Net(100)
model_name = 'value_net.pth'
if args.model_name is not None:
    model_name = args.model_name
PATH = 'model/'+ model_name
net.load_state_dict(torch.load(PATH))
for key in list(net.state_dict().keys()):
    print(key,net.state_dict()[key].size())
    if len(net.state_dict()[key].size()) == 1:
        print(torch.max(net.state_dict()[key],dim=0), "\n", torch.min(net.state_dict()[key],dim=0))
    else:
        print(torch.max(net.state_dict()[key],0),"\n",torch.min(net.state_dict()[key],0))
    print("")
