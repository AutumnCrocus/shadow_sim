from Network_model import *
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='ニューラルネットワークテストコード')
parser.add_argument('--model_name', help='モデル名')
parser.add_argument('--sample_num', help='全サンプル数')
parser.add_argument('--batch_size', help='バッチサイズ')
args = parser.parse_args()
print(args)
#net = Net(10173,10,1)
net = Net(19218,10,1)
#net = try_gpu(net)
model_name = 'value_net.pth'
if args.model_name is not None:
    model_name = args.model_name
PATH = 'model/'+ model_name
net.load_state_dict(torch.load(PATH))
#print(net.state_dict())
R = ReplayMemory(1000000)
from test import *  # importの依存関係により必ず最初にimport
from Field_setting import *
from Player_setting import *
from Policy import *
from Game_setting import Game
G = Game()
sample_num = 10000
batch_size = 100
if args.sample_num is not None:
    sample_num = int(args.sample_num)
if args.batch_size is not None:
    batch_size = int(args.batch_size)
print("Now sampling...")
for i in tqdm(range(sample_num)):
    f = Field(5)
    p1 = Player(9, True, policy=AggroPolicy())
    p1.name = "Alice"
    deck_type1 = random.choice(list(key_2_tsv_name.keys()))
    deck_type2 = random.choice(list(key_2_tsv_name.keys()))
    d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
    d1.set_leader_class(key_2_tsv_name[deck_type1][1])
    p2 = Player(9, False, policy=AggroPolicy())
    p2.name = "Bob"
    d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
    d2.set_leader_class(key_2_tsv_name[deck_type2][1])
    d1.shuffle()
    d2.shuffle()
    p1.deck = d1
    p2.deck = d2
    f.players = [p1, p2]
    p1.field = f
    p2.field = f
    train_data, reward = G.start_for_train_data(f, virtual_flg=True)
    for data in train_data:
        #R.push(torch.Tensor(data), torch.FloatTensor([reward]))
        R.push(data, torch.FloatTensor([reward]))

print("")
criterion = nn.SmoothL1Loss()
for i in range(10):
    inputs, targets = R.sample(batch_size)
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    print("output:{} target:{}".format(float(outputs[0]), float(targets[0])))
    print("{} MSELoss: {:.3f}".format(i + 1, float(loss.item())))



