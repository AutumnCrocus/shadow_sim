import torch
#import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
import torch.optim as optim
#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
State_value = namedtuple('Value', ('state','reward'))
input_size = 19218
class Net(nn.Module):
    def __init__(self,n_in, n_mid, n_out):
        super(Net,self).__init__()
        #self.fc1 = nn.Linear(n_in, n_mid)
        #self.fc2 = nn.Linear(n_mid, n_mid)
        #self.fc3_adv = nn.Linear(n_mid, n_out)
        #self.fc3_v = nn.Linear(n_mid,1)
        # model.parameters()で学習パラメータのイテレータを取得できるが，
        # listで保持しているとlist内のモジュールのパラメータは取得できない
        # optimについては後述
        self.fc1 = nn.Linear(n_in, n_mid)
        #layer = [nn.Linear(n_mid,n_mid) for _ in range(38)]
        layer = [ResNet(n_mid,n_mid) for _ in range(19)]
        self.layer = nn.ModuleList(layer)
        self.fc3_v = nn.Linear(n_mid,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(19):
            x = self.layer[i](x)
        """
        blocks = [h0]
        for i in range(19):
            print(type(blocks[2*i]))
            h1 = F.relu(self.layer[2*i](blocks[2*i]))
            blocks.append(h1)
            print(type(blocks[2 * i+1]))
            h2 = F.relu(self.layer[2*i+1](h1) + blocks[2*i])
            blocks.append(h2)
        """
        #self.blocks = nn.ModuleList(blocks)
        #h2 = F.relu(self.fc2(h1))
        #val = self.fc3_v(h2)
        #val = F.tanh(self.fc3_v(h2))
        #val = torch.tanh(self.fc3_v(blocks[-1]))
        #val = torch.sigmoid(self.fc3_v(blocks[-1]))
        x = torch.sigmoid(self.fc3_v(x))
        return x

class ResNet(nn.Module):
    def __init__(self,n_in,n_out):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + x)

        return h2

def get_data(f):
    input_field_data = []
    for hand_card in f.players[0].hand:
        input_field_data.extend(list(np.identity(4)[Card_Category[hand_card.card_category].value]))
        input_field_data.extend([hand_card.cost])
        input_field_data.extend(list(np.identity(1000)[hand_card.card_id+500]))

    for j in range(len(f.players[0].hand),9):
        input_field_data.extend(list(np.identity(4)[Card_Category.NONE.value]))
        input_field_data.extend([0])
        input_field_data.extend([0]*1000)
    for i in range(2):
        for card in f.card_location[i]:
            #1000+2+15=1017次元
            if card.card_category == "Creature":
                input_field_data.extend(list(np.identity(1000)[card.card_id+500]))
                input_field_data.extend([card.power, card.get_current_toughness(),])
                embed_ability = [int(ability_id in card.ability) for ability_id in range(1, 16)]
                input_field_data.extend(embed_ability)
                #input_field_data.extend([card.card_id, card.power, card.get_current_toughness(),
                #                         int(KeywordAbility.WARD.value in card.ability)])
            else:
                input_field_data.extend([0]*1000)
                input_field_data.extend([0, 0])
                input_field_data.extend([0] * 15)

        for k in range(len(f.card_location[i]),5):
            #input_field_data.extend([0, 0, 0, 0])
            input_field_data.extend([0] * 1000)
            input_field_data.extend([0, 0])
            input_field_data.extend([0] * 15)
    input_field_data.extend([f.players[0].life, f.players[1].life,f.current_turn[0]])

    return input_field_data
Field_START = 27
LIFE_START = 67

def state_change_to_full(origin):
    # [card_category,cost,card_id]*9*2 + [card_id,power,toughness,[ability]]*5*2+[life,life,turn]
    # 3*9 + 4*10 + 3 = 27 + 40 +3 = 70
    convert_states = []

    for data in origin:
        if len(origin) == 70:
            cell = origin
        else:
            cell = data.state
        assert len(cell) == 70,"cell_len:{}".format(len(cell))
        tmp = []
        for i in range(9):
            tmp.extend(list(np.identity(4)[cell[3*i]]))
            tmp.append(cell[3*i+1])
            tmp.extend(list(np.identity(1000)[cell[3*i+2] + 500]))
        #9*(4+1+1000) = 9*1005 = 9045
        for i in range(10):
            j = Field_START + 4*i
            assert j % 4 == 3,"j={}".format(j)
            assert type(cell[j]) == int and type(cell[j+1]) == int and type(cell[j+2]) == int\
                and type(cell[j+3]) == list,"cell={}".format(cell[j:j+4])
            tmp.extend(list(np.identity(1000)[cell[j] + 500]))
            tmp.extend([cell[j+1], cell[j+2]])
            embed_ability = [int(ability_id in cell[j+3]) for ability_id in range(1, 16)]
            tmp.extend(embed_ability)
        #10 *(1000+2+15) = 10170
        #9045 + 10170 = 19215
        assert len(cell[LIFE_START:]) == 3,"data:{}".format(cell[LIFE_START:])
        tmp.extend(cell[LIFE_START:])
        convert_states.append(torch.Tensor(tmp))
        if len(origin) == 70:
            break

    return convert_states



class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = []
        self.index = 0

    def push(self, state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリが満タンじゃないときには追加
        #[card_category,cost,card_id]*9*2 + [card_id,power,toughness,[ability]]*5*2+[life,life,turn]
        #3*18 + 4*10 + 3 = 54 + 40 +3 = 97
        self.memory[self.index] = State_value(state,reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        tmp = random.sample(self.memory, batch_size)
        #[card_category,cost,card_id]*9*2 + [card_id,power,toughness,[ability]]*5*2+[life,life,turn]
        #3*18 + 4*10 + 3 = 54 + 40 +3 = 97
        inputs = state_change_to_full(tmp)
        inputs = torch.stack(inputs, dim=0)
        outputs = [cell.reward for cell in tmp]
        outputs = torch.stack(outputs, dim=0)
        #inputs = [cell.state for cell in tmp]
        #inputs =torch.stack(inputs,dim=0)
        #outputs = [cell.reward for cell in tmp]
        #outputs = torch.stack(outputs, dim=0)
        return inputs, outputs
        #return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e
#net = Net(10173,10,1)
net = Net(input_size,10,1)
#net = try_gpu(net)

deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                  6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic", -2: "Sword_Basic",
                  -3: "Rune_Basic",
                  -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
                  100: "Test",
                  -9: "Spell-Rune", 11: "PtP-Forest", 12: "Mid-Shadow", 13: "Neutral-Blood"}

key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                  3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                  6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                  9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                  11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"],
                  13: ["Neutral_Blood.tsv", "BLOOD"]}
if __name__ == "__main__":
    from emulator_test import *  # importの依存関係により必ず最初にimport
    from Field_setting import *
    from Player_setting import *
    from Policy import *
    parser = argparse.ArgumentParser(description='ニューラルネットワーク学習コード')

    parser.add_argument('--episode_num', help='試行回数')
    parser.add_argument('--epoch_num', help='エポック数')
    parser.add_argument('--iteration_num', help='イテレーション数')
    parser.add_argument('--memory_size', help='リプレイメモリーのサイズ')
    parser.add_argument('--batch_size', help='バッチサイズ')
    args = parser.parse_args()
    print("args:{}".format(args))
    CAPACITY = 10000
    if args.memory_size is not None:
        CAPACITY = int(args.memory_size)
    import datetime
    t1 = datetime.datetime.now()
    print(t1)
    print(net)
    from Game_setting import Game
    max_episode = 1000
    if args.episode_num is not None:
        max_episode = int(args.episode_num)
    G = Game()
    R = ReplayMemory(CAPACITY)
    optimizer = optim.Adam(net.parameters())
    dtype = torch.int
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Now sampling...")
    for episode in tqdm(range(max_episode)):
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
        train_data, reward = G.start_for_train_data(f, virtual_flg=True,target_player_num=episode%2)
        for data in train_data:
            R.push(data,torch.FloatTensor([reward]))
    net.zero_grad()
    print("sample_num:{}/{}".format(len(R.memory),CAPACITY))
    epoch_num = 2
    iteration_num = 1000
    batch_size = 100
    if args.epoch_num is not None:
        epoch_num = int(args.epoch_num)
    if args.iteration_num is not None:
        iteration_num = int(args.iteration_num)
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    for epoch in range(epoch_num):
        print("epoch {}".format(epoch+1))
        running_loss = 0.0
        loss_history = []
        for i in tqdm(range(iteration_num)):
            inputs,targets = R.sample(batch_size)
            optimizer.zero_grad()
            #print(inputs[-1],targets[-1])
            outputs = net(inputs)
            #criterion = nn.MSELoss()
            criterion = nn.SmoothL1Loss()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % (iteration_num//10) == (iteration_num//10)-1:  # print every 1000 mini-batches
                loss_history.append((epoch + 1, i + 1, running_loss / (iteration_num//10)))
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / (iteration_num//10)))
                running_loss = 0.0
        print("losses")
        for loss_id in loss_history:
            print("[{} {}]:{:.3f}".format(loss_id[0],loss_id[1],loss_id[2]))
        print("")

    print('Finished Training')
    #PATH = './value_net.pth'
    import os

    #PATH = './value_net.pth'
    PATH = "model/{}_{}_{}_{}_{}_{}.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                         t1.second)
    torch.save(net.state_dict(), PATH)
    print("{} is saved.".format(PATH))
    t2 = datetime.datetime.now()

    print(t2-t1)
    correct = 0
    total = 0
    criterion = nn.SmoothL1Loss()


    for i in range(10):
        inputs,targets = R.sample(100)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        #accuracy = [int(outputs[j]*targets[j] > 0) for j in range(len(outputs))]
        #accuracy = sum(accuracy) / len(outputs)
        #print(inputs[0][:20])
        #print(inputs[0][20:40])
        #print(inputs[0][40:])
        print(inputs[0][-3:])
        print("output:{} target:{}".format(float(outputs[0]),float(targets[0])))
        print("{} MSELoss: {:.3f}".format(i + 1, float(loss.item())))
        #print("{} accuracy: {:.3%}".format(i+1,float(accuracy)))

