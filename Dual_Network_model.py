import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from my_enum import *
import torch.optim as optim

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Dual_State_value = namedtuple('Value', ('state', 'action', 'next_state', 'reward'))
input_size = 19218


class Dual_Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Dual_Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        layer = [Dual_ResNet(n_mid, n_mid) for _ in range(19)]
        self.layer = nn.ModuleList(layer)

        self.fc3_p1 = nn.Linear(n_mid, 2)
        self.bn_p1 = nn.BatchNorm1d(2)
        # 手札の枚数+自分の場のフォロワーの数(進化)＋ターン終了
        self.fc3_p2 = nn.Linear(2, 9 + 5*2 + 1)

        self.fc3_v1 = nn.Linear(n_mid, 5)
        self.bn_v1 = nn.BatchNorm1d(5)
        self.fc3_v2 = nn.Linear(5, 2)
        self.fc3_v3 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(19):
            x = self.layer[i](x)

        h_p1 = F.relu(self.bn_p1(self.fc3_p1(x)))
        out_p = self.fc3_p2(h_p1)

        h_v1 = F.relu(self.bn_v1(self.fc3_v1(x)))
        h_v2 = F.relu(self.fc3_v2(h_v1))
        out_v = torch.sigmoid(self.fc3_v3(h_v2))
        return out_p, out_v


class Dual_ResNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(Dual_ResNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)) + x)

        return h2


class Dual_Loss(nn.Module):

    def __init__(self):
        super(Dual_Loss, self).__init__()

    def forward(self, p, v, z, pai):
        #l = (z − v)^2 − π> log p + c||θ||2
        loss = torch.sum(
            torch.pow((z - v),2),
            dim=1)
        assert  all(not torch.isnan(cell) for cell in loss), "loss:{}".format(loss)

        tmp = torch.Tensor([0])

        for i in range(p.size()[0]):
            #print(p[0].size(), pai[0].size())
            soft_max_p = torch.softmax(p[i],dim=0)
            for j in range(p.size()[1]):

                tmp[0] += -pai[i][j] * torch.log(soft_max_p[j])


        loss = torch.mean(loss)
        #print(loss.size(), loss)
        loss += tmp[0]/20
        #L2正則化はoptimizer
        return loss

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

def state_set_change_to_full(origin):
    # [card_category,cost,card_id]*9*2 + [card_id,power,toughness,[ability]]*5*2+[life,life,turn]
    # 3*9 + 4*10 + 3 = 27 + 40 +3 = 70
    convert_states = []
    convert_actions = []
    convert_next_states = []
    for data in origin:
        cell = data.state
        next_cell = data.next_state
        assert len(cell) == 70,"cell_len:{}".format(len(cell))
        assert len(next_cell) == 70, "next_cell_len:{}".format(len(next_cell))
        tmp = []
        next_tmp = []
        for i in range(9):
            tmp.extend(list(np.identity(4)[cell[3*i]]))
            tmp.append(cell[3*i+1])
            tmp.extend(list(np.identity(1000)[cell[3*i+2] + 500]))
            next_tmp.extend(list(np.identity(4)[next_cell[3*i]]))
            next_tmp.append(next_cell[3*i+1])
            next_tmp.extend(list(np.identity(1000)[next_cell[3*i+2] + 500]))
        #9*(4+1+1000) = 9*1005 = 9045
        for i in range(10):
            j = Field_START + 4*i
            #assert type(cell[j]) == int and type(cell[j+1]) == int and type(cell[j+2]) == int\
            #    and type(cell[j+3]) == list,"cell={}".format(cell[j:j+4])
            tmp.extend(list(np.identity(1000)[cell[j] + 500]))
            tmp.extend([cell[j+1], cell[j+2]])
            embed_ability = [int(ability_id in cell[j+3]) for ability_id in range(1, 16)]
            tmp.extend(embed_ability)
            next_tmp.extend(list(np.identity(1000)[next_cell[j] + 500]))
            next_tmp.extend([next_cell[j+1], next_cell[j+2]])
            next_embed_ability = [int(ability_id in next_cell[j+3]) for ability_id in range(1, 16)]
            next_tmp.extend(next_embed_ability)
        #10 *(1000+2+15) = 10170
        #9045 + 10170 = 19215
        tmp.extend(cell[LIFE_START:])
        next_tmp.extend(next_cell[LIFE_START:])
        convert_states.append(torch.Tensor(tmp))
        convert_next_states.append(torch.Tensor(next_tmp))
        convert_actions.append(torch.Tensor(list(np.identity(20)[data.action])))


    return convert_states, convert_actions, convert_next_states



class Dual_ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリが満タンじゃないときには追加
        self.memory[self.index] = Dual_State_value(state,action,next_state, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        assert len(self.memory)>=batch_size,"{}<{}".format(len(self.memory),batch_size)
        tmp = random.sample(self.memory, batch_size)
        states,actions,next_states = state_set_change_to_full(tmp)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = [cell.reward for cell in tmp]
        rewards = torch.stack(rewards, dim=0)
        return states,actions,next_states,rewards
    def __len__(self):
        return len(self.memory)

net = Dual_Net(input_size,100,2)



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
    from Game_setting import Game
    parser = argparse.ArgumentParser(description='デュアルニューラルネットワーク学習コード')

    parser.add_argument('--episode_num', help='試行回数')
    parser.add_argument('--iteration_num', help='イテレーション数')
    parser.add_argument('--epoch_num', help='エポック数')
    parser.add_argument('--batch_size', help='バッチサイズ')
    parser.add_argument('--mcts', help='サンプリングAIをMCTSにする(オリジナルの場合は[OM])')
    args = parser.parse_args()
    G = Game()
    Over_all_R = Dual_ReplayMemory(100000)
    episode_len = 100
    if args.episode_num is not None:
        episode_len = int(args.episode_num)
    batch_size = 100
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    iteration = 10
    if args.iteration_num is not None:
        iteration = int(args.iteration_num)
    epoch_num = 2
    if args.epoch_num is not None:
        epoch_num = int(args.epoch_num)
    mcts = False
    if args.mcts is not None:
        mcts = True
    import datetime
    t1 = datetime.datetime.now()
    print(t1)
    print(net)
    R = Dual_ReplayMemory(100000)
    import copy
    net.zero_grad()
    prev_net = copy.deepcopy(net)
    for epoch in range(epoch_num):
        print("epoc {}".format(epoch+1))
        R = Dual_ReplayMemory(200*(episode_len//10))
        p1 = Player(9, True, policy=AggroPolicy())
        if args.mcts == "OM":
            p1 = Player(9, True, policy=Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net))
        elif mcts:
            p1 = Player(9, True, policy=Opponent_Modeling_MCTSPolicy())
        p1.name = "Alice"
        p2 = Player(9, False, policy=AggroPolicy())
        if args.mcts == "OM":
            p2 = Player(9, False, policy=Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net))
        elif mcts:
            p2 = Player(9, False, policy=Opponent_Modeling_MCTSPolicy())
        p2.name = "Bob"
        for episode in tqdm(range(episode_len)):
            f = Field(5)
            deck_type1 = 0 #random.choice(list(key_2_tsv_name.keys()))
            deck_type2 = 0 #random.choice(list(key_2_tsv_name.keys()))
            d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
            d1.set_leader_class(key_2_tsv_name[deck_type1][1])
            d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
            d2.set_leader_class(key_2_tsv_name[deck_type2][1])
            d1.shuffle()
            d2.shuffle()
            p1.deck = d1
            p2.deck = d2
            f.players = [p1, p2]
            p1.field = f
            p2.field = f
            train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)
            f.players[0].life = 20
            f.players[0].hand.clear()
            f.players[0].deck = None
            f.players[0].lib_out_flg = False
            f.players[1].life = 20
            f.players[1].hand.clear()
            f.players[1].deck = None
            f.players[1].lib_out_flg = False
            for data in train_data:
                R.push(data[0], data[1], data[2], torch.FloatTensor([reward]))
                Over_all_R.push(data[0], data[1], data[2], torch.FloatTensor([reward]))
            if (episode+1) % (episode_len//10) == 0:
                prev_net = copy.deepcopy(net)
                optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
                for i in range(iteration):
                    states, actions, next_states, rewards = R.sample(batch_size)
                    optimizer.zero_grad()
                    p,v = net(states)
                    z = rewards
                    pai = actions#20種類の抽象化した行動
                    criterion = Dual_Loss()
                    loss = criterion(p,v,z,pai)
                    loss.backward()
                    optimizer.step()
                p1 = Player(9, True, policy=AggroPolicy())
                if args.mcts == "OM":
                    p1 = Player(9, True, policy=Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net))
                elif mcts:
                    p1 = Player(9, True, policy=Opponent_Modeling_MCTSPolicy())
                p1.name = "Alice"
                p2 = Player(9, False, policy=AggroPolicy())
                if args.mcts == "OM":
                    p2 = Player(9, False, policy=Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net))
                elif mcts:
                    p2 = Player(9, False, policy=Opponent_Modeling_MCTSPolicy())
                p2.name = "Bob"
                R = Dual_ReplayMemory(200 * (episode_len//10))
    print('Finished Training')
    #PATH = './value_net.pth'
    import os

    #PATH = './value_net.pth'
    PATH = "model/Dual_{}_{}_{}_{}_{}_{}.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                         t1.second)
    torch.save(net.state_dict(), PATH)
    print("{} is saved.".format(PATH))
    t2 = datetime.datetime.now()
    print(t2)
    print(t2-t1)
    criterion = Dual_Loss()


    for i in range(10):
        states, actions, next_states, rewards = Over_all_R.sample(batch_size)
        p, v = net(states)
        z = rewards
        pai = actions  # 20種類の抽象化した行動
        outputs = net(states)
        loss = criterion(p,v,z,pai)
        print("{} MSELoss: {:.3f}".format(i + 1, float(loss.item())))