import os
os.environ["OMP_NUM_THREADS"] = "4"
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import copy
import random
from my_enum import *
import torch.optim as optim
from pytorch_memlab import profile
import argparse
# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Dual_State_value = namedtuple('Value', ('state', 'action', 'next_state', 'detailed_action_code','reward'))
Detailed_State_data = namedtuple('Value', ('hand_ids', 'hand_card_costs', 'follower_card_ids',
                                         'amulet_card_ids', 'follower_stats', 'follower_abilities', 'able_to_evo',
                                         'life_data', 'pp_data','able_to_play','able_to_attack', 'able_to_creature_attack'))
"""
   input = {'values', 'hand_ids','follower_card_ids', 
        'amulet_card_ids', 'follower_abilities', 'able_to_evo'}
"""
class New_Dual_Net(nn.Module):
    def __init__(self,n_mid):
        super(New_Dual_Net, self).__init__()
        #self.li1 = nn.Linear(32,n_mid)#手札(9)+盤面10枚(20)+両方の体力(2)＋ターン(1) = 32
        self.lin1 = nn.Linear(5,n_mid)#お互いの体力、手札枚数、経過ターン
        self.lin2 = nn.Linear(2, n_mid)#最大PP,残りPP
        #self.lin3 = nn.Linear(9, n_mid)#最大手札枚数
        self.lin3 = nn.Linear(2,n_mid)#フォロワーの攻撃力,体力
        self.lin4 = nn.Linear(3 * n_mid + 1, n_mid)#手札連結
        self.lin5 = nn.Linear(6 * n_mid, n_mid)#自分の場のフォロワー連結
        self.lin6 = nn.Linear(4 * n_mid, n_mid)  # 相手の場のフォロワー連結
        self.emb1 = nn.Embedding(3000,n_mid,padding_idx=0)#1000枚*3カテゴリー（空白含む）
        self.emb2 = nn.Embedding(1000, n_mid, padding_idx=0)  # フォロワー1000枚（空白含む）
        self.emb3 = nn.Embedding(1000, n_mid, padding_idx=0)  # アミュレット1000枚（空白含む）
        self.emb4 = nn.Embedding(16, n_mid, padding_idx=0)  # キーワード能力15個と空白
        self.emb5 = nn.Embedding(6, n_mid, padding_idx=0)  # 進化可能最大五体と空白
        self.emb6 = nn.Embedding(10, n_mid, padding_idx=0)  # 手札最大9枚の位置と空白
        self.emb7 = nn.Embedding(6, n_mid, padding_idx=0)  # プレイヤーに攻撃可能な最大五体と空白
        self.emb8 = nn.Embedding(6, n_mid, padding_idx=0)  # プレイヤーに攻撃可能な最大五体と空白
        self.fc1 = nn.Linear(n_mid, n_mid)
        layer = [Dual_ResNet(n_mid, n_mid) for _ in range(19)]
        self.layer = nn.ModuleList(layer)

        #self.fc3_p1 = nn.Linear(n_mid, 25)
        #self.bn_p1 = nn.BatchNorm1d(25)
        # 手札の枚数+自分の場のフォロワーの数*3＋ターン終了
        #self.fc3_p2 = nn.Linear(25, 25)
        self.action_value_net = Action_Value_Net(mid_size=n_mid)

        self.fc3_v1 = nn.Linear(n_mid, 5)
        self.bn_v1 = nn.BatchNorm1d(5)
        self.fc3_v2 = nn.Linear(5, 2)
        self.fc3_v3 = nn.Linear(2, 1)

        self.loss_fn = Dual_Loss()
        self.filtered_softmax = filtered_softmax()

    #@profile
    def forward(self, states,target=False):
        values = states['values']
        hand_ids = states['hand_ids']
        follower_card_ids = states['follower_card_ids']
        amulet_card_ids = states['amulet_card_ids']
        follower_abilities = states['follower_abilities']
        able_to_evo = states['able_to_evo']
        detailed_action_codes = states['detailed_action_codes']
        x1 = F.relu(self.lin1(values['life_datas']))

        pp_datas = F.relu(self.lin2(values['pp_datas']))
        able_to_plays = F.relu(torch.sum(F.relu(self.emb6(values['able_to_play'])),dim=1))
        able_to_attacks = F.relu(torch.sum(F.relu(self.emb7(values['able_to_attack'])), dim=1))
        able_to_creature_attacks = F.relu(torch.sum(F.relu(self.emb8(values['able_to_creature_attack'])), dim=1))
        x2 = [] #最大手札9枚分のデータ
        for i in range(9):
            hand_card_ids = F.relu(torch.sum(self.emb1(hand_ids[i]),dim=1))
            hand_card_costs = values['hand_card_costs'][i]
            tmp_x2 = torch.cat([pp_datas, hand_card_costs, hand_card_ids, able_to_plays], dim=1)
            x2.append(F.relu(self.lin4(tmp_x2)))
        x2 = torch.sum(torch.stack(x2,dim=2),dim=2)

        x3 = []#最大フォロワー10体分のデータ
        x4 = [] # 最大アミュレット10個分のデータ
        able_to_evos = F.relu(torch.sum(F.relu(self.emb5(able_to_evo)),dim=1))
        for i in range(10):#自分のフォロワー
            stats = F.relu(self.lin3(values['follower_stats'][i]))
            abilities = F.relu(torch.sum(self.emb4(follower_abilities[i]),dim=1))
            follower_ids = F.relu(torch.sum(self.emb2(follower_card_ids[i]), dim=1))

            if i <= 4:
                tmp_x3 = torch.cat(
                    [stats, abilities, follower_ids, able_to_evos, able_to_attacks, able_to_creature_attacks], dim=1)
                x3.append(F.relu(self.lin5(tmp_x3)))
                amulet_ids = F.relu(torch.sum(self.emb3(amulet_card_ids[i]),dim=1))
                x4.append(amulet_ids)
            else:
                tmp_x3 = torch.cat([stats, abilities, follower_ids, able_to_evos],dim=1)
                x3.append(F.relu(self.lin6(tmp_x3)))
                amulet_ids = F.relu(torch.sum(self.emb3(amulet_card_ids[i]),dim=1))
                x4.append(amulet_ids)


        x3 = torch.sum(torch.stack(x3, dim=2), dim=2)
        x4 = torch.sum(torch.stack(x4, dim=2), dim=2)
        x = x1 + x2 + x3 + x4
        x = F.relu(self.fc1(x))
        for i in range(19):
            x = self.layer[i](x)

        #h_p1 = F.relu(self.bn_p1(self.fc3_p1(x)))
        action_value_list = []
        action_categories = detailed_action_codes['action_categories']
        #card_locations = detailed_action_codes['card_locations']
        #card_ids = detailed_action_codes['card_ids']
        play_card_ids = detailed_action_codes['play_card_ids']
        attacking_card_ids = detailed_action_codes['attacking_card_ids']
        attacked_card_ids = detailed_action_codes['attacked_card_ids']
        evolving_card_ids = detailed_action_codes['evolving_card_ids']
        able_to_choice = detailed_action_codes['able_to_choice']
        #print(detailed_action_codes)
        for i in range(45):
            tmp = self.action_value_net(x, action_categories[i], play_card_ids[i], attacking_card_ids[i],
                                        attacked_card_ids[i], evolving_card_ids[i])
            action_value_list.append(tmp)
        h_p2 = torch.cat(action_value_list,dim=1)

        out_p = self.filtered_softmax(h_p2, able_to_choice)
        #assert len(out_p[0]) == 45, "{}".format(len(out_p[0]))
        #選べない行動を0にするベクトルをかけるsoftmax

        h_v1 = F.relu(self.bn_v1(self.fc3_v1(x)))
        h_v2 = F.relu(self.fc3_v2(h_v1))
        out_v = torch.tanh(self.fc3_v3(h_v2))
        if target:
            z = states['target']['rewards']
            pai = states['target']['actions']
            #print(z[0],pai[0])

            return out_p, out_v, self.loss_fn(out_p, out_v, z, pai)
        else:
            return out_p, out_v


class Dual_ResNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(Dual_ResNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)

    def forward(self, x):
        #assert all(not torch.isnan(cell) for cell in x[0]), "{}".format(x[0])
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)) + x)

        return h2


class Action_Value_Net(nn.Module):
    def __init__(self,mid_size = 100):
        super(Action_Value_Net, self).__init__()
        self.mid_size = mid_size
        self.emb1 = nn.Embedding(5, mid_size)  # 行動のカテゴリー
        self.emb2 = nn.Embedding(3000, mid_size, padding_idx=0)  # 1000枚*3カテゴリー（空白含む）
        self.emb3 = nn.Embedding(1000, mid_size, padding_idx=0)  # フォロワー1000枚
        self.lin1 = nn.Linear(6 * mid_size, mid_size)
        self.lin2 = nn.Linear(mid_size, 1)

    def forward(self, states, action_categories, play_card_ids,
                attacking_card_ids, attacked_card_ids, evolving_card_ids):

        embed_action_categories = torch.sum(self.emb1(action_categories), dim=1)

        embed_play_card_ids = torch.sum(self.emb2(play_card_ids), dim=1)
        embed_attacking_card_ids = torch.sum(self.emb3(attacking_card_ids), dim=1)
        embed_attacked_card_ids = torch.sum(self.emb3(attacked_card_ids), dim=1)
        embed_evolving_card_ids = torch.sum(self.emb3(evolving_card_ids), dim=1)
        tmp = torch.cat([states, embed_action_categories, embed_play_card_ids,
                         embed_attacking_card_ids, embed_attacked_card_ids,
                         embed_evolving_card_ids], dim=1)
        output = self.lin1(tmp)
        output = self.lin2(output)

        return output


class filtered_softmax(nn.Module):
    def __init__(self):
        super(filtered_softmax, self).__init__()

    def forward(self, x, label):
        x = torch.exp(x)
        x = x * label

        sum_of_x = torch.sum(x, dim=1,keepdim=True)
        #print(x.size(), sum_of_x.size())
        x = x / sum_of_x

        return x





class Dual_Loss(nn.Module):

    def __init__(self):
        super(Dual_Loss, self).__init__()

    def forward(self, p, v, z, pai):
        #l = (z − v)^2 − πlog p + c||θ||2
        #paiはスカラー値
        loss = torch.sum(
            torch.pow((z - v),2),
            dim=1)
        #assert  all(not torch.isnan(cell) for cell in loss), "loss:{}\n{}\n{}".format(loss,z,v)

        #tmp = torch.Tensor([0])
        tmp = []
        for i in range(p.size()[0]):
            tmp.append(-torch.log(p[i][pai[i]]+1e-8))
        tmp = torch.sum(torch.stack(tmp, dim=0), dim=1)
        #print("MSE:{}".format(torch.mean(loss)))
        #print("cross_entropy:{}".format(torch.mean(tmp)))
        MSE = torch.mean(loss)
        CEE = torch.mean(tmp)
        loss += tmp
        loss = torch.mean(loss)
        #L2正則化はoptimizer
        return loss, MSE, CEE

def get_data(f,player_num=0):
    hand_ids = []
    hand_card_costs = []
    player = f.players[player_num]
    opponent = f.players[1-player_num]
    for hand_card in player.hand:
        hand_ids.append(Card_Category[hand_card.card_category].value*(hand_card.card_id+500))
        hand_card_costs.append(hand_card.cost)

    for j in range(len(player.hand),9):
        hand_ids.append(0)
        hand_card_costs.append(0)
    follower_card_ids = []
    amulet_card_ids = []
    follower_stats = []
    follower_abilities = []
    able_to_evo = f.get_able_to_evo(player)
    able_to_evo = [cell+1 for cell in able_to_evo]
    for j in range(2):
        i = (player_num+j)%2
        for card in f.card_location[i]:
            if card.card_category == "Creature":
                follower_card_ids.append(card.card_id+500)
                follower_stats.append([card.power,card.get_current_toughness()])
                follower_abilities.append(card.ability[:])
                amulet_card_ids.append(0)
            else:
                follower_card_ids.append(0)
                follower_stats.append([0,0])
                follower_abilities.append([])
                amulet_card_ids.append(card.card_id+500)

        for k in range(len(f.card_location[i]),5):
            follower_card_ids.append(0)
            follower_stats.append([0, 0])
            follower_abilities.append([])
            amulet_card_ids.append(0)
    life_data = [player.life, opponent.life, len(player.hand),len(opponent.hand) ,f.current_turn[player_num]]
    pp_data = [f.cost[player_num],f.remain_cost[player_num]]
    able_to_play = f.get_able_to_play(player)
    able_to_play = [cell+1 for cell in able_to_play]
    able_to_attack = f.get_able_to_attack(player)
    able_to_creature_attack = f.get_able_to_creature_attack(player)

    datas = Detailed_State_data(hand_ids, hand_card_costs, follower_card_ids, amulet_card_ids,
                              follower_stats, follower_abilities, able_to_evo, life_data, pp_data,
                              able_to_play, able_to_attack, able_to_creature_attack)

    return datas

class New_Dual_ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, detailed_action_code, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリが満タンじゃないときには追加
        self.memory[self.index] = Dual_State_value(state, action, next_state, detailed_action_code, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        tmp = random.sample(self.memory, batch_size)
        states = [cell.state for cell in tmp]
        states = Detailed_State_data_2_Tensor(states)
        actions = [cell.action for cell in tmp]
        actions = torch.stack(actions, dim=0)
        rewards = [cell.reward for cell in tmp]
        rewards = torch.stack(rewards, dim=0)
        detailed_action_codes = [cell.detailed_action_code for cell in tmp]
        detailed_action_codes = Detailed_action_code_2_Tensor(detailed_action_codes)
        states['detailed_action_codes'] = detailed_action_codes

        return states, actions, rewards

    def __len__(self):
        return len(self.memory)


def Detailed_State_data_2_Tensor(datas,cuda=False):
    """
    ('hand_ids', 'hand_card_costs', 'follower_card_ids',
    'amulet_card_ids', 'follower_stats', 'follower_abilities',
    'follower_is_evolved', 'life_data', 'pp_data'))
    """
    hand_ids = [[], [], [], [], [], [], [], [], []]
    hand_card_costs = [[], [], [], [], [], [], [], [], []]

    follower_card_ids = [[],[],[],[],[],[],[],[],[],[]]
    follower_stats = [[],[],[],[],[],[],[],[],[],[]]
    follower_abilities = [[], [], [], [], [], [], [], [], [], []]
    amulet_card_ids = [[],[],[],[],[],[],[],[],[],[]]

    one_hot_able_actions = []
    able_to_evo = []
    able_to_play = []
    able_to_attack = []
    able_to_creature_attack = []
    pp_datas = []
    #values = []
    life_datas = []
    if torch.cuda.is_available() and cuda:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for data in datas:
            for i in range(9):
                hand_ids[i].append(torch.LongTensor([data.hand_ids[i]]).cuda())
                hand_card_costs[i].append(torch.Tensor([data.hand_card_costs[i]]).cuda())

            for i in range(10):
                follower_card_ids[i].append(torch.LongTensor([data.follower_card_ids[i]]).cuda())
                amulet_card_ids[i].append(torch.LongTensor([data.amulet_card_ids[i]]).cuda())

            for i in range(10):
                inputs = data.follower_abilities[i][:]
                inputs.extend([0 for _ in range(1, 16 - len(data.follower_abilities[i]))])
                follower_abilities[i].append(torch.LongTensor(inputs).cuda())
                follower_stats[i].append(data.follower_stats[i])

            tmp2 = data.able_to_evo[:]

            tmp2.extend([0 for _ in range(5 - len(data.able_to_evo))])
            able_to_evo.append(torch.LongTensor(tmp2).cuda())
            tmp2 = data.able_to_play[:]
            tmp2.extend([0 for _ in range(9 - len(data.able_to_play))])
            able_to_play.append(torch.LongTensor(tmp2).cuda())
            tmp2 = data.able_to_attack[:]
            tmp2.extend([0 for _ in range(5 - len(data.able_to_attack))])
            able_to_attack.append(torch.LongTensor(tmp2).cuda())
            tmp2 = data.able_to_creature_attack[:]
            tmp2.extend([0 for _ in range(5 - len(data.able_to_creature_attack))])
            able_to_creature_attack.append(torch.LongTensor(tmp2).cuda())

            tmp = [1.0]
            tmp.extend([int(cell in data.able_to_play) for cell in range(1, 10)])
            tmp.extend([int(cell in data.able_to_attack) for cell in range(1, 6)])
            tmp.extend([int(cell in data.able_to_creature_attack) for cell in range(1, 6)])
            tmp.extend([int(cell in data.able_to_evo) for cell in range(1, 6)])
            one_hot_able_actions.append(torch.Tensor(tmp).cuda())
            tmp_values = []
            pp_datas.append(torch.Tensor(data.pp_data).cuda())
            life_datas.append(torch.Tensor(data.life_data).cuda())
    else:
        for data in datas:
            for i in range(9):
                hand_ids[i].append(torch.LongTensor([data.hand_ids[i]]))
                hand_card_costs[i].append(torch.Tensor([data.hand_card_costs[i]]))

            for i in range(10):
                follower_card_ids[i].append(torch.LongTensor([data.follower_card_ids[i]]))
                amulet_card_ids[i].append(torch.LongTensor([data.amulet_card_ids[i]]))

            for i in range(10):
                inputs = data.follower_abilities[i][:]
                inputs.extend([0 for _ in range(1,16-len(data.follower_abilities[i]))])
                follower_abilities[i].append(torch.LongTensor(inputs))
                follower_stats[i].append(data.follower_stats[i])

            tmp2 = data.able_to_evo[:]

            tmp2.extend([0 for _ in range(5-len(data.able_to_evo))])
            able_to_evo.append(torch.LongTensor(tmp2))
            tmp2 = data.able_to_play[:]
            tmp2.extend([0 for _ in range(9-len(data.able_to_play))])
            able_to_play.append(torch.LongTensor(tmp2))
            tmp2 = data.able_to_attack[:]
            tmp2.extend([0 for _ in range(5-len(data.able_to_attack))])
            able_to_attack.append(torch.LongTensor(tmp2))
            tmp2 = data.able_to_creature_attack[:]
            tmp2.extend([0 for _ in range(5-len(data.able_to_creature_attack))])
            able_to_creature_attack.append(torch.LongTensor(tmp2))

            tmp = [1.0]
            tmp.extend([int(cell in data.able_to_play) for cell in range(1, 10)])
            tmp.extend([int(cell in data.able_to_attack) for cell in range(1, 6)])
            tmp.extend([int(cell in data.able_to_creature_attack) for cell in range(1, 6)])
            tmp.extend([int(cell in data.able_to_evo) for cell in range(1, 6)])
            one_hot_able_actions.append(torch.Tensor(tmp))
            tmp_values = []
            pp_datas.append(torch.Tensor(data.pp_data))
            life_datas.append(torch.Tensor(data.life_data))
    ans = None
    if torch.cuda.is_available() and cuda:
        ans = {'values': {'life_datas': torch.stack(life_datas, dim=0).cuda(),
                          'hand_card_costs': [torch.stack(hand_card_costs[i], dim=0) for i in range(9)],
                          'follower_stats': [torch.Tensor(follower_stats[i]).cuda() for i in range(10)],
                          'pp_datas': torch.stack(pp_datas, dim=0).cuda(),
                          'able_to_play': torch.stack(able_to_play, dim=0),
                          'able_to_attack': torch.stack(able_to_attack, dim=0),
                          'able_to_creature_attack': torch.stack(able_to_creature_attack, dim=0),
                          'one_hot_able_actions': torch.stack(one_hot_able_actions, dim=0)
                          },
               'hand_ids': [torch.stack(hand_ids[i], dim=0) for i in range(9)],
               'follower_card_ids': [torch.stack(follower_card_ids[i], dim=0) for i in range(10)],
               'amulet_card_ids': [torch.stack(amulet_card_ids[i], dim=0) for i in range(10)],
               'follower_abilities': [torch.stack(follower_abilities[i], dim=0)for i in range(10)],
               'able_to_evo': torch.stack(able_to_evo, dim=0)}

    else:
        ans = {'values':{'life_datas':torch.stack(life_datas, dim=0),
                         'hand_card_costs': [torch.stack(hand_card_costs[i], dim=0) for i in range(9)],
                         'follower_stats': [torch.Tensor(follower_stats[i]) for i in range(10)],
                         'pp_datas':torch.stack(pp_datas,dim=0),
                         'able_to_play':torch.stack(able_to_play,dim=0),
                         'able_to_attack':torch.stack(able_to_attack, dim=0),
                         'able_to_creature_attack':torch.stack(able_to_creature_attack, dim=0),
                         'one_hot_able_actions':torch.stack(one_hot_able_actions,dim=0)
                         },
               'hand_ids': [torch.stack(hand_ids[i], dim=0) for i in range(9)],
               'follower_card_ids': [torch.stack(follower_card_ids[i], dim=0) for i in range(10)],
               'amulet_card_ids': [torch.stack(amulet_card_ids[i], dim=0) for i in range(10)],
               'follower_abilities':[torch.stack(follower_abilities[i], dim=0) for i in range(10)],
               'able_to_evo':torch.stack(able_to_evo, dim=0)}
    return ans


def Detailed_action_code_2_Tensor(action_codes, cuda = False):
    tensor_action_categories = [[] for _ in range(45)]
    tensor_play_card_ids_in_action = [[] for _ in range(45)]
    tensor_attacking_card_ids_in_action = [[] for _ in range(45)]
    tensor_attacked_card_ids_in_action = [[] for _ in range(45)]
    tensor_evolving_card_ids_in_action = [[] for _ in range(45)]
    able_to_choice = []
    if torch.cuda.is_available() and cuda:
        for action_code in action_codes:
            for i in range(45):
                target_action = action_code['action_codes'][i]
                tensor_action_categories[i].append(torch.LongTensor([target_action[0]]).cuda())
                tensor_play_card_ids_in_action[i].append(torch.LongTensor([target_action[1]]).cuda())
                tensor_attacking_card_ids_in_action[i].append(torch.LongTensor([target_action[2]]).cuda())
                tensor_attacked_card_ids_in_action[i].append(torch.LongTensor([target_action[3]]).cuda())
                tensor_evolving_card_ids_in_action[i].append(torch.LongTensor([target_action[4]]).cuda())
            able_to_choice.append(torch.Tensor(action_code['able_to_choice']).cuda())

    else:
        for action_code in action_codes:
            for i in range(45):
                target_action = action_code['action_codes'][i]
                tensor_action_categories[i].append(torch.LongTensor([target_action[0]]))
                tensor_play_card_ids_in_action[i].append(torch.LongTensor([target_action[1]]))
                tensor_attacking_card_ids_in_action[i].append(torch.LongTensor([target_action[2]]))
                tensor_attacked_card_ids_in_action[i].append(torch.LongTensor([target_action[3]]))
                tensor_evolving_card_ids_in_action[i].append(torch.LongTensor([target_action[4]]))
            able_to_choice.append(torch.Tensor(action_code['able_to_choice']))


    action_codes_dict = {'action_categories': [torch.stack(tensor_action_categories[i], dim=0) for i in range(45)],
                         'play_card_ids': [torch.stack(tensor_play_card_ids_in_action[i], dim=0) for i in range(45)],
                         'attacking_card_ids': [torch.stack(tensor_attacking_card_ids_in_action[i], dim=0) for i in range(45)],
                         'attacked_card_ids': [torch.stack(tensor_attacked_card_ids_in_action[i], dim=0) for i in
                                                range(45)],
                         'evolving_card_ids': [torch.stack(tensor_evolving_card_ids_in_action[i], dim=0) for i in
                                                range(45)],
                         'able_to_choice': torch.stack(able_to_choice,dim=0)}
    return action_codes_dict




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
    parser = argparse.ArgumentParser(description='デュアルニューラルネットワーク学習コード')

    parser.add_argument('--episode_num', help='試行回数')
    parser.add_argument('--iteration_num', help='イテレーション数')
    parser.add_argument('--epoch_num', help='エポック数')
    parser.add_argument('--batch_size', help='バッチサイズ')
    parser.add_argument('--mcts', help='サンプリングAIをMCTSにする(オリジナルの場合は[OM])')
    parser.add_argument('--deck', help='サンプリングに用いるデッキの選び方')
    parser.add_argument('--cuda', help='gpuを使用するかどうか')
    args = parser.parse_args()
    net = New_Dual_Net(100)
    if torch.cuda.is_available() and args.cuda == "True":
        net = net.cuda()
        print("cuda is available.")
    cuda_flg = args.cuda == "True"
    from test import *  # importの依存関係により必ず最初にimport
    from Field_setting import *
    from Player_setting import *
    from Policy import *
    from Game_setting import Game

    deck_sampling_type = False
    if args.deck is not None:
        deck_sampling_type = True

    G = Game()
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
    #print(net)
    R = New_Dual_ReplayMemory(100000)
    net.zero_grad()
    prev_net = copy.deepcopy(net)
    import os
    for epoch in range(epoch_num):
        print("epoch {}".format(epoch+1))
        R = New_Dual_ReplayMemory(100000)
        p1 = Player(9, True, policy=AggroPolicy())
        if args.mcts == "OM":
            p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net,cuda=cuda_flg))
        elif mcts:
            p1 = Player(9, True, policy=Opponent_Modeling_MCTSPolicy())
        p1.name = "Alice"
        p2 = Player(9, False, policy=AggroPolicy())
        if args.mcts == "OM":
            p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net,cuda=cuda_flg))
        elif mcts:
            p2 = Player(9, False, policy=Opponent_Modeling_MCTSPolicy())
        p2.name = "Bob"
        win_num = 0
        for episode in tqdm(range(episode_len)):
            f = Field(5)
            deck_type1 = 0
            deck_type2 = 0
            if deck_sampling_type:
                deck_type1 = random.choice(list(key_2_tsv_name.keys()))
                deck_type2 = random.choice(list(key_2_tsv_name.keys()))
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
            #import cProfile
            #cProfile.run("G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)",sort="tottime")
            #assert False
            train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)
            f.players[0].life = 20
            f.players[0].hand.clear()
            f.players[0].deck = None
            f.players[0].lib_out_flg = False
            f.players[1].life = 20
            f.players[1].hand.clear()
            f.players[1].deck = None
            f.players[1].lib_out_flg = False
            if torch.cuda.is_available() and cuda_flg:
                for i in range(2):
                    for data in train_data[i]:
                        R.push(data[0], torch.LongTensor([data[1]]).cuda(), data[2], data[3],\
                               torch.FloatTensor([reward[i]]).cuda())
            else:
                for i in range(2):
                    for data in train_data[i]:
                        R.push(data[0], torch.LongTensor([data[1]]), data[2], data[3], torch.FloatTensor([reward[i]]))
            win_num += int(reward[episode % 2] > 0)

        print("sample_size:{}".format(len(R.memory)))
        print("win_rate:{:.2%}".format(win_num/episode_len))
        prev_net = copy.deepcopy(net)
        optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
        sum_of_loss = 0
        sum_of_MSE = 0
        sum_of_CEE = 0
        p,pai,z,states,loss = None, None, None, None,None
        for i in tqdm(range(iteration)):
            states, actions, rewards = R.sample(batch_size)
            optimizer.zero_grad()
            states['target'] = {'actions':actions, 'rewards':rewards}
            p, v, loss = net(states,target=True)
            z = rewards
            pai = actions#45種類の抽象化した行動
            loss[0].backward()
            sum_of_loss += float(loss[0].item())
            sum_of_MSE += float(loss[1].item())
            sum_of_CEE += float(loss[2].item())
            optimizer.step()

        print("{}".format(epoch + 1))
        print("AVE | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}"\
              .format(sum_of_loss/iteration,sum_of_MSE/iteration,sum_of_CEE/iteration))
        if torch.isnan(loss[0]):
            for key in list(net.state_dict().keys()):
                print(key, net.state_dict()[key].size())
                if len(net.state_dict()[key].size()) == 1:
                    print(torch.max(net.state_dict()[key], dim=0), "\n", torch.min(net.state_dict()[key], dim=0))
                else:
                    print(torch.max(net.state_dict()[key], 0), "\n", torch.min(net.state_dict()[key], 0))
                print("")
            assert False
        if epoch_num > 4 and (epoch+1) % (epoch_num//4) == 0 and epoch+1 < epoch_num:
            PATH = "model/Dual_{}_{}_{}_{}_{}_{}_{:.0%}.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                 t1.second, (epoch+1)/epoch_num)
            if torch.cuda.is_available() and cuda_flg:
                PATH = "model/Dual_{}_{}_{}_{}_{}_{}_{:.0%}_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                        t1.second, (epoch + 1) / epoch_num)
            torch.save(net.state_dict(), PATH)
            print("{} is saved.".format(PATH))

    print('Finished Training')
    #PATH = './value_net.pth'

    #PATH = './value_net.pth'


    PATH = "model/Dual_{}_{}_{}_{}_{}_{}_all.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                         t1.second)
    if torch.cuda.is_available() and cuda_flg:
        PATH = "model/Dual_{}_{}_{}_{}_{}_{}_all_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                             t1.second)
    torch.save(net.state_dict(), PATH)
    print("{} is saved.".format(PATH))
    t2 = datetime.datetime.now()
    print(t2)
    print(t2-t1)
