import os
os.environ["OMP_NUM_THREADS"] = "4"
import torch
#import torchvision
#import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import copy
import random
from my_enum import *
import torch.optim as optim
#from pytorch_memlab import profile
import argparse
from torch.autograd import detect_anomaly
# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Dual_State_value = namedtuple('Value', ('state', 'action', 'before_state', 'detailed_action_code','reward'))
Detailed_State_data = namedtuple('Value', ('hand_ids', 'hand_card_costs', 'follower_card_ids',
                                         'amulet_card_ids', 'follower_stats', 'follower_abilities', 'able_to_evo',
                                         'life_data', 'pp_data','able_to_play','able_to_attack', 'able_to_creature_attack',
                                           'deck_data'))
"""
   input = {'values', 'hand_ids','follower_card_ids', 
        'amulet_card_ids', 'follower_abilities', 'able_to_evo'}
"""

class New_Dual_Net(nn.Module):
    def __init__(self,n_mid):
        super(New_Dual_Net, self).__init__()
        self.state_net =Dual_State_Net(n_mid)
        #self.life_layer = nn.Linear(5,n_mid)#お互いの体力、手札枚数、経過ターン
        #self.pp_layer = nn.Linear(2, n_mid)#最大PP,残りPP

        #self.lin3 = nn.Linear(9, n_mid)#最大手札枚数
        #self.lin3 = nn.Linear(2,n_mid)#フォロワーの攻撃力,体力
        #self.follower_layer = nn.Linear(5,n_mid)#フォロワーの攻撃力,体力,フォロワーであるか,フォロワーに攻撃可能、プレイヤーに攻撃可能
        #self.hand_convert_layer = nn.Linear(3 * n_mid + 1, n_mid)#手札連結
        #self.lin5 = nn.Linear(6 * n_mid, n_mid)#自分の場のフォロワー連結
        #self.lin6 = nn.Linear(4 * n_mid, n_mid)  # 相手の場のフォロワー連結

        #self.field_layer1 = nn.Linear(9 * n_mid//2, n_mid)
        #self.field_layer2 = nn.Linear(10*n_mid,n_mid)

        #self.deck_layer = nn.Linear(40*n_mid,n_mid)
        #self.hand_card_layer = nn.Linear(9*n_mid,n_mid)
        #self.amulet_layer = nn.Linear(10*n_mid,n_mid)
        self.emb1 = self.state_net.emb1#nn.Embedding(3000,n_mid,padding_idx=0)#1000枚*3カテゴリー（空白含む）
        #self.emb2 = self.emb1
        #self.emb2 = nn.Embedding(1000, n_mid, padding_idx=0)  # フォロワー1000枚（空白含む）

        #self.emb3 = self.emb1
        #self.emb3 = nn.Embedding(1000, n_mid, padding_idx=0)  # アミュレット1000枚（空白含む）
        #self.emb4 = nn.Embedding(16, n_mid, padding_idx=0)  # キーワード能力15個と空白
        #self.emb4 = nn.Embedding(16, n_mid//10, padding_idx=0)
        #self.emb5 = nn.Embedding(6, n_mid, padding_idx=0)  # 進化可能最大五体と空白
        #self.emb6 = nn.Embedding(10, n_mid, padding_idx=0)  # 手札最大9枚の位置と空白
        #self.emb7 = nn.Embedding(6, n_mid, padding_idx=0)  # プレイヤーに攻撃可能な最大五体と空白
        #self.emb8 = nn.Embedding(6, n_mid, padding_idx=0)  # プレイヤーに攻撃可能な最大五体と空白
        #self.emb9 = nn.Embedding(9, n_mid, padding_idx=0)#両プレイヤーのリーダークラス
        #self.fc0 = nn.Linear(6*n_mid,n_mid)
        #self.fc0 = nn.Linear(7*n_mid, n_mid)
        #self.fc1 = nn.Linear(n_mid, n_mid)

        #self.small_layer = nn.Linear(n_mid,n_mid//10)
        #self.big_layer = nn.Linear(n_mid//10, n_mid)
        layer = [Dual_ResNet(2*n_mid, 2*n_mid) for _ in range(20)]
        self.layer = nn.ModuleList(layer)
        self.layer_len = len(self.layer)

        #self.fc3_p1 = nn.Linear(n_mid, 25)
        #self.bn_p1 = nn.BatchNorm1d(25)
        # 手札の枚数+自分の場のフォロワーの数*3＋ターン終了
        #self.fc3_p2 = nn.Linear(25, 25)
        self.action_value_net = Action_Value_Net(self,mid_size=n_mid)

        #self.fc3_v1 = nn.Linear(n_mid, 5)
        #self.bn_v1 = nn.BatchNorm1d(5)
        #self.fc3_v2 = nn.Linear(5, 2)
        #self.fc3_v3 = nn.Linear(2, 1)

        self.loss_fn = Dual_Loss()
        self.filtered_softmax = filtered_softmax()
        self.n_mid = n_mid
        self.mish = Mish()
        #self.direct_layer = nn.Linear(n_mid, n_mid)
        self.final_layer = nn.Linear(n_mid,1)
        #self.value_layer = nn.Linear(5+15,1)
        #self.concat_layer = nn.Linear(n_mid+26+1,n_mid)
        #self.hand_value_layer = nn.Linear(n_mid,1)
        #self.class_eye = torch.cat([torch.Tensor([[0] * 8]), torch.eye(8)], dim=0)

        #self.ability_eye = torch.cat([torch.Tensor([[0] * 15]), torch.eye(15)], dim=0)

        #self.prelu_1 = nn.PReLU(init=0.01)
        #self.prelu_2 = nn.PReLU(init=0.01)
        #self.prelu_3 = nn.PReLU(init=0.01)
        #self.prelu_4 = nn.PReLU(init=0.01)
        self.prelu = nn.PReLU(init=0.01)
        self.integrate_layer = nn.Linear(2*n_mid,n_mid)



    #@profile
    def forward(self, states,target=False):

        values = states['values']
        detailed_action_codes = states['detailed_action_codes']
        action_categories = detailed_action_codes['action_categories']
        play_card_ids = detailed_action_codes['play_card_ids']
        field_card_ids = detailed_action_codes['field_card_ids']
        able_to_choice = detailed_action_codes['able_to_choice']
        action_choice_len = detailed_action_codes['action_choice_len']
        x1 = self.state_net(states)
        x2 = self.state_net(states["before_states"])
        x = torch.cat([x1,x2],dim=1)#self.prelu(self.integrate_layer(torch.cat([x1,x2],dim=1)))
        for i in range(self.layer_len):
            x = self.layer[i](x)
        x=self.prelu(self.integrate_layer(x))
        tmp = self.action_value_net(x, action_categories, play_card_ids, field_card_ids,values,able_to_choice,target=target)
        h_p2 = tmp

        out_p = self.filtered_softmax(h_p2, able_to_choice)

        out_v = torch.tanh(self.final_layer(x))
        if target:
            z = states['target']['rewards']
            pai = states['target']['actions']
            return out_p, out_v, self.loss_fn(out_p, out_v, z, pai,action_choice_len)
        else:
            return out_p, out_v

class Dual_State_Net(nn.Module):
    def __init__(self, n_mid):
        super(Dual_State_Net, self).__init__()
        self.value_layer = nn.Linear(5+15+1,1)
        self.life_layer = nn.Linear(5, n_mid)
        self.hand_value_layer = nn.Linear(n_mid,1)
        self.field_value_layer = nn.Linear(n_mid, 1)
        self.emb1 = nn.Embedding(3000,n_mid,padding_idx=0)
        self.concat_layer = nn.Linear(n_mid+26+1,n_mid)
        self.class_eye = torch.cat([torch.Tensor([[0] * 8]), torch.eye(8)], dim=0)

        self.ability_eye = torch.cat([torch.Tensor([[0] * 15]), torch.eye(15)], dim=0)

        self.prelu_1 = nn.PReLU(init=0.01)
        self.prelu_2 = nn.PReLU(init=0.01)
        self.prelu_3 = nn.PReLU(init=0.01)
        self.prelu_4 = nn.PReLU(init=0.01)
        self.prelu_5 = nn.PReLU(init=0.01)

    def forward(self, states):
        values = states['values']
        hand_ids = states['hand_ids']
        follower_card_ids = states['follower_card_ids']
        amulet_card_ids = states['amulet_card_ids']
        follower_abilities = states['follower_abilities']
        class_datas = values['class_datas']
        stats = values['follower_stats']
        class_values = self.class_eye[class_datas].view(-1,16).to(stats.device)
        x4 = self.ability_eye[follower_abilities]
        x4 = torch.sum(x4,dim=2)
        abilities = x4.to(stats.device)
        field_card_ids = self.prelu_5(self.field_value_layer(self.emb1(follower_card_ids))).view(-1, 10,1)
        x1 = torch.cat([stats, abilities,field_card_ids],dim=2)
        x1 = self.prelu_1(self.value_layer(x1))
        exist_filter = (follower_card_ids != 0).float().view(-1,10,1)
        x1 = x1 * exist_filter
        follower_values=x1.view(-1,10)
        life_values = self.prelu_2(self.life_layer(values['life_datas']))
        #if True in torch.isnan(life_values):
        #    print(life_values)
        #    assert False,"nan in life_values"
        hand_ids = self.prelu_3(self.hand_value_layer(self.emb1(hand_ids))).view(-1, 9)
        #if True in torch.isnan(hand_ids):
        #    print(hand_ids)
        #    assert False,"nan in hand_ids"
        hand_card_values = torch.sum(hand_ids,dim=1).view(-1,1)
        x1 = torch.cat([follower_values,life_values,class_values,hand_card_values],dim=1)

        x1 = self.prelu_4(self.concat_layer(x1))
        #if True in torch.isnan(x1):
        #    print(x1)
        #    assert False,"nan in second_x1"

        return x1

class Dual_ResNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(Dual_ResNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)
        self.prelu1 = nn.PReLU(init=0.01)
        self.prelu2 = nn.PReLU(init=0.01)
        #self.mish = Mish()

    def forward(self, x):

        #return torch.sigmoid(self.fc1(x))
        h1 = self.prelu1(self.fc1(x))
        #h2 = F.relu(self.fc2(h1))
        h2 = self.prelu2(self.fc2(h1) + x)
        #h1 = self.mish(self.fc1(x))
        #h2 = self.mish(self.fc2(h1) + x)
        return h2



class Action_Value_Net(nn.Module):
    def __init__(self,parent_net,mid_size = 100):
        super(Action_Value_Net, self).__init__()
        self.mid_size = mid_size
        self.emb1 = nn.Embedding(5, mid_size)  # 行動のカテゴリー
        self.emb2 = parent_net.emb1#nn.Embedding(3000, mid_size, padding_idx=0)  # 1000枚*3カテゴリー（空白含む）
        #self.emb3 = nn.Embedding(1000, mid_size, padding_idx=0)  # フォロワー1000枚
        self.lin1 = nn.Linear(5*mid_size+4, mid_size)#nn.Linear(7 * mid_size, mid_size)
        #self.lin1 = nn.Linear(5 * mid_size, mid_size)
        self.lin2 = nn.Linear(mid_size, 1)
        #self.lin3 = nn.Linear(36,mid_size)
        self.lin3 = nn.Linear(66, mid_size)
        layer = [Dual_ResNet(mid_size, mid_size) for _ in range(10)]
        self.lin4 = nn.ModuleList(layer)
        #self.mish = Mish()
        self.action_catgory_eye = torch.cat([torch.Tensor([[0] * 4]), torch.eye(4)], dim=0)
        self.prelu_1 = nn.PReLU(init=0.01)
        self.prelu_2 = nn.PReLU(init=0.01)

        self.prelu_3 = nn.PReLU(init=0.01)
        self.prelu_4 = nn.PReLU(init=0.01)


    def forward(self, states, action_categories, play_card_ids, field_card_ids,values,label,target=False):
        life_datas = values['life_datas']
        pp_datas = values['pp_datas']
        hand_card_costs = values['hand_card_costs']
        stats = values['follower_stats'].view(-1,50)

        embed_action_categories = self.action_catgory_eye[action_categories].to(stats.device)#self.emb1(action_categories)(-1,45,4)
        embed_play_card_ids = self.emb2(play_card_ids)
        embed_play_card_ids = self.prelu_3(embed_play_card_ids)

        embed_field_card_ids = self.emb2(field_card_ids).view(-1,45,3*self.mid_size)#self.emb3(field_card_ids).view(-1,45,3*self.mid_size)
        embed_field_card_ids = self.prelu_4(embed_field_card_ids)

        new_states = states#.unsqueeze(1)
        new_states = torch.stack([new_states]*45,dim=1)
        tmp = torch.cat([new_states,embed_action_categories,embed_play_card_ids,embed_field_card_ids], dim=2)
        output = self.prelu_1(self.lin1(tmp))
        output = self.prelu_2(self.lin2(output)).view(-1,45)
        output = output * label

        return output


class filtered_softmax(nn.Module):
    def __init__(self):
        super(filtered_softmax, self).__init__()

    def forward(self, x, label):
        x = torch.softmax(x,dim=1)
        x = x*label
        return x

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        tmp_x = x * torch.tanh(F.softplus(x))
        #if True in torch.isnan(tmp_x):
        #    print(x)
        #    print(tmp_x)
        #    assert False
        return tmp_x

class Dual_Loss(nn.Module):

    def __init__(self):
        super(Dual_Loss, self).__init__()

    def forward(self, p, v, z, pai,action_choice_len):
        #l = (z − v)^2 − πlog p + c||θ||2
        #paiはスカラー値
        #print("p:{}".format(p[0:10]))
        #print("z:{}".format(z[0:10]))
        loss = torch.sum(
            torch.pow((z - v),2),
           dim=1)
        #loss = torch.sum(torch.abs(z-v),dim=1)
        #print("z:{}".format(z))
        #print("v:{}".format(v))
        #print("loss:{}".format(loss))
        MSE = torch.mean(loss)
        #print("loss:",loss)
        #print("mean:",MSE)

        tmp_CEE = p[range(p.size()[0]),pai]+1.0e-8
        choice_len_term = 1/torch.sqrt(action_choice_len)
        #print(choice_len_term)
        CEE = -torch.log(tmp_CEE)*choice_len_term
        CEE = torch.mean(CEE)
        #pai = pai.t()[0]
        #CEE = self.cross_entropy(p,pai)#softmaxも含まれている
        loss = MSE + CEE
        #L2正則化はoptimizer

        return loss, MSE, CEE

class New_Dual_ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = []
        self.index = 0

    def push(self, state, action, before_state, detailed_action_code, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリが満タンじゃないときには追加
        self.memory[self.index] = Dual_State_value(state, action, before_state, detailed_action_code, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size,all=False,cuda=False):
        if all:
            #tmp = self.memory
            tmp = random.sample(self.memory, len(self.memory))
        else:
            tmp = random.sample(self.memory, batch_size)
        states = [cell.state for cell in tmp]
        states = Detailed_State_data_2_Tensor(states,cuda=cuda)
        before_states = [cell.before_state for cell in tmp]
        before_states = Detailed_State_data_2_Tensor(before_states,cuda=cuda)
        actions = [cell.action for cell in tmp]
        actions = torch.LongTensor(actions)#torch.stack(actions, dim=0)
        rewards = [[cell.reward] for cell in tmp]
        rewards = torch.FloatTensor(rewards)
        detailed_action_codes = [cell.detailed_action_code for cell in tmp]
        detailed_action_codes = Detailed_action_code_2_Tensor(detailed_action_codes,cuda=cuda)
        states['detailed_action_codes'] = detailed_action_codes
        states['before_states'] = before_states
        if cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()

        return states, actions, rewards

    def __len__(self):
        return len(self.memory)


def get_data(f,player_num=0):
    hand_ids = []
    hand_card_costs = []
    player = f.players[player_num]
    opponent = f.players[1-player_num]
    """
    for hand_card in player.hand:
        #hand_ids.append(Card_Category[hand_card.card_category].value*(hand_card.card_id+500))
        none_flg = int(Card_Category[hand_card.card_category].value != 0)
        converted_card_id = (Card_Category[hand_card.card_category].value-1)*1000+\
                            hand_card.card_id+500
        hand_ids.append(none_flg*converted_card_id)
        hand_card_costs.append(hand_card.cost)
    """
    #hand_ids = [int(Card_Category[card.card_category].value != 0)*\
    #                       ((Card_Category[card.card_category].value-1)*1000+
    #                            card.card_id+500)for card in player.hand]
    hand_ids = [((Card_Category[card.card_category].value-1)*1000+
                                card.card_id+500)for card in player.hand]
    #hand_card_costs = [card.cost for card in player.hand]
    hand_card_costs = [card.cost/10 for card in player.hand]
    """
    for j in range(len(player.hand),9):
        hand_ids.append(0)
        hand_card_costs.append(0)
    """
    hand_ids.extend([0]*(9-len(player.hand)))
    hand_card_costs.extend([0]*(9-len(player.hand)))
    deck_data = sorted([((Card_Category[card.card_category].value-1)*1000+
                                card.card_id+500) for card in player.deck.deck])
    deck_data.extend([0]*(40-len(player.deck.deck)))
    #follower_card_ids = []
    #amulet_card_ids = []
    #follower_stats = []
    #follower_abilities = []
    opponent_num = 1- player_num
    able_to_evo = f.get_able_to_evo(player)
    able_to_evo = [cell+1 for cell in able_to_evo]
    follower_card_ids = [f.card_location[player_num][i].card_id + 500
                         if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Creature" else 0 for i in range(5)] \
                        + [f.card_location[opponent_num][i].card_id + 500
                           if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Creature" else 0 for i in range(5)]
    follower_stats = [[f.card_location[player_num][i].power/10, f.card_location[player_num][i].get_current_toughness()/10,
                       1, int(f.card_location[player_num][i].can_attack_to_follower()), int(f.card_location[player_num][i].can_attack_to_player())]
                      if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Creature" else [0, 0, 0, 0, 0] for i in range(5)] \
                     + [[f.card_location[opponent_num][i].power/10, f.card_location[opponent_num][i].get_current_toughness()/10,
                         1, 1, 1]
                        if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Creature" else [0, 0, 0, 0, 0] for i in range(5)]
    """
    follower_stats = [[f.card_location[player_num][i].power/100, f.card_location[player_num][i].get_current_toughness()/100]
                      if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Creature" else [0, 0] for i in range(5)] \
                     + [[f.card_location[opponent_num][i].power/100, f.card_location[opponent_num][i].get_current_toughness()/100]
                        if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Creature" else [0, 0] for i in range(5)]
    """
    follower_abilities = [f.card_location[player_num][i].ability[:]
                          if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Creature" else [] for i in range(5)] \
                         + [f.card_location[opponent_num][i].ability[:]
                            if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Creature" else [] for i in range(5)]

    #amulet_card_ids = [f.card_location[player_num][i].card_id + 500
    #                   if i < len(f.card_location[player_num]) and f.card_location[player_num][
    #    i].card_category == "Amulet" else 0 for i in range(5)] \
    #                  + [f.card_location[opponent_num][i].card_id + 500
    #                     if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
    #    i].card_category == "Amulet" else 0 for i in range(5)]
    amulet_card_ids = [f.card_location[player_num][i].card_id + 500
                       if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Amulet" else 0 for i in range(5)] \
                      + [f.card_location[opponent_num][i].card_id + 500
                         if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Amulet" else 0 for i in range(5)]
    """
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
    """
    able_to_play = f.get_able_to_play(player)
    able_to_play = [cell+1 for cell in able_to_play]
    able_to_attack = f.get_able_to_attack(player)
    able_to_creature_attack = f.get_able_to_creature_attack(player)
    #life_data = [player.life, opponent.life,len(player.hand),len(opponent.hand) ,f.current_turn[player_num]]
    life_data = [player.life/20, opponent.life/20, len(player.hand)/9, len(opponent.hand)/9,f.current_turn[player_num]/10]
    #pp_data = [f.cost[player_num],f.remain_cost[player_num]]
    pp_data = [f.cost[player_num]/10, f.remain_cost[player_num]/10]

    class_data = [player.deck.leader_class.value,
                    opponent.deck.leader_class.value]
    life_data = (life_data, class_data)
    datas = Detailed_State_data(hand_ids, hand_card_costs, follower_card_ids, amulet_card_ids,
                              follower_stats, follower_abilities, able_to_evo, life_data, pp_data,
                              able_to_play, able_to_attack, able_to_creature_attack,deck_data)

    return datas



def Detailed_State_data_2_Tensor(datas,cuda=False):
    data_len = len(datas)
    #print(type(datas))
    #print(type(datas[0]))
    #hand_ids = torch.LongTensor([[0 for _ in range(9)] for _ in range(data_len)])
    hand_ids = torch.LongTensor([[datas[i].hand_ids[j] for j in range(9)] for i in range(data_len)])
    #hand_card_costs = torch.Tensor([[0 for j in range(9)] for i in range(data_len)])
    hand_card_costs = torch.Tensor([[datas[i].hand_card_costs[j] for j in range(9)] for i in range(data_len)])
    follower_card_ids = torch.LongTensor([[datas[i].follower_card_ids[j] for j in range(10)] for i in range(data_len)])
    #follower_card_ids = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    amulet_card_ids = torch.LongTensor([[datas[i].amulet_card_ids[j] for j in range(10)] for i in range(data_len)])
    #amulet_card_ids = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    follower_abilities = torch.LongTensor([[[datas[i].follower_abilities[j][k] if k < len(
        datas[i].follower_abilities[j]) else 0 for k in range(15)] for j in range(10)] for i in range(data_len)])
    follower_stats = torch.Tensor([[datas[i].follower_stats[j] for j in range(10)] for i in range(data_len)])
    #follower_stats = torch.Tensor([[(0,0) for _ in range(10)] for _ in range(data_len)])
    #able_to_evo = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    able_to_evo = torch.LongTensor(
        [[datas[i].able_to_evo[j] if j < len(datas[i].able_to_evo) else 0 for j in range(10)] for i in range(data_len)])
    able_to_play = torch.LongTensor(
        [[datas[i].able_to_play[j] if j < len(datas[i].able_to_play) else 0 for j in range(9)] for i in
         range(data_len)])
    #able_to_play = torch.LongTensor([[0 for _ in range(9)] for _ in range(data_len)])
    able_to_attack = torch.LongTensor(
        [[datas[i].able_to_attack[j] if j < len(datas[i].able_to_attack) else 0 for j in range(5)] for i in
         range(data_len)])
    able_to_creature_attack = torch.LongTensor(
        [[datas[i].able_to_creature_attack[j] if j < len(datas[i].able_to_creature_attack) else 0 for j in range(5)] for
         i in range(data_len)])

    pp_datas = torch.Tensor([datas[i].pp_data for i in range(data_len)])
    life_datas = torch.Tensor([datas[i].life_data[0] for i in range(data_len)])
    class_datas = torch.LongTensor([datas[i].life_data[1] for i in range(data_len)])
    deck_datas = torch.LongTensor([datas[i].deck_data for i in range(data_len)])

    ans = {'values': {'life_datas': life_datas,
                      'class_datas': class_datas,
                      'hand_card_costs': hand_card_costs,
                      'follower_stats': follower_stats,
                      'pp_datas': pp_datas,
                      'able_to_play': able_to_play,
                      'able_to_attack': able_to_attack,
                      'able_to_creature_attack': able_to_creature_attack,
                      },
           'hand_ids': hand_ids,
           'follower_card_ids': follower_card_ids,
           'amulet_card_ids': amulet_card_ids,
           'follower_abilities': follower_abilities,
           'able_to_evo': able_to_evo,
           'deck_datas':deck_datas}
    if cuda:
        for key in list(ans.keys()):
            if key == "values":
                for sub_key in list(ans["values"].keys()):
                    ans["values"][sub_key] = ans["values"][sub_key].cuda()
            else:
                ans[key] = ans[key].cuda()
    return ans


def Detailed_action_code_2_Tensor(action_codes, cuda = False):
    action_code_len = len(action_codes)

    tensor_action_categories = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][0] for j in range(45)] for i in range(action_code_len)])
    tensor_play_card_ids_in_action = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][1] for j in range(45)] for i in range(action_code_len)])
    tensor_field_card_ids_in_action = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][2:5] for j in range(45)] for i in range(action_code_len)])
    able_to_choice = torch.Tensor([action_codes[i]['able_to_choice'] for i in range(action_code_len)])
    action_choice_len = torch.Tensor([[int(sum(action_codes[i]['able_to_choice']))] for i in range(action_code_len)])
    #assert 0 not in action_choice_len,"{}".format(action_choice_len)
    #print(able_to_choice)
    action_codes_dict = {'action_categories': tensor_action_categories,
                         'play_card_ids': tensor_play_card_ids_in_action,
                         'field_card_ids': tensor_field_card_ids_in_action,
                         'able_to_choice': able_to_choice,
                         'action_choice_len':action_choice_len}
    if cuda:
        for key in list(action_codes_dict.keys()):
            action_codes_dict[key] = action_codes_dict[key].cuda()

    return action_codes_dict




deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                  6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic", -2: "Sword_Basic",
                  -3: "Rune_Basic",
                  -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
                  100: "TEST",
                  -9: "Spell-Rune", 11: "PtP-Forest", 12: "Mid-Shadow", 13: "Neutral-Blood"}

key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                  3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                  6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                  9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                  11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"],
                  13: ["Neutral_Blood.tsv", "BLOOD"],100: ["TEST.tsv", "SHADOW"],
                  -2: ["Sword_Basic.tsv", "SWORD"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='デュアルニューラルネットワーク学習コード')

    parser.add_argument('--episode_num', help='試行回数')
    parser.add_argument('--iteration_num', help='イテレーション数')
    parser.add_argument('--epoch_num', help='エポック数')
    parser.add_argument('--batch_size', help='バッチサイズ')
    parser.add_argument('--mcts', help='サンプリングAIをMCTSにする(オリジナルの場合は[OM])')
    parser.add_argument('--deck', help='サンプリングに用いるデッキの選び方')
    parser.add_argument('--cuda', help='gpuを使用するかどうか')
    parser.add_argument('--multi_train', help="学習時も並列化するかどうか")
    parser.add_argument('--epoch_interval', help="モデルの保存間隔")
    parser.add_argument('--fixed_deck_id', help="使用デッキidの固定")
    parser.add_argument('--cpu_num', help="使用CPU数", default=2 if torch.cuda.is_available() else 3)
    parser.add_argument('--batch_num', help='サンプルに対するバッチの数')
    args = parser.parse_args()
    deck_flg = int(args.fixed_deck_id) if args.fixed_deck_id is not None else None
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

    #deck_sampling_type = False
    #if args.deck is not None:
    #    deck_sampling_type = True

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
    optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
    for epoch in range(epoch_num):
        print("epoch {}".format(epoch+1))
        R = New_Dual_ReplayMemory(100000)
        p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=cuda_flg))
        p1.name = "Alice"
        p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net, cuda=cuda_flg))
        p2.name = "Bob"
        win_num = 0
        for episode in tqdm(range(episode_len)):
            f = Field(5)
            deck_type1 = deck_flg
            deck_type2 = deck_flg
            if deck_flg is None:
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
            for i in range(2):
                for data in train_data[i]:
                    R.push(data[0], data[1], data[2], data[3], reward[i])
            win_num += int(reward[episode % 2] > 0)

        print("sample_size:{}".format(len(R.memory)))
        print("win_rate:{:.2%}".format(win_num/episode_len))
        prev_net = copy.deepcopy(net)

        sum_of_loss = 0
        sum_of_MSE = 0
        sum_of_CEE = 0
        p,pai,z,states,loss = None, None, None, None,None
        current_net, prev_optimizer = None, None
        for i in tqdm(range(iteration)):
            print("\ni:{}\n".format(i))
            states, actions, rewards = R.sample(batch_size)

            states['target'] = {'actions':actions, 'rewards':rewards}
            p, v, loss = net(states,target=True)
            z = rewards
            pai = actions#45種類の抽象化した行動
            if (i + 1) % 100== 0:
                print("target:{} output:{}".format(z[0],v[0]))
                print("target:{} output:{}".format(pai[0], p[0]))
                print("loss:{}".format([loss[j].item() for j in range(3)]))
            if torch.isnan(loss):
                # section 3
                net = current_net
                optimizer = torch.optim.Adam(net.parameters())
                optimizer.load_state_dict(prev_optimizer.state_dict())
            else:
                current_net = copy.deepcopy(net)
                prev_optimizer = copy.deepcopy(optimizer)
            optimizer.zero_grad()
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
