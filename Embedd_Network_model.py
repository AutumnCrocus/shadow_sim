# -*- coding: utf-8 -*-

import torch
import os
#os.environ["OMP_NUM_THREADS"] = "4" if torch.cuda.is_available() else "6"
os.environ["PYTHONWARNINGS"] = 'ignore:semaphore_tracker:UserWarning'
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
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class New_Dual_Net(nn.Module):
    def __init__(self,n_mid):
        super(New_Dual_Net, self).__init__()
        self.state_net =Dual_State_Net(n_mid)
        
        self.emb1 = self.state_net.emb1#nn.Embedding(3000,n_mid,padding_idx=0)#1000枚*3カテゴリー（空白含む）
        
        layer = [Dual_ResNet(n_mid+25,n_mid+25) for _ in range(3)]
        self.layer = nn.ModuleList(layer)
        self.layer_len = len(self.layer)
        self.action_value_net = Action_Value_Net(self,mid_size=n_mid)
        self.loss_fn = Dual_Loss()
        self.filtered_softmax = filtered_softmax()
        self.n_mid = n_mid
        self.mish = torch.sigmoid#Mish()
        #self.direct_layer = nn.Linear(n_mid, n_mid)
        preprocess_layer = [Dual_ResNet(n_mid,n_mid) for _ in range(5)]
        self.preprocess_layer = nn.ModuleList(preprocess_layer)
        self.final_layer = nn.Linear(n_mid,1)
        nn.init.kaiming_normal_(self.final_layer.weight)
        #self.conv = nn.Conv1d(in_channels=100,out_channels=1,kernel_size=1)
        self.relu = torch.tanh#torch.sigmoid()#nn.ReLU()
        self.prelu = torch.tanh#torch.sigmoid()#nn.PReLU(init=0.01)
        self.integrate_layer = nn.Linear(n_mid+25,n_mid)
        nn.init.kaiming_normal_(self.integrate_layer.weight)
        self.rnn = nn.LSTM(input_size=n_mid,hidden_size=n_mid,batch_first=True,num_layers=3)
        #nn.init.kaiming_normal_(self.rnn.weight)
        #encoder_layers = nn.TransformerEncoderLayer(n_mid, 4 ,dropout=0.01)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)
        ans = {'values': {'life_datas': None,
                          'class_datas': None,
                          'deck_type_datas': None,
                          'hand_card_costs': None,
                          'follower_stats': None,
                          'pp_datas': None,
                          'able_to_play': None,
                          'able_to_attack': None,
                          'able_to_creature_attack': None,
                          },
               'hand_ids': None,
               'follower_card_ids': None,
               'amulet_card_ids': None,
               'follower_abilities': None,
               'able_to_evo': None,
               'deck_datas': None,
               'detailed_action_codes': {'action_categories': None,
                         'play_card_ids': None,
                         'field_card_ids': None,
                         'able_to_choice': None,
                         'action_choice_len':None},
               'before_states':{'values': {'life_datas': None,
                          'class_datas': None,
                          'deck_type_datas': None,
                          'hand_card_costs': None,
                          'follower_stats': None,
                          'pp_datas': None,
                          'able_to_play': None,
                          'able_to_attack': None,
                          'able_to_creature_attack': None,
                          },
                   'hand_ids': None,
                   'follower_card_ids': None,
                   'amulet_card_ids': None,
                   'follower_abilities': None,
                   'able_to_evo': None,
                   'deck_datas': None}
               }
        self.states_keys = tuple(ans.keys())
        self.normal_states_keys = tuple(set(self.states_keys) - {'values', 'detailed_action_codes', 'before_states'})
        self.value_keys = tuple(ans['values'].keys())
        self.action_code_keys = tuple(ans['detailed_action_codes'].keys())
        self.cuda_flg = False

        value_layer = [nn.Linear(n_mid,n_mid)for _ in range(3)]
        for i in range(len(value_layer)):
            nn.init.kaiming_normal_(value_layer[i].weight)
        self.value_layer = nn.ModuleList(value_layer)



    #@profile
    def forward(self, states,target=False):

        values = states['values']
        detailed_action_codes = states['detailed_action_codes']
        # action_categories = detailed_action_codes['action_categories']
        # play_card_ids = detailed_action_codes['play_card_ids']
        # field_card_ids = detailed_action_codes['field_card_ids']
        able_to_choice = detailed_action_codes['able_to_choice']
        action_choice_len = detailed_action_codes['action_choice_len']
        current_states = self.state_net(states)
        before_states = states["before_states"]
        #print("size:",before_states.size())
        split_states = torch.split(before_states,[1,1,1,1],dim=1)
        embed_action_categories = self.action_value_net.action_catgory_eye[split_states[0]]
        #.to(stats.device)#self.emb1(action_categories)(-1,45,4)
        embed_acting_card_ids = self.action_value_net.emb2(split_states[1])
        embed_acting_card_ids = self.action_value_net.prelu_3(embed_acting_card_ids)
        embed_acted_card_ids = self.action_value_net.emb2(split_states[2])#(-1,45,n_mid,?)
        embed_acted_card_sides = self.action_value_net.side_emb(split_states[3])  # (-1,45,?,n_mid)
        #print("split:",embed_action_categories.size(),embed_acting_card_ids.size(),
        #     embed_acted_card_ids.size(),embed_acted_card_sides.size())
        before_states = torch.cat([embed_action_categories,embed_acting_card_ids,
                                  embed_acted_card_ids,embed_acted_card_sides],dim=2).view(-1,25)
        #before_states = self.state_net(states["before_states"])
        current_states = current_states# - before_states
        #print(x_1[0])
        #x_2 = self.state_net(states["before_states"])
        #x = torch.cat([x1,x2],dim=1)#self.prelu(self.integrate_layer(torch.cat([x1,x2],dim=1)))
        #x = torch.stack([x2,x1],dim=1)
        #x.size() = (-1,100)
        #required (-1,2,100)
        #x_3 = torch.stack([x_2, x_1], dim=1)
        #x_3_1 = x_3.reshape(-1,2*self.n_mid)

        # x_4, (_,_) = self.rnn(x_3)
        # x_4 = x_4.reshape(-1,2*self.n_mid)
        
        #x_3 = x_1
        #_,(h0, c0) = self.rnn(x_2.unsqueeze(1))
        #x_3,(_,_) = self.rnn(x_1.unsqueeze(1),(h0,c0))
        
        #x_4=self.transformer_encoder(x_3).reshape(-1,self.n_mid)
        #x_4 = x_3#.reshape(-1,self.n_mid)#+x_1+x_2

        #print(x.size())


        x3 = torch.cat([current_states,before_states],dim=1)#current_states
        for i in range(self.layer_len):
            x3 = self.layer[i](x3)

        

        # for i in range(self.layer_len):
        #     x = self.layer[i](x)
        x=self.prelu(self.integrate_layer(x3))#+x3
        #print("current_states",current_states,"\n")
        #print("x3",x3,"\n")
        #print("x",x,"\n")
        tmp = self.action_value_net(x,detailed_action_codes,values,target=target)
        h_p2 = tmp

        out_p = self.filtered_softmax(h_p2, able_to_choice)

        #for i in range(3):
        #    x = self.relu(self.value_layer[i](x))
        #before_x = self.conv(x.unsqueeze(-1))
        v_x = x#x-x.max(dim=0).values if target else x
        #print(v_x[0])
        for i in range(3):
            v_x = self.preprocess_layer[i](v_x)
        out_v = torch.tanh(self.final_layer(v_x))#+before_x)
        if target:
            #if self.cuda_flg:states['target'] = {key:states['target'][key].cuda() \
            #                                    for key in ('rewards','actions')}
            #             print(x_4)
            #             print(x_1)
            #             print(x3)
            #             print(x)
            
            z = states['target']['rewards']
            pai = states['target']['actions']
            return out_p, out_v, self.loss_fn(out_p, out_v, z, pai,action_choice_len)
        else:
            return out_p, out_v

    def cuda(self):
        self.state_net.cuda_all()
        self.action_value_net.cuda_all()
        print("model is formed to cuda")
        self.cuda_flg = True
        return super(New_Dual_Net, self).cuda()


    def cpu(self):
        self.state_net.cpu()
        self.action_value_net.cpu()
        print("model is formed to cpu")
        self.cuda_flg = False
        return super(New_Dual_Net, self).cpu()

class Dual_State_Net(nn.Module):
    def __init__(self, n_mid):
        super(Dual_State_Net, self).__init__()
        self.short_mid = n_mid//10
        self.value_layer = nn.Linear(6+15+10+1+1+1,self.short_mid)#nn.Linear(5+15+n_mid,n_mid)
        nn.init.kaiming_normal_(self.value_layer.weight)

        self.life_layer = nn.Linear(5, self.short_mid)#nn.Linear(5, n_mid)
        nn.init.kaiming_normal_(self.life_layer.weight)

        self.hand_value_layer = nn.Linear(10, 10)#nn.Linear(n_mid, n_mid)
        nn.init.kaiming_normal_(self.hand_value_layer.weight)

        self.hand_integrate_layer = nn.Linear(10, self.short_mid)#nn.Linear(n_mid, n_mid)
        nn.init.kaiming_normal_(self.hand_integrate_layer.weight)

        self.deck_value_layer = nn.Linear(10, 10)#nn.Linear(n_mid, n_mid)
        nn.init.kaiming_normal_(self.deck_value_layer.weight)

        self.deck_integrate_layer = nn.Linear(10, self.short_mid)#nn.Linear(n_mid, n_mid)
        nn.init.kaiming_normal_(self.deck_integrate_layer.weight)

        self.amulet_value_layer = nn.Linear(10, self.short_mid)#nn.Linear(n_mid, n_mid)
        nn.init.kaiming_normal_(self.amulet_value_layer.weight)
        self.field_value_layer = nn.Linear(10, 10)#nn.Linear(n_mid, n_mid)
        nn.init.kaiming_normal_(self.field_value_layer.weight)

        self.emb1 = nn.Embedding(3000,10,padding_idx=0)
        #nn.Embedding(3000,n_mid,padding_idx=0)
        #nn.init.kaiming_normal_(self.emb1.weight)

        self.concat_layer = nn.Linear(self.short_mid,self.short_mid)
        nn.init.kaiming_normal_(self.concat_layer.weight)

        #self.concat_layer = nn.Linear(n_mid+10*2+1+16+8,n_mid)
        self.class_eye = torch.cat([torch.Tensor([[0] * 8]), torch.eye(8)], dim=0)

        self.ability_eye = torch.cat([torch.Tensor([[0] * 15]), torch.eye(15)], dim=0)
        self.deck_type_eye = torch.cat([torch.Tensor([[0] * 4]), torch.eye(4)], dim=0)
        self.pos_encoder = PositionalEncoding(n_mid, dropout=0.1)

        #prelu_layer = [Mish() for i in range(7)]
        #prelu_layer = [Mish() for i in range(7)]
        #[Mish() for i in range(7)]
        #[nn.PReLU(init=0.01) for i in range(7)]
        #self.prelu_layer = nn.ModuleList(prelu_layer)
        self.prelu_layer = Mish()#torch.tanh

        #self.modify_layer = nn.Linear(94*self.short_mid,4*n_mid)
        #self.second_modify_layer = nn.Linear(4*n_mid,2*n_mid)
        #self.third_modify_layer = nn.Linear(2*n_mid,n_mid)
        hidden_layer_num = 5
        origin = 94*self.short_mid
        node_shrink_range = (origin - n_mid) // hidden_layer_num
        self.modify_layer_num = hidden_layer_num
        node_size_list = [origin - i * node_shrink_range for i in range(hidden_layer_num)] + [n_mid]
        modify_layer = [nn.Linear(node_size_list[i], node_size_list[i+1]) for i in range(hidden_layer_num)]
        self.modify_layer = nn.ModuleList(modify_layer)

        #nn.init.kaiming_normal_(self.modify_layer.weight)
        self.n_mid = n_mid
        self.mish = torch.tanh
        #self.init_weights()
        #layer = [nn.Linear(n_mid,n_mid) for _ in range(3)]
        #self.layer = nn.ModuleList(layer)

    def cuda_all(self):
        self.class_eye = self.class_eye.cuda()
        self.ability_eye = self.ability_eye.cuda()
        self.deck_type_eye = self.deck_type_eye.cuda()
        return super(Dual_State_Net, self).cuda()

    def cpu(self):
        self.class_eye = self.class_eye.cpu()
        self.ability_eye = self.ability_eye.cpu()
        self.deck_type_eye = self.deck_type_eye.cpu()
        return super(Dual_State_Net, self).cpu()

    def init_weights(self):
        initrange = 0.1
        self.emb1.weight.data.uniform_(-initrange, initrange)

    def forward(self, states):
        values = states['values']
        hand_ids = states['hand_ids']
        follower_card_ids = states['follower_card_ids']
        amulet_card_ids = states['amulet_card_ids']
        follower_abilities = states['follower_abilities']
        life_datas = values['life_datas']
        class_datas = values['class_datas']
        deck_type_datas = values['deck_type_datas']
        stats = values['follower_stats']
        deck_datas = states["deck_datas"]
        able_to_attack = values["able_to_attack"].view(-1,10,1)
        able_to_creature_attack = values["able_to_creature_attack"].view(-1,10,1)
        able_to_evo = states["able_to_evo"].view(-1,10,1)
        #class_values = self.class_eye[class_datas].view(-1,16).to(stats.device)
        #deck_type_values = self.deck_type_eye[deck_type_datas].view(-1,8).to(stats.device)
        class_values = self.class_eye[class_datas].view(-1, 16).unsqueeze(-1)#.to(stats.device)
        class_values = class_values.expand(-1, 16, self.short_mid)#.expand(-1, 16, self.n_mid)
        deck_type_values = self.deck_type_eye[deck_type_datas].view(-1, 8).unsqueeze(-1)#.to(stats.device)
        deck_type_values = deck_type_values.expand(-1, 8, self.short_mid)#.expand(-1, 8, self.n_mid)
        x1 = self.ability_eye[follower_abilities]
        x1 = torch.sum(x1,dim=2)
        abilities = x1#.to(stats.device)

        src1 = self.emb1(follower_card_ids)
        # src1 = self.emb1(follower_card_ids)*np.sqrt(self.n_mid)
        # src1 = self.pos_encoder(src1)

        follower_cards = self.prelu_layer(self.field_value_layer(src1).view(-1, 10, 10))#+src1.view(-1, 10, 10))
        x2 = torch.cat([stats, abilities,follower_cards,able_to_attack,able_to_creature_attack,able_to_evo],dim=2)
        _follower_values = self.prelu_layer(self.value_layer(x2))
        
        follower_exist_filter = (follower_card_ids != 0).float().unsqueeze(-1)#.view(-1,self.n_mid)
        #print("follower_cards:{},exist_filter1:{}".format(follower_cards.size(),exist_filter1.size()))
        #exist_filter1.expand(*[-1,10,self.n_mid])
        follower_values = _follower_values * follower_exist_filter
        """
        follower_cards = self.prelu_5(self.field_value_layer(self.emb1(follower_card_ids))).view(-1, 10,1)
        x1 = torch.cat([stats, abilities,follower_cards],dim=2)
        x1 = self.prelu_1(self.value_layer(x1))
        exist_filter1 = (follower_card_ids != 0).float().view(-1,10,1)
        x1 = x1 * exist_filter1
        follower_values=x1.view(-1,10)
        """
        src2 = self.emb1(amulet_card_ids)
        # src2 = self.emb1(amulet_card_ids)*np.sqrt(self.n_mid)
        # src2 = self.pos_encoder(src2)
        amulet_cards = self.prelu_layer(self.field_value_layer(src2).view(-1, 10,10))#+src2.view(-1, 10,10))
        #print("amulet_cards:{}".format(amulet_cards.size()))
        _amulet_values = torch.sigmoid(self.amulet_value_layer(amulet_cards))#amulet_cards
        amulet_exist_filter = (amulet_card_ids != 0).float().unsqueeze(-1)#.view(-1,self.n_mid)
        #exist_filter2.expand(*[-1, 10, self.n_mid])
        amulet_values = _amulet_values * amulet_exist_filter
        """
        amulet_cards = self.prelu_5(self.field_value_layer(self.emb1(amulet_card_ids))).view(-1, 10,1)
        x2 = amulet_cards
        exist_filter2 = (amulet_cards != 0).float().view(-1,10,1)
        x2 = x2 * exist_filter2
        amulet_values=x2.view(-1,10)
        """


        life_values = self.prelu_layer(self.life_layer(life_datas)).view(-1, 1,self.short_mid)

        #hand_ids = self.prelu_3(self.hand_value_layer(self.emb1(hand_ids))).view(-1, 9)
        # hand_card_values = torch.sum(hand_ids,dim=1).view(-1,1)
        src3 = self.emb1(hand_ids)
        # src3 = self.emb1(hand_ids)*np.sqrt(self.n_mid)
        # src3 = self.pos_encoder(src3)
        hand_cards = self.prelu_layer(self.hand_value_layer(src3).view(-1, 9,10))#+src3.view(-1, 9,10))
        _hand_card_values = torch.sigmoid(self.hand_integrate_layer(hand_cards))#hand_cards
        hand_exist_filter = (hand_ids != 0).float().unsqueeze(-1)
        hand_card_values = _hand_card_values * hand_exist_filter

        src4 = self.emb1(deck_datas)
        # src4 = self.emb1(deck_datas)*np.sqrt(self.n_mid)
        # src4 = self.pos_encoder(src4)
        deck_cards = self.prelu_layer(self.deck_value_layer(src4).view(-1, 40,10))#+src4.view(-1,40,10))
        _deck_card_values = torch.sigmoid(self.deck_integrate_layer(deck_cards))#deck_cards
        deck_exist_filter = (deck_datas != 0).float().unsqueeze(-1)
        deck_card_values = _deck_card_values * deck_exist_filter
        
        #deck_card_values = torch.sum(hand_ids, dim=1).view(-1, 1)

        #x = torch.cat([follower_values,amulet_values,life_values,class_values,hand_card_values],dim=1)
        input_tensor = [follower_values,amulet_values,life_values,\
                       class_values,deck_type_values,hand_card_values,deck_card_values]
        #print([cell.size() for cell in input_tensor])
        before_x = torch.cat(input_tensor,dim=1)

        #x1 = torch.cat([follower_values,life_values,class_values,hand_card_values],dim=1)

        x = self.prelu_layer(self.concat_layer(before_x)).view(-1,94*self.short_mid)#+before_x).view(-1,94*self.short_mid)

        for i in range(self.modify_layer_num):
            x = self.prelu_layer(self.modify_layer[i](x))
        #x = self.prelu_layer(self.modify_layer(x))
        #x = self.prelu_layer(self.second_modify_layer(x))
        #x = self.prelu_layer(self.third_modify_layer(x))
        #print(before_x)
        #print(x)


        return x

class Dual_ResNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(Dual_ResNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.prelu1 = Mish()#nn.PReLU(init=0.01)
        self.prelu2 = Mish()#nn.PReLU(init=0.01)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        #self.mish = Mish()

    def forward(self, x):

        #return torch.sigmoid(self.fc1(x))
        h1 = self.bn1(self.prelu1(self.fc1(x)))
        #h2 = F.relu(self.fc2(h1))
        h2 = self.bn2(self.prelu2(self.fc2(h1) + x))
        #h1 = self.mish(self.fc1(x))
        #h2 = self.mish(self.fc2(h1) + x)
        return h2



class Action_Value_Net(nn.Module):
    def __init__(self,parent_net,mid_size = 100):
        super(Action_Value_Net, self).__init__()
        self.n_mid = mid_size
        self.short_mid = mid_size//10
        #self.emb1 = nn.Embedding(5, mid_size)  # 行動のカテゴリー
        #nn.init.kaiming_normal_(self.emb1.weight)
        self.emb2 = parent_net.emb1#nn.Embedding(3000, mid_size, padding_idx=0)  # 1000枚*3カテゴリー（空白含む）
        nn.init.kaiming_normal_(self.emb2.weight)
        #self.emb3 = nn.Embedding(1000, mid_size, padding_idx=0)  # フォロワー1000枚
        #self.lin1 = nn.Linear(5*mid_size+4, mid_size)#nn.Linear(7 * mid_size, mid_size)
        self.lin1 = nn.Linear(2*mid_size+10+4, mid_size)#nn.Linear(3*mid_size+4, mid_size)
        # #nn.Linear(7 * mid_size, mid_size)

        nn.init.kaiming_normal_(self.lin1.weight)
        #self.lin1 = nn.Linear(5 * mid_size, mid_size)
        self.lin2 = nn.Linear(mid_size, 1)
        nn.init.kaiming_normal_(self.lin2.weight)
        #self.lin3 = nn.Linear(36,mid_size)
        self.lin3 = nn.Linear(66, mid_size)
        nn.init.kaiming_normal_(self.lin3.weight)
        layer = [Dual_ResNet(mid_size, mid_size) for _ in range(3)]
        self.lin4 = nn.ModuleList(layer)
        #self.mish = Mish()
        self.action_catgory_eye = torch.cat([torch.Tensor([[0] * 4]), torch.eye(4)], dim=0)
        self.side_emb = nn.Embedding(3,1,padding_idx=2)
        self.association_layer = nn.Linear(10+1,mid_size)#nn.Linear(mid_size+1,mid_size)
        nn.init.kaiming_normal_(self.association_layer.weight)
        self.prelu_1 = nn.PReLU(init=0.01)
        self.prelu_2 = nn.PReLU(init=0.01)

        self.prelu_3 = nn.PReLU(init=0.01)
        self.prelu_4 = nn.PReLU(init=0.01)

    def cuda_all(self):
        self.action_catgory_eye = self.action_catgory_eye.cuda()
        return super(Action_Value_Net, self).cuda()

    def cpu(self):
        self.action_catgory_eye = self.action_catgory_eye.cpu()
        return super(Action_Value_Net, self).cpu()


    def forward(self, states, detailed_action_codes,values,target=False):
        life_datas = values['life_datas']
        pp_datas = values['pp_datas']
        hand_card_costs = values['hand_card_costs']
        stats = values['follower_stats'].view(-1,6*10)
        # action_categories = detailed_action_codes['action_categories']
        # play_card_ids = detailed_action_codes['play_card_ids']
        # field_card_ids = detailed_action_codes['field_card_ids']
        action_categories = detailed_action_codes['action_categories']
        acting_card_ids = detailed_action_codes['acting_card_ids']
        acted_card_ids = detailed_action_codes['acted_card_ids']
        acted_card_sides = detailed_action_codes['acted_card_sides']
        label = detailed_action_codes['able_to_choice']
        #action_choice_len = detailed_action_codes['action_choice_len']

        embed_action_categories = self.action_catgory_eye[action_categories]#.to(stats.device)#self.emb1(action_categories)(-1,45,4)

        # embed_play_card_ids = self.emb2(play_card_ids)
        # embed_play_card_ids = self.prelu_3(embed_play_card_ids)
        embed_acting_card_ids = self.emb2(acting_card_ids)
        embed_acting_card_ids = self.prelu_3(embed_acting_card_ids)

        # embed_field_card_ids = self.emb2(field_card_ids).view(-1,45,3*self.mid_size)#self.emb3(field_card_ids).view(-1,45,3*self.mid_size)
        # embed_field_card_ids = self.prelu_4(embed_field_card_ids)
        embed_acted_card_ids = self.emb2(acted_card_ids)#(-1,45,n_mid,?)

        #print("emb_acted:{}".format(embed_acted_card_ids.size()))
        embed_acted_card_sides = self.side_emb(acted_card_sides)  # (-1,45,?,n_mid)
        #print(acted_card_ids,acted_card_sides,embed_acted_card_ids.size(),embed_acted_card_sides.size())
        embed_acted_card_ids = torch.cat([embed_acted_card_ids,embed_acted_card_sides],dim=3)
        embed_acted_card_ids = torch.sigmoid(self.association_layer(embed_acted_card_ids))
        embed_acted_card_ids = torch.sum(embed_acted_card_ids,dim=2)
        embed_acted_card_ids = embed_acted_card_ids.view(-1,45,self.n_mid)
        #self.emb3(field_card_ids).view(-1,45,3*self.mid_size)
        embed_acted_card_ids = self.prelu_4(embed_acted_card_ids)

        new_states = states#.unsqueeze(1)
        new_states = torch.stack([new_states]*45,dim=1)
        input_tensors = [new_states, embed_action_categories, embed_acting_card_ids, embed_acted_card_ids]
        #print("{}".format([cell.size() for cell in input_tensors]))
        # tmp = torch.cat([new_states,embed_action_categories,embed_play_card_ids,embed_field_card_ids], dim=2)
        tmp = torch.cat(input_tensors, dim=2)
        output1 = self.prelu_1(self.lin1(tmp))
        output2 = self.prelu_2(self.lin2(output1)).view(-1,45)
        output = output2 * label

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
        return tmp_x

class Dual_Loss(nn.Module):

    def __init__(self):
        super(Dual_Loss, self).__init__()

    def forward(self, p, v, z, pai,action_choice_len):

        #tmp_MSE = torch.sum(
        #    torch.pow((z - v),2),
        #   dim=1)
        LOSS_EPSILON=1.0e-4
        tmp_MSE = torch.sum(
            -(z+1)*torch.log((v+1)/2+LOSS_EPSILON)/2+(z-1)*torch.log((1-v)/2+LOSS_EPSILON)/2,
           dim=1)
        
        

        MSE = torch.mean(tmp_MSE)
        #print("MSE:{}".format(MSE))
        #print("loss:",loss)
        #print("mean:",MSE)

        tmp_CEE1 = p[range(p.size()[0]),pai]+LOSS_EPSILON
        #choice_len_term = 1/torch.sqrt(action_choice_len)
        #print(choice_len_term)
        tmp_CEE2 = -torch.log(tmp_CEE1)#*choice_len_term
        CEE = torch.mean(tmp_CEE2)
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
        states = Detailed_State_data_2_Tensor(states,cuda=cuda,normalize=True)
        before_states = [list(cell.before_state) for cell in tmp]
        #print(before_states)
        before_states = torch.LongTensor(before_states)
        before_states = before_states.cuda() if cuda else before_states
        #Detailed_State_data_2_Tensor(before_states,cuda=cuda,normalize=True)
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
                                card.card_id+500) for card in player.hand]
    #hand_card_costs = [card.cost for card in player.hand]
    hand_card_costs = [card.cost/20 for card in player.hand]
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
    opponent_creature_location = f.get_creature_location()[opponent_num]
    opponent_mask =  [1 if i in opponent_creature_location else 0 for i in range(5)]
    able_to_evo = f.get_able_to_evo(player)
    able_to_evo = [1 if i in able_to_evo else 0 for i in range(5)] + opponent_mask#able_to_evo = [cell+1 for cell in able_to_evo]
    follower_card_ids = [f.card_location[player_num][i].card_id + 500
                         if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Creature" else 0 for i in range(5)] \
                        + [f.card_location[opponent_num][i].card_id + 500
                           if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Creature" else 0 for i in range(5)]
    follower_stats = [[f.card_location[player_num][i].power/20, f.card_location[player_num][i].get_current_toughness()/20,
                       1, int(f.card_location[player_num][i].can_attack_to_follower()), int(f.card_location[player_num][i].can_attack_to_player()),1]
                      if i < len(f.card_location[player_num]) and f.card_location[player_num][
        i].card_category == "Creature" else [0, 0, 0, 0, 0,0] for i in range(5)] \
                     + [[f.card_location[opponent_num][i].power/20, f.card_location[opponent_num][i].get_current_toughness()/20,
                         1, 1, 1,1]
                        if i < len(f.card_location[opponent_num]) and f.card_location[opponent_num][
        i].card_category == "Creature" else [0, 0, 0, 0, 0,0] for i in range(5)]
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
    able_to_play = [1 if i in able_to_play else 0 for i in range(9)]#[cell+1 for cell in able_to_play]
    #able_to_play = f.get_able_to_play(player)
    #able_to_play = [cell+1 for cell in able_to_play]
    able_to_attack = f.get_able_to_attack(player)
    able_to_attack = [1 if i in able_to_attack else 0 for i in range(5)] + opponent_mask
    able_to_creature_attack = f.get_able_to_creature_attack(player)
    able_to_creature_attack = [1 if i in able_to_creature_attack else 0 for i in range(5)] + opponent_mask
    #life_data = [player.life, opponent.life,len(player.hand),len(opponent.hand) ,f.current_turn[player_num]]
    life_data = [player.life/20, opponent.life/20, len(player.hand)/10, len(opponent.hand)/10,f.current_turn[player_num]/10]
    #pp_data = [f.cost[player_num],f.remain_cost[player_num]]
    pp_data = [f.cost[player_num]/10, f.remain_cost[player_num]/10,f.cost[1-player_num]/10, f.remain_cost[1-player_num]/10]

    class_data = [player.deck.leader_class.value,
                    opponent.deck.leader_class.value]
    deck_type_data = [player.deck.deck_type,opponent.deck.deck_type]
    life_data = (life_data, class_data,deck_type_data)
#     datas = Detailed_State_data(hand_ids, hand_card_costs, follower_card_ids, amulet_card_ids,
#                               follower_stats, follower_abilities, able_to_evo, life_data, pp_data,
#                               able_to_play, able_to_attack, able_to_creature_attack,deck_data)
    datas = {"hand_ids":hand_ids, 
             "hand_card_costs":hand_card_costs, 
             "follower_card_ids":follower_card_ids, 
             "amulet_card_ids":amulet_card_ids,
             "follower_stats":follower_stats, 
             "follower_abilities":follower_abilities, 
             "able_to_evo":able_to_evo, 
             "life_data":life_data, 
             "pp_data":pp_data,
             "able_to_play":able_to_play, 
             "able_to_attack":able_to_attack,
             "able_to_creature_attack":able_to_creature_attack,
             "deck_data":deck_data}

    return datas



def Detailed_State_data_2_Tensor(datas,cuda=False, normalize=False):
    data_len = len(datas)
    #print(type(datas))
    #print(type(datas[0]))
    #hand_ids = torch.LongTensor([[0 for _ in range(9)] for _ in range(data_len)])
    hand_ids = torch.LongTensor([[datas[i]["hand_ids"][j] for j in range(9)] for i in range(data_len)])
    #hand_card_costs = torch.Tensor([[0 for j in range(9)] for i in range(data_len)])
    hand_card_costs = torch.Tensor([[datas[i]["hand_card_costs"][j] for j in range(9)] for i in range(data_len)])
    follower_card_ids = torch.LongTensor([[datas[i]["follower_card_ids"][j] for j in range(10)] for i in range(data_len)])
    #follower_card_ids = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    amulet_card_ids = torch.LongTensor([[datas[i]["amulet_card_ids"][j] for j in range(10)] for i in range(data_len)])
    #amulet_card_ids = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    follower_abilities = torch.LongTensor([[[datas[i]["follower_abilities"][j][k] if k < len(
        datas[i]["follower_abilities"][j]) else 0 for k in range(15)] for j in range(10)] for i in range(data_len)])
    follower_stats = torch.Tensor([[datas[i]["follower_stats"][j] for j in range(10)] for i in range(data_len)])
    #follower_stats = torch.Tensor([[(0,0) for _ in range(10)] for _ in range(data_len)])
    #able_to_evo = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    
    able_to_evo = torch.Tensor(
        [datas[i]["able_to_evo"] for i in range(data_len)])
    #able_to_evo = torch.LongTensor(
    #    [[datas[i].able_to_evo[j] if j < len(datas[i].able_to_evo) else 0 for j in range(10)] for i in range(data_len)])
    
    able_to_play = torch.Tensor(
        [datas[i]["able_to_play"] for i in
         range(data_len)])
    #able_to_play = torch.LongTensor(
    #    [[datas[i].able_to_play[j] if j < len(datas[i].able_to_play) else 0 for j in range(9)] for i in
    #     range(data_len)])
    able_to_attack = torch.Tensor(
        [datas[i]["able_to_attack"] for i in
         range(data_len)])
    able_to_creature_attack = torch.Tensor(
        [datas[i]["able_to_creature_attack"] for
         i in range(data_len)])
    #able_to_play = torch.LongTensor([[0 for _ in range(9)] for _ in range(data_len)])
    #able_to_attack = torch.LongTensor(
    #    [[1 if j < len(datas[i].able_to_attack) else 0 for j in range(5)] for i in
    #     range(data_len)])
    #able_to_creature_attack = torch.LongTensor(
    #    [[1 if j < len(datas[i].able_to_creature_attack)  else 0 for j in range(5)] for
    #     i in range(data_len)])

    pp_datas = torch.Tensor([datas[i]["pp_data"] for i in range(data_len)])
    life_datas = torch.Tensor([datas[i]["life_data"][0] for i in range(data_len)])
    class_datas = torch.LongTensor([datas[i]["life_data"][1] for i in range(data_len)])
    deck_type_datas = torch.LongTensor([datas[i]["life_data"][2] for i in range(data_len)])
    deck_datas = torch.LongTensor([datas[i]["deck_data"] for i in range(data_len)])
    #if normalize:
    #   normalized_tensors =[hand_card_costs,follower_stats,pp_datas,life_datas]
    #    for tensor in normalized_tensors:
    #       mean = torch.mean(tensor)
    #        std = torch.std(tensor)
    #        tensor = (tensor-mean)/std

    ans = {'values': {'life_datas': life_datas,
                      'class_datas': class_datas,
                      'deck_type_datas':deck_type_datas,
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
    tensor_acting_card_ids_in_action = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][1] for j in range(45)] for i in range(action_code_len)])
    tensor_acted_card_ids_in_action = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][2] for j in range(45)] for i in range(action_code_len)])
    tensor_acted_card_sides_in_action = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][3] for j in range(45)] for i in range(action_code_len)])
    able_to_choice = torch.Tensor([action_codes[i]['able_to_choice'] for i in range(action_code_len)])
    action_choice_len = torch.Tensor([[int(sum(action_codes[i]['able_to_choice']))] for i in range(action_code_len)])
    action_codes_dict = {'action_categories': tensor_action_categories,
                         'acting_card_ids': tensor_acting_card_ids_in_action,
                         'acted_card_ids': tensor_acted_card_ids_in_action,
                         'acted_card_sides': tensor_acted_card_sides_in_action,
                         'able_to_choice': able_to_choice,
                         'action_choice_len':action_choice_len}
    #
    # tensor_action_categories = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][0] for j in range(45)] for i in range(action_code_len)])
    # tensor_play_card_ids_in_action = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][1] for j in range(45)] for i in range(action_code_len)])
    # tensor_field_card_ids_in_action = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][2:5] for j in range(45)] for i in range(action_code_len)])
    # able_to_choice = torch.Tensor([action_codes[i]['able_to_choice'] for i in range(action_code_len)])
    # action_choice_len = torch.Tensor([[int(sum(action_codes[i]['able_to_choice']))] for i in range(action_code_len)])
    # action_codes_dict = {'action_categories': tensor_action_categories,
    #                      'play_card_ids': tensor_play_card_ids_in_action,
    #                      'field_card_ids': tensor_field_card_ids_in_action,
    #                      'able_to_choice': able_to_choice,
    #                      'action_choice_len':action_choice_len}
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


# +

def deck_id_2_deck_type(type_num):
    if type_num in [0,1]:
        return DeckType.AGGRO.value
    elif type_num in [5,6,7]:
        return DeckType.CONTROL.value
    elif type_num in [8,9,10,-9,11]:
        return DeckType.COMBO.value
    else:
        return DeckType.MID.value
        


# -

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
