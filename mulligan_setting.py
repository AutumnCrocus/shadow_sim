# +
import random
from abc import ABCMeta, abstractmethod
import numpy as np
from my_moduler import get_module_logger
from mulligan_setting import *
mylogger = get_module_logger(__name__)

class Mulligan_Policy:

    def __init__(self):
        self.data_use_flg=False

    __metaclass__ = ABCMeta

    @abstractmethod
    def decide(self,hand,deck):
        pass


# -

class Random_mulligan_policy(Mulligan_Policy):
    def decide(self,hand,deck):
        hand_id=[i for i in range(len(hand))]
        change_cards_id=sorted(random.sample(hand_id,random.randint(0,len(hand))))
        return change_cards_id

class Simple_mulligan_policy(Mulligan_Policy):
    def decide(self,hand,deck):
        change_cards_id=[]
        for i in range(len(hand)):
            if hand[i].cost>3:
                change_cards_id.append(i)

        return change_cards_id

class Min_cost_mulligan_policy(Mulligan_Policy):
    def decide(self,hand,deck):
        change_cards_id=[]
        cost_set=set()
        for i in range(len(deck.deck)):
            cost_set.add(deck.deck[i].cost)
        min_cost=min(cost_set)+1
        for i in range(len(hand)):
            if hand[i].cost > min_cost:
                change_cards_id.append(i)
        
        return change_cards_id


class Test_mulligan_policy(Mulligan_Policy):
    def __init__(self):
        self.data_use_flg=True
        self.win_data=[]
        self.mulligan_data=[]
        self.deck_type=None
        self.no_mulligan_flg=False
    def append_win_data(self,data):
        if self.no_mulligan_flg==False:
            self.win_data.append(data)
    def append_mulligan_data(self,data):
        if data!=[]:
            self.mulligan_data.append(data)
    def decide(self,hand,deck):
        change_cards_id=[]
        if self.win_data!=[]:
            exploit_data=[]
            for i,data in enumerate(self.mulligan_data):
                if self.win_data[i]==True and len(data)>0:
                    exploit_data.append(data)
            if len(exploit_data)==0:
                change_cards_id=self.random_policy(hand,deck)
                input_data=tuple([hand[card_id].origin_cost for card_id in change_cards_id])
                self.append_mulligan_data(input_data)
                return change_cards_id
            #mylogger.info("exploit_data:{}".format(exploit_data))
            cost_border=[sum(exploit_data[i])/len(exploit_data[i]) for i in range(len(exploit_data))]
            length=len(cost_border)
            if length==0:
                change_cards_id=self.random_policy(hand,deck)
                input_data=tuple([hand[card_id].origin_cost for card_id in change_cards_id])
                self.append_mulligan_data(input_data)
                return change_cards_id

            cost_border=sum(cost_border)/length 

            for i in range(len(hand)):
                if hand[i].cost>=cost_border:
                    change_cards_id.append(i)            

            input_data=tuple([hand[card_id].origin_cost for card_id in change_cards_id])
            self.append_mulligan_data(input_data)
            return change_cards_id

        hand_id=[i for i in range(len(hand))]
        change_cards_id=sorted(random.sample(hand_id,random.randint(0,len(hand))))
        input_data=tuple([hand[card_id].origin_cost for card_id in change_cards_id])
        self.append_mulligan_data(input_data)
        return change_cards_id
    
    def random_policy(self,hand,deck):
        hand_id=[i for i in range(len(hand))]
        change_cards_id=sorted(random.sample(hand_id,random.randint(0,len(hand))))
        return change_cards_id

# +
class Deep_mulligan_policy(Mulligan_Policy):
    def __init__(self,node_num=10):
        self.data_use_flg=False
        self.mulligan_net=Mulligan_Net(node_num=node_num)
        
    def decide(self,hand,deck):
        change_cards_id=[]
        cost_set=set()
        for i in range(len(deck.deck)):
            cost_set.add(deck.deck[i].cost)
        min_cost=min(cost_set)+1
        for i in range(len(hand)):
            if hand[i].cost > min_cost:
                change_cards_id.append(i)
        
        return change_cards_id
    def get_distribution(hand,deck):
        mulligan_state=[get_mulligan_data(hand,deck)]
        input_data=mulligan_data_2_tensor(mulligan_state)
        output_data = self.mulligan_net(input_data)
        



# -


import torch
import torch.nn as nn


# +

class Mulligan_Net(nn.Module):
    def __init__(self, node_num=10):
        super(Mulligan_Net, self).__init__()
        self.emb_layer = nn.Embedding(3000,node_num,padding_idx=0)
        origin=400
        self.origin=origin
        hidden_layer_num=10
        node_shrink_range=(origin-node_num)// hidden_layer_num

        self.hidden_layer_num=hidden_layer_num
        
        node_size_list=[origin-i*node_shrink_range for i in range(hidden_layer_num)]+[node_num]
        print(node_size_list)
        layer = [nn.Linear(node_size_list[i], node_size_list[i+1]) for i in range(hidden_layer_num)]
        self.hidden_layer = nn.ModuleList(layer)
        self.output_layer = nn.Linear(node_num,3)
        

    def forward(self, data):
        hand_ids=data['hand_ids']
        deck_data=data['deck_data']
        x1 = self.emb_layer(hand_ids)
        x2 = self.emb_layer(deck_data)
        x=torch.cat([x1,x2],dim=1).view(-1,self.origin)
        for i in range(self.hidden_layer_num):
            x = torch.relu(self.hidden_layer[i](x))
        x = torch.sigmoid(self.output_layer(x))
        return x
# -

from my_enum import *
def get_mulligan_data(hand,deck):
    hand_ids = [((Card_Category[card.card_category].value-1)*1000+
                                card.card_id+500) for card in hand]
    deck_data = sorted([((Card_Category[card.card_category].value-1)*1000+
                                card.card_id+500) for card in deck.deck])
    #deck_data.extend([0]*(40-len(deck.deck)))
    return (hand_ids,deck_data)



def mulligan_data_2_tensor(data,cuda=False):
        data_len = len(data)
        hand_ids = torch.LongTensor([[data[i][0][j] for j in range(3)] for i in range(data_len)])
        deck_datas = torch.LongTensor([data[i][1] for i in range(data_len)])
        ans = {
           'hand_ids': hand_ids,
           'deck_data':deck_datas}
        if cuda:
            ans['hand_ids'] = ans['hand_ids'].cuda()
            ans['deck_data'] = ans['deck_data'].cuda()
        return ans


"""
from card_setting import *
hand=[Creature(9),Creature(39),Creature(29)]

# +
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
import csv
def tsv_to_deck(tsv_name):
    deck = Deck()
    with open("Deck_TSV/" + tsv_name) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            card_category = row[0]
            # mylogger.info(row)
            if card_category == "Creature":
                deck.append(Creature(creature_name_to_id[row[1]]), num=int(row[2]))
            elif card_category == "Spell":
                deck.append(Spell(spell_name_to_id[row[1]]), num=int(row[2]))
            elif card_category == "Amulet":
                deck.append(Amulet(amulet_name_to_id[row[1]]), num=int(row[2]))
            else:
                assert False, "{} {}".format(card_category)
    return deck
d1 = tsv_to_deck(key_2_tsv_name[100][0])
d1.shuffle()
d1.deck.pop()
d1.deck.pop()
d1.deck.pop()
len(d1.deck)

# +

mulligan_state=[get_mulligan_data(hand,d1)]
input_data=mulligan_data_2_tensor(mulligan_state)
# -

mulligan_net = Mulligan_Net(10)

mulligan_net(input_data)
"""


