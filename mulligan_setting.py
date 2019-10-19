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

"""
def Random_mulligan_policy(hand,deck):
     hand_id=[i for i in range(len(hand))]
    change_cards_id=sorted(random.sample(hand_id,random.randint(0,len(hand))))
    return change_cards_id



def Simple_mulligan_policy(hand,deck):
    change_cards_id=[]
    for i in range(len(hand)):
        if hand[i].cost>3:
            change_cards_id.append(i)

    return change_cards_id

def Min_cost_mulligan_policy(hand,deck):
    change_cards_id=[]
    cost_set=set()
    for i in range(len(deck.deck)):
        cost_set.add(deck.deck[i].cost)
    min_cost=min(cost_set)+1
    for i in range(len(hand)):
        if hand[i].cost > min_cost:
            change_cards_id.append(i)
    
    return change_cards_id

def Test_mulligan_policy(hand,deck):
    change_cards_id=[]
    for i in range(len(hand)):
        if hand[i].cost>3:
            change_cards_id.append(i)

    return change_cards_id 

"""

