from abc import ABCMeta, abstractmethod
import operator
import random
import copy 
import math
import numpy as np
#import networkx as nx
import Field_setting
#import matplotlib.pyplot as plt
import random
from my_moduler import get_module_logger
#import tensorflow as tmp_field_list
import os.path
from card_setting import *
mylogger = get_module_logger(__name__)
max_life_value=math.exp(-1)-math.exp(-20)
#import numba
from my_enum import *
class Policy:

    __metaclass__ = ABCMeta

    @abstractmethod
    def decide(self,player,opponent,field):
        pass
    @abstractmethod
    def __str__(self):
        pass



class RandomPolicy(Policy):
    def __defaults__(self):
        return None
    def __str__(self):
        return 'RandomPolicy'
    def __init__(self):
        self.policy_type=0
    def decide(self,player,opponent,field):
        (ward_list,_,can_be_attacked,regal_targets)=field.get_situation(player,opponent)
        (can_play,can_attack,can_evo),(able_to_play,able_to_attack,able_to_creature_attack,able_to_evo)\
            =field.get_flag_and_choices(player,opponent,regal_targets)
        target_id=0
        """
        can_attack=True
        can_play=True
        if len(able_to_play)==0:
            can_play=False
        if len(able_to_creature_attack)==0:
            can_attack=False
        """
        length=len(able_to_play+able_to_attack)+1

        depth=1-len(able_to_evo)
        if can_evo==False:
            depth=0
        tmp=random.randint(depth,length)

        if tmp < 1 and can_evo==True:
            card_id=random.choice(able_to_evo)
            card=field.card_location[player.player_num][card_id]
            target_id=None
            if card.evo_target!=None:
                choices=field.get_regal_targets(card,player_num=player.player_num)
                if choices!=[]:
                    target_id=random.choice(choices)
            return -1,card_id,target_id#進化

        if tmp == 1 or (can_play==False and can_attack==False):
            return 0,0,0#ターン終了

        if tmp > 1 and tmp <= len(able_to_play)+1 and can_play==True:
            
            card_id=random.choice(able_to_play)
            if player.hand[card_id].have_target==0:
                return 1,card_id,0
            else:
                target_id=None
                if regal_targets[card_id]!=[]:
                    target_id=random.choice(regal_targets[card_id])
                return 1,card_id,target_id#カードのプレイ

        elif tmp > len(able_to_play)+1 and len(able_to_creature_attack)>0:
            card_id=random.choice(able_to_creature_attack)
            if ward_list!=[]:
                target_id=random.choice(ward_list)
                return 2,card_id,target_id#フォロワーへの攻撃

            else:
                if len(can_be_attacked)>0:
                    target_id = random.choice(can_be_attacked)
                    return 2,card_id,target_id#フォロワーへの攻撃

                elif card_id in able_to_attack:
                    return 3,card_id,None#プレイヤーへの攻撃


        return 0,0,0#ターン終了



class AggroPolicy(Policy):
    def __str__(self):
        return 'AgrroPolicy'
    
    def __init__(self):
        self.policy_type=1
    def decide(self,player,opponent,field):
        (ward_list,_,can_be_attacked,regal_targets)=field.get_situation(player,opponent)
        (can_play,can_attack,can_evo),(able_to_play,able_to_attack,able_to_creature_attack,able_to_evo)\
            =field.get_flag_and_choices(player,opponent,regal_targets)

        can_attack=True
        can_play=True
        end_flg=False

        if len(able_to_play)==0:
            can_play=False
        if len(able_to_creature_attack)==0:
            can_attack=False

        if can_play == False and can_attack == False and can_evo==False:
            return 0,0,0#ターン終了
        if can_play==True:
            #card_id=random.choice(able_to_play)
            able_to_play_cost = [player.hand[i].cost for i in able_to_play] 
            card_id = able_to_play[able_to_play_cost.index(max(able_to_play_cost))]            
            if player.hand[card_id].have_target==0:
                return 1,card_id,0
            else:
                target_id=None
                if regal_targets[card_id]!=[]:
                    target_id=random.choice(regal_targets[card_id])
                return 1,card_id,target_id

        if can_evo==True:
            can_evolve_power=[field.card_location[player.player_num][i].power for i in able_to_evo]
            max_index=can_evolve_power.index(max(can_evolve_power))
            card_id=able_to_evo[max_index]
            target_id=None
            card=field.card_location[player.player_num][card_id]
            if card.evo_target!=None:
                choices=field.get_regal_targets(card,player_num=player.player_num)
                if choices!=[]:
                    target_id=random.choice(choices)
            return -1,card_id,target_id#進化

        
        if can_attack==True:  
            opponent_creatures_stats=[]
            able_to_creature_attack_power = [field.card_location[player.player_num][i].power for i in able_to_creature_attack]
            card_id = able_to_creature_attack[able_to_creature_attack_power.index(max(able_to_creature_attack_power))]
            attack_creature=field.card_location[player.player_num][card_id]
            target_id = None
            creature_attack_flg=True
            if ward_list!=[]:
                ward_creatures_stats = [(field.card_location[opponent.player_num][ward_list[i]].power,\
                    field.card_location[opponent.player_num][ward_list[i]].get_current_toughness())
                        for i in range(len(ward_list))]
                opponent_creatures_stats=ward_creatures_stats

                for i,ele in enumerate(ward_creatures_stats):
                    if ele[1] <= sum(able_to_creature_attack_power):
                        target_id = ward_list[i]
                        break

            else:
                able_to_attack_power = [field.card_location[player.player_num][i].power for i in able_to_attack]
                if (len(field.get_creature_location()[opponent.player_num]) > 0 or opponent.life - sum(able_to_attack_power) > 0):

                    opponent_creatures_stats = [(field.card_location[opponent.player_num][i].power,\
                        field.card_location[opponent.player_num][i].get_current_toughness())\
                        for i in can_be_attacked]
                    leader_attack_flg=True
                    for i in range(len(can_be_attacked)):
                        if opponent_creatures_stats[i][1] <= attack_creature.power:
                            if opponent_creatures_stats[i][0] < attack_creature.get_current_toughness():

                                target_id = can_be_attacked[i]
                                leader_attack_flg=False
                                break
                            
                    if leader_attack_flg==True and attack_creature.can_attack_to_player():
                        creature_attack_flg=False

                elif attack_creature.can_attack_to_player():
                    creature_attack_flg=False
                    
            if creature_attack_flg==False:
                if attack_creature.player_attack_regulation==None or attack_creature.player_attack_regulation(field,player)==True:
                    return 3,card_id,0#プレイヤーへの攻撃   
                


            if len(opponent_creatures_stats)>0 and target_id!=None:
                return 2,card_id,target_id#クリーチャーへの攻撃
                
                


        return 0,0,0#ターン終了

class GreedyPolicy(Policy):
    def __str__(self):
        return 'GreedyPolicy'
    def __init__(self):
        self.policy_type=2
    def state_value(self,field,player_num):
        if field.players[1-player_num].life<=0:
            return 100000
        return (field.players[1-player_num].max_life-field.players[1-player_num].life)*100 + \
            (len(field.get_creature_location()[player_num])-len(field.get_creature_location()[1-player_num]))*5


    def decide(self,player,opponent,field):
        (ward_list,can_be_targeted,can_be_attacked,regal_targets)=field.get_situation(player,opponent)
        (can_play,can_attack,can_evo),(able_to_play,able_to_attack,able_to_creature_attack,able_to_evo)\
            =field.get_flag_and_choices(player,opponent,regal_targets)
        first=player.player_num
        dicision=[0,0,0]
        max_state_value=-10000
        can_play,can_attack=True,True
        if len(able_to_play)==0:
            can_play=False
        if len(able_to_creature_attack)==0:
            can_attack=False
        length=len(able_to_play+able_to_creature_attack)+1


        #tmp_field_list=[copy.deepcopy(field) for i in range(length)]
        tmp_field_list=[]
        for i in range(length):
            new_field=Field(5)
            new_field.set_data(field)
            tmp_field_list.append(new_field)
        state_value_list=[0 for i in range(length)]#各行動後の状態価値のリスト
        state_value_list[0]=self.state_value(tmp_field_list[0],first)

        target_creatures_toughness=[]
        if len(can_be_targeted) > 0:
            target_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness()\
                 for i in can_be_targeted]

        if can_evo==True and able_to_play==[]:

            leader_flg=False
            can_evolve_power=[field.card_location[player.player_num][i].power for i in able_to_evo]
            evo_id=can_evolve_power.index(max(can_evolve_power))
            direct_index=able_to_evo[0]
            for i,ele in enumerate(able_to_evo):
                if field.card_location[player.player_num][ele].is_tapped==False:
                    leader_flg=True
                    direct_index=i

                    break
            if len(field.get_creature_location()[opponent.player_num])==0 and leader_flg:
                evo_id=direct_index

            target_id=None
            creature=field.card_location[first][able_to_evo[evo_id]]
            if creature.evo_target!=None:
                choices=field.get_regal_targets(creature,player_num=player.player_num)
                if choices!=[]:
                    target_id=random.choice(choices)
            #evo_field=copy.deepcopy(field)
            evo_field = Field_setting.Field(5)
            evo_field.set_data(field)
            evo_field.players[first].creature_evolve(evo_field.card_location[first][able_to_evo[evo_id]],evo_field,virtual=True,target=target_id)
            evo_field.solve_field_trigger_ability(virtual=True,player_num=player.player_num)
            if evo_field.stack!=[]:
                evo_field.solve_lastword_ability(virtual=True,player_num=player.player_num)
            tmp_state_value=self.state_value(evo_field,first)
            if max_state_value<tmp_state_value:
                max_state_value=tmp_state_value
                dicision=[-1,able_to_evo[evo_id],target_id]


 
        if can_play==True:
            for i in range(1,len(able_to_play)+1):
                target_id=None
                card=player.hand[able_to_play[i-1]]
                if card.card_category!="Spell" and len(field.card_location[player.player_num])==field.max_field_num:
                    continue
                if card.have_target!=0:
                    choices=field.get_regal_targets(card,target_type=1,player_num=player.player_num)
                    if choices!=[]:
                        target_id=random.choice(choices)
                    elif card.card_category=="Spell":
                        raise Exception("target_category:{} name:{}".format(card.have_target,card.name))
                tmp_field_list[i].players[first].play_card(tmp_field_list[i],able_to_play[i-1],tmp_field_list[i].players[first],\
                    tmp_field_list[i].players[1-first],virtual=True,target=target_id)
                tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                if tmp_field_list[i].stack!=[]:
                    tmp_field_list[i].solve_lastword_ability(virtual=True,player_num=player.player_num)
                state_value_list[i]=self.state_value(tmp_field_list[i],first)
                if max_state_value < state_value_list[i] and tmp_field_list[i].players[first].life>0:
                    max_state_value=state_value_list[i]
                    dicision=[1,able_to_play[i-1],target_id]


        opponent_creatures_toughness=[]
        if len(can_be_attacked) > 0:
            opponent_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness()\
                 for i in can_be_attacked]
        
        if can_attack==True:
            if ward_list!=[]:
                ward_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness()\
                    for i in ward_list]
                able_to_creature_attack_power=[field.card_location[player.player_num][i].power for i in able_to_creature_attack]
                for i in range(len(able_to_play)+1,length):
                    target_id=None
                    attacker_id=able_to_creature_attack[i-len(able_to_play)+1-2]
                    attacker_power=field.card_location[first][attacker_id].power
                    if min(ward_creatures_toughness)<=sum(able_to_creature_attack_power):
                        target_id=ward_list[ward_creatures_toughness.index(min(ward_creatures_toughness))]
                        tmp_field_list[i].players[first].attack_to_creature(tmp_field_list[i],attacker_id,target_id,virtual=True)
                        tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                        if tmp_field_list[i].stack!=[]:
                            tmp_field_list[i].solve_lastword_ability(virtual=True,player_num=player.player_num)
                        state_value_list[i]+=self.state_value(tmp_field_list[i],first)
                        if max_state_value < state_value_list[i] and target_id!=None:
                            max_state_value=state_value_list[i]
                            #mylogger.info("target_id:{},name:{}".format(target_id,field.card_location[opponent.player_num][target_id].name))
                            dicision=[2,attacker_id,target_id]
            else:
                for i in range(len(able_to_play)+1,length):
                    direct_flg=False
                    target_id=None
                    attacker_id=able_to_creature_attack[i-len(able_to_play)+1-2]
                    attacker_power=field.card_location[first][attacker_id].power

                    if (len(opponent_creatures_toughness)==0 or min(opponent_creatures_toughness)>attacker_power\
                        ) and field.card_location[player.player_num][attacker_id].is_tapped==False:
                        return 3,attacker_id,None


                    elif (opponent_creatures_toughness)!=[] and min(opponent_creatures_toughness)<=attacker_power:
                        target_id=can_be_attacked[opponent_creatures_toughness.index(min(opponent_creatures_toughness))]
                        if field.card_location[opponent.player_num][target_id]==None:
                            raise Exception("No exist target!")
                        tmp_field_list[i].players[first].attack_to_creature(tmp_field_list[i],attacker_id,target_id,virtual=True)
                        tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                        if tmp_field_list[i].stack!=[]:
                            tmp_field_list[i].solve_lastword_ability(virtual=True,player_num=player.player_num)
                        state_value_list[i]+=self.state_value(tmp_field_list[i],first)

                    elif field.card_location[player.player_num][attacker_id].is_tapped==False:
                        direct_flg=True
                        return 3,attacker_id,None
                 

                    
                    if tmp_field_list[i].players[1-first].life<=0:
                        state_value_list[i]=10000
                    
                    if max_state_value < state_value_list[i]:
                        if direct_flg==True and field.card_location[first][attacker_id].is_tapped==False:
                            max_state_value=state_value_list[i]
                            dicision=[3,attacker_id,0]                    
                        if direct_flg==False and opponent_creatures_toughness!=[] and target_id!=None:
                            max_state_value=state_value_list[i]
                            #mylogger.info("target_id:{},name:{}".format(target_id,field.card_location[opponent.player_num][target_id].name))
                            dicision=[2,attacker_id,target_id]

        
        return dicision[0],dicision[1],dicision[2]

class FastGreedyPolicy(GreedyPolicy):
    def __init__(self):
        self.f=lambda x:int(x)+(1,0)[x-int(x)<0.5]

    def __str__(self):
        return 'FastGreedyPolicy(now freezed)'
    def __init__(self):
        self.policy_type=2

class Node:

    def __init__(self,field=None,player_num=0,finite_state_flg=False,depth=0):
        self.field=field
        self.finite_state_flg=finite_state_flg
        self.value=0.0
        self.visit_num=0
        self.is_root=False
        self.uct = 0.0
        self.depth=depth
        self.child_nodes=[]
        self.max_child_visit_num=[None,None]
        self.parent_node=None
        self.regal_targets={}
        #self.node_id=node_id
        #if self.field!=None:
        #    self.parent_node=None
        #else:
        #    raise Exception("Null field")
        children_moves=[]
        end_flg=field.check_game_end()
        if end_flg==False:
            (ward_list,_,can_be_attacked,regal_targets)=\
                field.get_situation(field.players[player_num],field.players[1-player_num])

            (_,_,can_evo),(able_to_play,able_to_attack,able_to_creature_attack,able_to_evo)=\
                field.get_flag_and_choices(field.players[player_num],field.players[1-player_num],regal_targets)
            self.regal_targets=regal_targets
            children_moves.append((0,0,0))
            #mylogger.info("able_to_play:{}".format(able_to_play))
            for play_id in able_to_play:
                if field.players[player_num].hand[play_id].card_category!="Spell" and len(field.card_location[player_num])>=field.max_field_num:
                    continue
                #if field.players[player_num].hand[play_id].have_target!=0 \
                #    and play_id in regal_targets and len(regal_targets[play_id])>0:
                if field.players[player_num].hand[play_id].have_target!=0 and len(regal_targets[play_id])>0:
                    for i in range(len(regal_targets[play_id])):
                        children_moves.append((1,play_id,regal_targets[play_id][i]))
                else:
                    if field.players[player_num].hand[play_id].card_category=="Spell":
                        if field.players[player_num].hand[play_id].have_target==0:
                            children_moves.append((1,play_id,None))
                    else:
                        children_moves.append((1,play_id,None))

            if ward_list==[]:
                for attacker_id in able_to_creature_attack:
                    for target_id in can_be_attacked:
                        children_moves.append((2,attacker_id,target_id))
                for attacker_id in able_to_attack:
                    children_moves.append((3,attacker_id,None))
            else:
                for attacker_id in able_to_creature_attack:
                    for target_id in ward_list:
                        children_moves.append((2,attacker_id,target_id))
            
            if can_evo:
                for evo_id in able_to_evo:
                    evo_creature=field.card_location[player_num][evo_id]
                    if evo_creature.evo_target==None:
                        children_moves.append((-1,evo_id,None))
                    else:
                        targets=field.get_regal_targets(evo_creature,target_type=0,player_num=player_num)
                        for target_id in targets:
                            children_moves.append((-1,evo_id,target_id))

            #if ward_list==[]:
            #    for attacker_id in able_to_attack:
            #        children_moves.append((3,attacker_id,None))

        self.children_moves=children_moves
 

 

    def print_tree(self,single=False):
        if self.is_root==True:
            print("ROOT(id:{})".format(id(self)))
        if self.parent_node!=None:
            print("parent_id:{})".format(id(self.parent_node)))
        print("depth:{} finite:{} mean_value:{} visit_num:{}".format(self.depth,self.finite_state_flg,int(self.value/max(1,self.visit_num)),self.visit_num))
        """
        if single==False:
            
            if self.child_nodes!=[]:
                print("child_node_num:{}".format(len(self.child_nodes)))
                print("child_node_set:{")
                for child in self.child_nodes:
                    print("action:{}".format(child[0]))
                    child[1].print_tree()
                print("}")
        """

    def get_exist_action(self):
        exist_action = []
        for i in range(len(self.child_nodes)):
            exist_action.append(self.child_nodes[i][0])
        
        return exist_action

    
    def get_able_action_list(self):
        return sorted(list(set(self.children_moves)-set(self.get_exist_action())))




EPSILON = 10e-6
ITERATION=100
WIN_BONUS=100000

class MCTSPolicy(Policy):
    def __init__(self):
        self.tree=None
        self.first_action_flg=False
        self.action_seq=[]
        self.initial_seq=[]
        self.num_simulations=0
        self.uct_c = 1. / np.sqrt(2)
        self.play_out_policy=RandomPolicy()
        self.end_count=0
        self.decide_node_seq=[]
        self.starting_node=None
        self.current_node=None
        self.node_index=0
        self.policy_type=3

    #import numba
    #@numba.jit
    def state_value(self,field,player_num):
        if field.players[1-player_num].life<=0:
            return WIN_BONUS
        power_sum=0
        for card_id in field.get_creature_location()[1-player_num]:
            power_sum+=field.card_location[1-player_num][card_id].power
        if power_sum>=field.players[player_num].life:
            return -WIN_BONUS


        return (field.players[1-player_num].max_life-field.players[1-player_num].life)*100 + \
            (len(field.get_creature_location()[player_num])-len(field.get_creature_location()[1-player_num]))*10 + len(field.players[player_num].hand)

    def decide(self,player,opponent,field):
        if self.action_seq==[]:
            
            action=self.uct_search(player,opponent,field)
            self.node_index=0
            #mylogger.info("Root")
            assert self.starting_node!=None and self.current_node !=None,"{},{}".format(self.starting_node,self.current_node)
            return action
        else:
            self.node_index+=1
            if self.initial_seq[-1]==(0,0,0):
                self.current_node=self.decide_node_seq[self.node_index]
            action=self.action_seq.pop(0)
            #mylogger.info("Blanch")
            assert self.starting_node!=None and self.current_node !=None,"{},{}".format(self.starting_node,self.current_node)
            return action

    def uct_search(self,player,opponent,field):
        field.get_regal_target_dict(player,opponent)
        player_num=player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field,player_num=player.player_num)
        mark_node=None
        starting_node.is_root=True
        self.starting_node=starting_node
        self.current_node=starting_node
        end_flg=False
        self.decide_node_seq=[]
        self.decide_node_seq.append(starting_node)
        if starting_node.get_able_action_list()==[(0,0,0)]:
            return 0,0,0
        for i in range(ITERATION):

            node = self.tree_policy(starting_node,player_num=player_num)
            if node.field.players[1-player_num].life<=0 and node.field.players[player_num].life>0:
                self.end_count+=1
                end_flg=True
                #node.print_tree()
                mark_node=node
                break
            value = self.default_policy(node,player_num=player_num)
            self.back_up(node,value,player_num=player_num)

            if starting_node.max_child_visit_num[0]!=None and starting_node.max_child_visit_num[1]!=None:
                if starting_node.max_child_visit_num[1].visit_num-starting_node.max_child_visit_num[0].visit_num>ITERATION-i:
                    break

            elif starting_node.max_child_visit_num[1]!=None:
                if starting_node.max_child_visit_num[1].visit_num>50:
                    break

        if end_flg==True:
            while True:
                if mark_node.parent_node==None:
                    break
                for child in mark_node.parent_node.child_nodes:
                    if child[1]==mark_node:
                        self.action_seq.append(child[0])
                        break
                mark_node=mark_node.parent_node
            self.action_seq=self.action_seq[::-1]
            self.initial_seq=self.action_seq[:]
        else:
            pointer=starting_node
            count=0
            length_of_children=len(pointer.child_nodes)
            while length_of_children>0:
                tmp_field=pointer.field
                tmp_player=pointer.field.players[player_num]
                next_node,tmp_move=self.best(pointer,player_num=player_num)
                self.action_seq.append(tmp_move)

                if tmp_move==(0,0,0):
                    break
                pointer=next_node
                self.decide_node_seq.append(pointer)
                length_of_children=len(pointer.child_nodes)

                count+=1
                if count>100:
                    raise Exception("Infinite loop")

        self.initial_seq=self.action_seq[:]
        move=self.action_seq.pop(0)

        assert self.starting_node!=None and self.current_node !=None,"{},{}".format(self.starting_node,self.current_node)
        return move

    def tree_policy(self,node,player_num=0):

        length_of_children=len(node.child_nodes)
        check=self.fully_expand(node,player_num=player_num)
        if length_of_children== 0 and check==False:
            return self.expand(node,player_num=player_num)
        count=0
        while node.finite_state_flg==False:
            if length_of_children>0:
                if random.uniform(0,1)<.5:
                    node, _ = self.best(node,player_num=player_num)
                else:
                    check=self.fully_expand(node,player_num=player_num)
                    if check == False:
                        return self.expand(node,player_num=player_num)
                    else:
                        node, _ = self.best(node,player_num=player_num)
                length_of_children=len(node.child_nodes)
            else:
                return self.expand(node,player_num=player_num)


            count+=1
            if count>100:
                field=node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                raise Exception("infinite loop!")

        return node
    
    def default_policy(self,node,player_num=0):
        if node.finite_state_flg==True:
            return self.state_value(node.field,player_num)
        sum_of_value=0
        end_flg=False
        for i in range(10):
            if node.finite_state_flg==False:
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)

                current_field.players[0].deck.shuffle()#デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])


                while True:
                    (action_num,card_id,target_id)=self.play_out_policy.decide(current_field.players[player_num],current_field.players[1-player_num],\
                        current_field)
 
                    end_flg=current_field.players[player_num].execute_action(current_field,current_field.players[1-player_num],\
                        action_code=(action_num,card_id,target_id),virtual=True)

                    if current_field.check_game_end()==True or end_flg==True:
                        break

                    current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                if current_field.check_game_end()==True:
                    sum_of_value += WIN_BONUS
                    return sum_of_value  
                current_field.end_of_turn(player_num,virtual=True)
                if current_field.check_game_end()==True:
                    sum_of_value += WIN_BONUS
                    return sum_of_value
                else:
                    sum_of_value += self.state_value(current_field,player_num)
            
        return sum_of_value/10

    def fully_expand(self,node,player_num=0):
        return len(node.child_nodes)==len(node.children_moves) or node.finite_state_flg==True#turn_endの場合を追加
    
    def expand(self,node,player_num=0):

        field=node.field

        new_choices = node.get_able_action_list()
        if new_choices==[]:
            field.show_field()

            mylogger.info("children_moves:{} exist_action:{}".format(node.children_moves,node.get_exist_action()))
            raise Exception()
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.players[0].deck.shuffle()
        next_field.players[1].deck.shuffle()
        next_field.get_situation(next_field.players[player_num],next_field.players[1-player_num])#regal_targetsの更新のため
        next_node=None
        if move[0]==0:
            next_node=Node(field=next_field,player_num=player_num,finite_state_flg=True,depth=node.depth+1)
        else:
            if move[0]==1 and move[2]==None and node.regal_targets[move[1]]!=[]:
                mylogger.info("in_node:{}".format(node.regal_targets[move[1]]))
                raise Exception("Null target!")

            next_field.players[player_num].execute_action(next_field,next_field.players[1-player_num],action_code=move,virtual=True)
            flg =(next_field.check_game_end()==True)
            """
            if move[0]==1 and field.players[player_num].hand[move[1]].name=="Fate's Hand":
                length=len(field.players[player_num].hand)
                new_length=len(next_field.players[player_num].hand)
                if length > new_length and next_field.players[player_num].lib_out_flg==False:
                    mylogger.info("No drraw error")
                    mylogger.info("Prev:")
                    field.players[player_num].show_hand()
                    field.players[player_num].deck.show_all()
                    mylogger.info("Next:")
                    next_field.players[player_num].show_hand()
                    next_field.players[player_num].deck.show_all()
                    raise Exception()
            """
            
            next_node=Node(field=next_field,player_num=player_num,\
                finite_state_flg= flg==True,depth=node.depth+1)
        next_node.parent_node=node
        node.child_nodes.append((move,next_node))
        return next_node

    def best(self,node,player_num=0):
        children = node.child_nodes
        uct_values = {}
        for i in range(len(children)):
            uct_values[i]=self.uct(children[i][1],node,player_num=player_num)
        
        uct_values_list=list(uct_values.values())
        max_uct_value=max(uct_values_list)
        max_list_index=uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action=children[max_list_index][0]

        return max_value_node,action



    def uct(self,child_node,node,player_num=0):
        over_all_n = node.visit_num
        n = child_node.visit_num
        w = child_node.value
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n+epsilon)
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / ( n + epsilon))

        value = exploitation_value+exploration_value

        child_node.uct = value

        return value


    def back_up(self,last_visited,reward,player_num=0):
        current=last_visited
        while True:
            current.visit_num += 1
            current.value += reward

            if current.is_root==True:
                break

            elif current.parent_node.is_root==True:
                best_childen=current.parent_node.max_child_visit_num
                best_node=best_childen[1]
                second_node=best_childen[0]
                if best_node==None:#ベストが空
                    best_node=current
                elif second_node==None:#ベストが1つのみ
                    if current!=best_node:
                        if best_node.visit_num<current.visit_num:
                            best_childen=[best_node,current]
                        else:
                            best_childen=[current,best_node]
                else:
                    if second_node.visit_num>best_node.visit_num:
                        best_children=[best_node,second_node]

                    different_flg= current not in best_children[0]
                    if different_flg==True:
                        if current.visit_num>best_children[1].visit_num:
                            best_children=[best_children[1],current]

                        elif current.visit_num>best_children[0].visit_num:
                           best_children=[current,best_children[1]]
                
            current = current.parent_node
            if current==None:
                break
    


    
    def __str__(self):
        return 'MCTSPolicy'

class Shallow_MCTSPolicy(MCTSPolicy):


    def tree_policy(self,node,player_num=0):
        length_of_children=len(node.child_nodes)
        if length_of_children== 0 and self.fully_expand(node,player_num=player_num)==False:
            return self.expand(node,player_num=player_num)
        count=0
        while node.finite_state_flg==False:
            if length_of_children>0:
                if random.uniform(0,1)<.5:
                    node, _ = self.best(node,player_num=player_num)
                else:
                    if self.fully_expand(node,player_num=player_num) == False and node.depth<1:#拡張は深さ1まで
                        return self.expand(node,player_num=player_num)
                    else:
                        node, _ = self.best(node,player_num=player_num)
                length_of_children=len(node.child_nodes)
            else:
                break


            count+=1
            if count>100:
                field=node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                raise Exception("infinite loop!")

        return node

    

class Test_MCTSPolicy(MCTSPolicy):

    def __init__(self):
        super().__init__()
        self.opponent_policy=RandomPolicy()
    
    def default_policy(self,node,player_num=0):
        if node.field.check_game_end()==True:
            return self.state_value(node.field,player_num)
        current_field = Field_setting.Field(5)       
        sum_of_value=0
        end_flg=False
        for i in range(10):
            current_field.set_data(node.field)
            current_field.players[0].deck.shuffle()#デッキの並びは不明だから
            current_field.players[1].deck.shuffle()
            if node.finite_state_flg==False:
                current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                while True:
                    (action_num,card_id,target_id)=self.play_out_policy.decide(current_field.players[player_num],current_field.players[1-player_num],\
                        current_field)
                    end_flg=current_field.players[player_num].execute_action(current_field,current_field.players[1-player_num],\
                        action_code=(action_num,card_id,target_id),virtual=True)

                    if current_field.check_game_end()==True or end_flg==True:
                        break
                    current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                if current_field.check_game_end()==True:
                    sum_of_value += 100000
                    continue
                current_field.end_of_turn(player_num,virtual=True)
                if current_field.check_game_end()==True:
                    sum_of_value += 100000
                    continue
            #相手ターンのシミュレーション
            current_field.untap(1-player_num)
            current_field.increment_cost(1-player_num)
            current_field.start_of_turn(1-player_num,virtual=True)
            current_field.players[1-player_num].mulligan(current_field.players[1-player_num].deck,virtual=True)
            current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
            while True:
                (action_num,card_id,target_id)=self.opponent_policy.decide(current_field.players[1-player_num],current_field.players[player_num],\
                    current_field)
                end_flg=current_field.players[1-player_num].execute_action(current_field,current_field.players[player_num],\
                    action_code=(action_num,card_id,target_id),virtual=True)
                if current_field.check_game_end()==True or end_flg==True:
                    break
                
                current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
            if current_field.check_game_end()==True:
                sum_of_value -= WIN_BONUS
            else:
                sum_of_value += self.state_value(current_field,player_num)

            
        return sum_of_value/10



class Test_2_MCTSPolicy(Test_MCTSPolicy):

    def __init__(self):
        super().__init__()
        self.opponent_policy=AggroPolicy()

    



class Test_3_MCTSPolicy(Test_MCTSPolicy):

    def __init__(self):
        super().__init__()
        self.opponent_policy=GreedyPolicy()



class Aggro_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        #RandomPolicyとAggroPolicyのPlayOutでの比較
        self.play_out_policy=AggroPolicy()



