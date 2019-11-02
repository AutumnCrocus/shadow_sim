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
        self.name="RandomPolicy"
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
        self.name="AggroPolicy"
    def decide(self,player,opponent,field):
        (ward_list,_,can_be_attacked,regal_targets)=field.get_situation(player,opponent)
        (can_play,can_attack,can_evo),(able_to_play,able_to_attack,able_to_creature_attack,able_to_evo)\
            =field.get_flag_and_choices(player,opponent,regal_targets)

        #can_attack=True
        #can_play=True
        #end_flg=False

        if len(able_to_play)==0:
            can_play=False
        if len(able_to_creature_attack)==0:
            can_attack=False

        #if can_play == False and can_attack == False and can_evo==False:
        #    return 0,0,0#ターン終了
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
                if attack_creature.player_attack_regulation==None or attack_creature.player_attack_regulation(player)==True:
                    return 3,card_id,0#プレイヤーへの攻撃   
                


            if len(opponent_creatures_stats)>0 and target_id!=None:
                return 2,card_id,target_id#クリーチャーへの攻撃
                
                


        return 0,0,0#ターン終了

class GreedyPolicy(Policy):
    def __str__(self):
        return 'GreedyPolicy'
    def __init__(self):
        self.policy_type=2
        self.name="GreedyPolicy"
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
        length=len(able_to_play+able_to_creature_attack)+1
        end_field_id_list=[]

        #tmp_field_list=[copy.deepcopy(field) for i in range(length)]
        #mylogger.info("len:{}".format(length))
        tmp_field_list=[]
        for i in range(length):
            new_field=Field_setting.Field(5)
            new_field.set_data(field)
            #new_field.get_flag_and_choices(new_field.players[player.player_num],new_field.players[opponent.player_num],regal_targets)
            tmp_field_list.append(new_field)
        state_value_list=[0 for i in range(length)]#各行動後の状態価値のリスト
        state_value_list[0]=self.state_value(tmp_field_list[0],first)

        target_creatures_toughness=[]
        if len(can_be_targeted) > 0:
            target_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness()\
                 for i in can_be_targeted]

        if can_evo==True and len(able_to_evo) > 0 and able_to_play==[]:

            leader_flg=False
            can_evolve_power=[field.card_location[player.player_num][i].power for i in able_to_evo]
            evo_id=able_to_evo[can_evolve_power.index(max(can_evolve_power))]
            
            direct_index=able_to_evo[0]
            for i,ele in enumerate(able_to_evo):
                if field.card_location[player.player_num][ele].can_attack_to_player():
                    leader_flg=True
                    direct_index=ele

                    break
            if len(field.get_creature_location()[opponent.player_num])==0 and leader_flg:
                evo_id=direct_index
            assert evo_id in able_to_evo
            target_id=None
            creature=field.card_location[first][evo_id]
            if creature.evo_target!=None:
                choices=field.get_regal_targets(creature,player_num=player.player_num)
                if choices!=[]:
                    target_id=random.choice(choices)
            evo_field = Field_setting.Field(5)
            evo_field.set_data(field)
            assert evo_field.get_able_to_evo(evo_field.players[first]) == field.get_able_to_evo(player)
            evo_field.players[first].execute_action(evo_field,evo_field.players[1-first],action_code=(Action_Code.EVOLVE.value,evo_id,target_id),virtual=True)
            #evo_field.players[first].creature_evolve(evo_field.card_location[first][able_to_evo[evo_id]],evo_field,virtual=True,target=target_id)
            #evo_field.solve_field_trigger_ability(virtual=True,player_num=player.player_num)
            #if evo_field.stack!=[]:
            #if len(evo_field.stack)>0:
            #    evo_field.solve_lastword_ability(virtual=True,player_num=player.player_num)
            #evo_field.ability_resolution(virtual=True,player_num=player.player_num)
            tmp_state_value=self.state_value(evo_field,first)
            if max_state_value<tmp_state_value:
                max_state_value=tmp_state_value
                dicision=[Action_Code.EVOLVE.value,evo_id,target_id]


 
        if can_play==True:
            #mylogger.info(range(1,len(able_to_play)+1))
            for i in range(1,len(able_to_play)+1):
                assert i not in  end_field_id_list,"{} {}".format(i,end_field_id_list)
                #mylogger.info("i(play):{}".format(i))
                target_id=None
                card_id=able_to_play[i-1]
                card=player.hand[card_id]
                if card.card_category!="Spell" and len(field.card_location[player.player_num])==field.max_field_num:
                    continue
                if card.have_target!=0:
                    choices=field.get_regal_targets(card,target_type=1,player_num=player.player_num)
                    if choices!=[]:
                        target_id=random.choice(choices)
                    elif card.card_category=="Spell":
                        raise Exception("target_category:{} name:{}".format(card.have_target,card.name))
                tmp_field_list[i].players[first].execute_action(tmp_field_list[i],tmp_field_list[i].players[1-first],\
                    action_code=(Action_Code.PLAY_CARD.value,card_id,target_id),virtual=True)
                #tmp_field_list[i].players[first].play_card(tmp_field_list[i],able_to_play[i-1],tmp_field_list[i].players[first],\
                #    tmp_field_list[i].players[1-first],virtual=True,target=target_id)
                #tmp_field_list[i].ability_resolution(virtual=True,player_num=player.player_num)
                #tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                #if tmp_field_list[i].stack!=[]:
                #if len(tmp_field_list[i].stack)>0:
                #    tmp_field_list[i].solve_lastword_ability(virtual=True,player_num=player.player_num)
                state_value_list[i]=self.state_value(tmp_field_list[i],first)
                end_field_id_list.append(i)
                if max_state_value < state_value_list[i] and tmp_field_list[i].players[first].life>0:
                    max_state_value=state_value_list[i]
                    dicision=[Action_Code.PLAY_CARD.value,card_id,target_id]


        opponent_creatures_toughness=[]
        if len(can_be_attacked) > 0:
            opponent_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness()\
                 for i in can_be_attacked]
        
        if can_attack==True:
            if ward_list!=[]:
                ward_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness()\
                    for i in ward_list]
                able_to_creature_attack_power=[field.card_location[player.player_num][i].power for i in able_to_creature_attack]
                #mylogger.info(range(len(able_to_play)+1,length))
                for i in range(len(able_to_play)+1,length):
                    assert i not in  end_field_id_list,"{} {}".format(i,end_field_id_list)
                    #mylogger.info("i(attack):{}".format(i))
                    target_id=None
                    attacker_id=able_to_creature_attack[i-len(able_to_play)+1-2]
                    attacker_power=tmp_field_list[i].card_location[first][attacker_id].power
                    if min(ward_creatures_toughness)<=sum(able_to_creature_attack_power):
                        target_id=ward_list[ward_creatures_toughness.index(min(ward_creatures_toughness))]
                        tmp_field_list[i].players[first].execute_action(tmp_field_list[i],tmp_field_list[i].players[1-first],\
                            action_code=(Action_Code.ATTACK_TO_FOLLOWER.value,attacker_id,target_id),virtual=True)
                        #tmp_field_list[i].players[first].attack_to_follower(tmp_field_list[i],attacker_id,target_id,virtual=True)
                        #tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                        #end_field_id_list.append(i)
                        #if tmp_field_list[i].stack!=[]:
                        #if len(tmp_field_list[i].stack)>0:
                        #    tmp_field_list[i].solve_lastword_ability(virtual=True,player_num=player.player_num)
                        state_value_list[i]+=self.state_value(tmp_field_list[i],first)
                        if max_state_value < state_value_list[i] and target_id!=None:
                            max_state_value=state_value_list[i]
                            #mylogger.info("target_id:{},name:{}".format(target_id,field.card_location[opponent.player_num][target_id].name))
                            dicision=[Action_Code.ATTACK_TO_FOLLOWER.value,attacker_id,target_id]
            else:
                for i in range(len(able_to_play)+1,length):
                    assert i not in  end_field_id_list,"{} {}".format(i,end_field_id_list)
                    direct_flg=False
                    target_id=None
                    attacker_id=able_to_creature_attack[i-(len(able_to_play)+1)]
                    attacking_creature=tmp_field_list[i].card_location[player.player_num][attacker_id]
                    assert attacking_creature.can_attack_to_follower()
                    
                    attacker_power=attacking_creature.power

                    if (len(opponent_creatures_toughness)==0 or min(opponent_creatures_toughness)>attacker_power\
                        ) and attacking_creature.can_attack_to_player():
                        if attacker_id in able_to_attack:
                            return Action_Code.ATTACK_TO_PLAYER.value,attacker_id,0


                    elif (opponent_creatures_toughness)!=[] and min(opponent_creatures_toughness)<=attacker_power:
                        target_id=can_be_attacked[opponent_creatures_toughness.index(min(opponent_creatures_toughness))]
                        defencing_creature=tmp_field_list[i].card_location[opponent.player_num][target_id]
                        assert defencing_creature.can_be_attacked()
                        tmp_field_list[i].players[first].execute_action(tmp_field_list[i],tmp_field_list[i].players[1-first],\
                            action_code=(Action_Code.ATTACK_TO_FOLLOWER.value,attacker_id,target_id),virtual=True)
                        #tmp_field_list[i].players[first].attack_to_follower(tmp_field_list[i],attacker_id,target_id,virtual=True)
                        #tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                        #if tmp_field_list[i].stack!=[]:
                        #if len(tmp_field_list[i].stack)>0:
                        #    tmp_field_list[i].solve_lastword_ability(virtual=True,player_num=player.player_num)
                        #    tmp_field_list[i].solve_field_trigger_ability(virtual=True,player_num=player.player_num)
                        state_value_list[i]+=self.state_value(tmp_field_list[i],first)
                        end_field_id_list.append(i)

                    elif attacking_creature.can_attack_to_player():
                        if attacker_id in able_to_attack:
                             
                            direct_flg=True
                            return Action_Code.ATTACK_TO_PLAYER.value,attacker_id,0
                 

                    
                    if tmp_field_list[i].players[1-first].life<=0:
                        state_value_list[i]=10000
                    
                    if max_state_value < state_value_list[i]:
                        if direct_flg==True and attacking_creature.can_attack_to_player():
                            if attacker_id in able_to_attack:
                                max_state_value=state_value_list[i]
                                dicision=[Action_Code.ATTACK_TO_PLAYER.value,attacker_id,0]                    
                        if direct_flg==False and opponent_creatures_toughness!=[] and target_id!=None:
                            max_state_value=state_value_list[i]
                            #mylogger.info("target_id:{},name:{}".format(target_id,field.card_location[opponent.player_num][target_id].name))
                            dicision=[Action_Code.ATTACK_TO_FOLLOWER.value,attacker_id,target_id]

        #mylogger.info("Interval")
        return dicision[0],dicision[1],dicision[2]

class FastGreedyPolicy(GreedyPolicy):
    def __init__(self):
        self.f=lambda x:int(x)+(1,0)[x-int(x)<0.5]
        self.name="FastGreedyPolicy"

    def __str__(self):
        return 'FastGreedyPolicy(now freezed)'
    def __init__(self):
        self.policy_type=2
class New_GreedyPolicy(GreedyPolicy):
    def state_value(self,field,player_num):
        if field.check_game_end():
            if field.players[1-player_num].life<=0 or len(field.players[1-player_num].deck.deck)==0:
                return 1.0
            return 0.0

        player = field.players[player_num]
        opponent = field.players[1-player_num]
        
        before_evo_turn = int(field.current_turn[player_num]<field.able_to_evo_turn[player_num])
        accumulated_damage_ratio = (opponent.max_life-opponent.life)/opponent.max_life
        assert self.probability_check_func(accumulated_damage_ratio) ,"{}".format(accumulated_damage_ratio)
        life_advantage = ((player.life - opponent.life)+20)/40
        assert self.probability_check_func(life_advantage),"{}".format(life_advantage)
        player_hand_len=len(player.hand)
        opponent_hand_len=len(opponent.hand)
        hand_advantage = (player_hand_len - opponent_hand_len+max(player_hand_len,opponent_hand_len))/(2*max(1,max(player_hand_len,opponent_hand_len)))
        assert self.probability_check_func(hand_advantage),"{}".format(hand_advantage)
        player_board_len=len(field.get_creature_location()[player_num])
        opponent_board_len=len(field.get_creature_location()[1-player_num])
        board_advantage = (player_board_len-opponent_board_len+max(player_board_len,opponent_board_len))/(2*max(1,max(player_board_len,opponent_board_len)))
        assert self.probability_check_func(board_advantage),"{}".format(board_advantage)
        tempo_advantage = (field.cost[player_num]-field.remain_cost[player_num])/field.cost[player_num]
        assert self.probability_check_func(tempo_advantage),"{}".format(tempo_advantage)
        value = \
            life_advantage*self.hyper_parameter[0]+\
                hand_advantage*self.hyper_parameter[1]+\
                    (1-before_evo_turn)*board_advantage*self.hyper_parameter[2]+before_evo_turn*board_advantage*self.hyper_parameter[3]+\
                        accumulated_damage_ratio*self.hyper_parameter[4]+\
                            tempo_advantage*self.hyper_parameter[5]
                            
        max_value = sum(self.hyper_parameter)-((1-before_evo_turn)*self.hyper_parameter[3]+before_evo_turn*self.hyper_parameter[2])
        min_value = 0


            
        value = max(value,0.001)
        assert (max_value-min_value)*self.eta>1,"{} {}".format(max_value,min_value)
        probability=np.log(max(1+EPSILON,(value-min_value)*self.eta))/np.log(max(1+EPSILON,(max_value-min_value)*self.eta))
        

        assert self.probability_check_func(probability),"probability:{}".format(probability)
        probability=max(0,probability)
        return probability
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
        self.action_value_dict={}
        self.current_probability=None
        self.children_moves=[(0,0,0)]
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
        #for action in self.children_moves:
        #    self.action_value_dict[action]=0
 

 

    def print_tree(self,single=False):
        if self.is_root==True:
            print("ROOT(id:{})".format(id(self)))
        #if self.parent_node!=None:
        #    print("   "*self.depth+("parent_id:{})".format(id(self.parent_node))))
        print("   "*self.depth+"depth:{} finite:{} mean_value:{} visit_num:{}".format(self.depth,self.finite_state_flg,int(self.value/max(1,self.visit_num)),self.visit_num))
        
        if single==False:
            
            if self.child_nodes!=[]:
                print("   "*self.depth+"child_node_num:{}".format(len(self.child_nodes)))
                print("   "*self.depth+"child_node_set:{")
                for child in self.child_nodes:
                    print("   "*self.depth+"action:{}".format(child[0]))
                    child[1].print_tree()
                print("   "*self.depth+"}")
        

    def get_exist_action(self):
        exist_action = []
        for i in range(len(self.child_nodes)):
            exist_action.append(self.child_nodes[i][0])
        
        return exist_action

    
    def get_able_action_list(self):
        return sorted(list(set(self.children_moves)-set(self.get_exist_action())))
    
    def print_estimated_action_value(self):
        if self.visit_num==0:
            print("this node is never simulated!")
            return
        print("visit_num:{} depth:{}".format(self.visit_num,self.depth))
        for key in list(self.action_value_dict.keys()):
            print("{}:{}".format(key,self.action_value_dict[key]/(self.visit_num+1)))
        
        #for cell in self.child_nodes:
        #    print("{}:{} times".format(cell[0],cell[1].visit_num))
        #print("")




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
        self.name="MCTSPolicy"

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

                action_count=0
                while True:
                    (action_num,card_id,target_id)=self.play_out_policy.decide(current_field.players[player_num],current_field.players[1-player_num],\
                        current_field)
 
                    end_flg=current_field.players[player_num].execute_action(current_field,current_field.players[1-player_num],\
                        action_code=(action_num,card_id,target_id),virtual=True)

                    if current_field.check_game_end()==True or end_flg==True:
                        break

                    current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                    action_count+=1
                    if action_count>100:
                        current_field.players[player_num].show_hand()
                        current_field.show_field()
                        assert False
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
             
        sum_of_value=0
        end_flg=False
        current_field = Field_setting.Field(5)  
        for i in range(10):
            
            current_field.set_data(node.field)
            current_field.players[0].deck.shuffle()#デッキの並びは不明だから
            current_field.players[1].deck.shuffle()
            if node.finite_state_flg==False:
                
                while True:
                    current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                    (action_num,card_id,target_id)=self.play_out_policy.decide(current_field.players[player_num],current_field.players[1-player_num],\
                        current_field)
                    end_flg=current_field.players[player_num].execute_action(current_field,current_field.players[1-player_num],\
                        action_code=(action_num,card_id,target_id),virtual=True)

                    if current_field.check_game_end()==True or end_flg==True:
                        break
                if current_field.check_game_end()==True:
                    sum_of_value += WIN_BONUS
                    continue
                current_field.end_of_turn(player_num,virtual=True)
                if current_field.check_game_end()==True:
                    sum_of_value += WIN_BONUS
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
        self.name="Aggro_MCTSPolicy"
        #RandomPolicyとAggroPolicyのPlayOutでの比較
        self.play_out_policy=AggroPolicy()

class New_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name="New_MCTSPolicy"
        self.hyper_parameter=[1,1,1,1,1,1]#[1/6]*6
        self.probability_check_func=lambda x:x>=0 and x<=1
        self.eta=100
    def state_value(self,field,player_num):
        if field.check_game_end():
            if field.players[1-player_num].life<=0 or len(field.players[1-player_num].deck.deck)==0:
                return 1.0
            return 0.0

        player = field.players[player_num]
        opponent = field.players[1-player_num]
        
        before_evo_turn = int(field.current_turn[player_num]<field.able_to_evo_turn[player_num])
        accumulated_damage_ratio = (opponent.max_life-opponent.life)/opponent.max_life
        assert self.probability_check_func(accumulated_damage_ratio) ,"{}".format(accumulated_damage_ratio)
        life_advantage = ((player.life - opponent.life)+20)/40
        assert self.probability_check_func(life_advantage),"{}".format(life_advantage)
        player_hand_len=len(player.hand)
        opponent_hand_len=len(opponent.hand)
        hand_advantage = (player_hand_len - opponent_hand_len+max(player_hand_len,opponent_hand_len))/(2*max(1,max(player_hand_len,opponent_hand_len)))
        assert self.probability_check_func(hand_advantage),"{}".format(hand_advantage)
        player_board_len=len(field.get_creature_location()[player_num])
        opponent_board_len=len(field.get_creature_location()[1-player_num])
        board_advantage = (player_board_len-opponent_board_len+max(player_board_len,opponent_board_len))/(2*max(1,max(player_board_len,opponent_board_len)))
        assert self.probability_check_func(board_advantage),"{}".format(board_advantage)
        tempo_advantage = (field.cost[player_num]-field.remain_cost[player_num])/field.cost[player_num]
        assert self.probability_check_func(tempo_advantage),"{}".format(tempo_advantage)
        value = \
            life_advantage*self.hyper_parameter[0]+\
                hand_advantage*self.hyper_parameter[1]+\
                    (1-before_evo_turn)*board_advantage*self.hyper_parameter[2]+before_evo_turn*board_advantage*self.hyper_parameter[3]+\
                        accumulated_damage_ratio*self.hyper_parameter[4]+\
                            tempo_advantage*self.hyper_parameter[5]
                            
        max_value = sum(self.hyper_parameter)-((1-before_evo_turn)*self.hyper_parameter[3]+before_evo_turn*self.hyper_parameter[2])
        min_value = 0


            
        value = max(value,0.001)
        assert (max_value-min_value)*self.eta>1,"{} {}".format(max_value,min_value)
        probability=np.log(max(1+EPSILON,(value-min_value)*self.eta))/np.log(max(1+EPSILON,(max_value-min_value)*self.eta))
        

        assert self.probability_check_func(probability),"probability:{}".format(probability)
        probability=max(0,probability)
        return probability

class New_Aggro_MCTSPolicy(Aggro_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name="New_Aggro_MCTSPolicy"
        self.hyper_parameter=[1,1,1,1,1,1]#[1/6]*6
        self.probability_check_func=lambda x:x>=0 and x<=1
        self.eta=100

    def state_value(self,field,player_num):
        if field.check_game_end():
            if field.players[1-player_num].life<=0 or len(field.players[1-player_num].deck.deck)==0:
                return 1.0
            return 0.0

        player = field.players[player_num]
        opponent = field.players[1-player_num]
        
        before_evo_turn = int(field.current_turn[player_num]<field.able_to_evo_turn[player_num])
        accumulated_damage_ratio = (opponent.max_life-opponent.life)/opponent.max_life
        assert self.probability_check_func(accumulated_damage_ratio) ,"{}".format(accumulated_damage_ratio)
        life_advantage = ((player.life - opponent.life)+20)/40
        assert self.probability_check_func(life_advantage),"{}".format(life_advantage)
        player_hand_len=len(player.hand)
        opponent_hand_len=len(opponent.hand)
        hand_advantage = (player_hand_len - opponent_hand_len+max(player_hand_len,opponent_hand_len))/(2*max(1,max(player_hand_len,opponent_hand_len)))
        assert self.probability_check_func(hand_advantage),"{}".format(hand_advantage)
        player_board_len=len(field.get_creature_location()[player_num])
        opponent_board_len=len(field.get_creature_location()[1-player_num])
        board_advantage = (player_board_len-opponent_board_len+max(player_board_len,opponent_board_len))/(2*max(1,max(player_board_len,opponent_board_len)))
        assert self.probability_check_func(board_advantage),"{}".format(board_advantage)
        tempo_advantage = (field.cost[player_num]-field.remain_cost[player_num])/field.cost[player_num]
        assert self.probability_check_func(tempo_advantage),"{}".format(tempo_advantage)
        value = \
            life_advantage*self.hyper_parameter[0]+\
                hand_advantage*self.hyper_parameter[1]+\
                    (1-before_evo_turn)*board_advantage*self.hyper_parameter[2]+before_evo_turn*board_advantage*self.hyper_parameter[3]+\
                        accumulated_damage_ratio*self.hyper_parameter[4]+\
                            tempo_advantage*self.hyper_parameter[5]
                            
        max_value = sum(self.hyper_parameter)-((1-before_evo_turn)*self.hyper_parameter[3]+before_evo_turn*self.hyper_parameter[2])
        min_value = 0


            
        value = max(value,0.001)
        assert (max_value-min_value)*self.eta>1,"{} {}".format(max_value,min_value)
        probability=np.log(max(1+EPSILON,(value-min_value)*self.eta))/np.log(max(1+EPSILON,(max_value-min_value)*self.eta))
        

        assert self.probability_check_func(probability),"probability:{}".format(probability)
        probability=max(0,probability)
        return probability

class EXP3_MCTSPolicy(Policy):
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
        self.next_node=None
        self.prev_node=None
        self.hyper_parameter=[1,1,1,1,1,1]#[1/6]*6
        self.eta=1
        self.probability_check_func=lambda x:x>=0 and x<=1
        self.name="EXP3_MCTSPolicy"
    def state_value(self,field,player_num):
        if field.check_game_end():
            if field.players[1-player_num].life<=0 or len(field.players[1-player_num].deck.deck)==0:
                return 1.0
            return 0.0
        player = field.players[player_num]
        opponent = field.players[1-player_num]
        
        before_evo_turn = int(field.current_turn[player_num]<field.able_to_evo_turn[player_num])
        accumulated_damage_ratio = (opponent.max_life-opponent.life)/opponent.max_life
        assert self.probability_check_func(accumulated_damage_ratio) ,"{}".format(accumulated_damage_ratio)
        life_advantage = ((player.life - opponent.life)+20)/40
        assert self.probability_check_func(life_advantage),"{}".format(life_advantage)
        player_hand_len=len(player.hand)
        opponent_hand_len=len(opponent.hand)
        hand_advantage = (player_hand_len - opponent_hand_len+max(player_hand_len,opponent_hand_len))/(2*max(1,max(player_hand_len,opponent_hand_len)))
        assert self.probability_check_func(hand_advantage),"{}".format(hand_advantage)
        player_board_len=len(field.get_creature_location()[player_num])
        opponent_board_len=len(field.get_creature_location()[1-player_num])
        board_advantage = (player_board_len-opponent_board_len+max(player_board_len,opponent_board_len))/(2*max(1,max(player_board_len,opponent_board_len)))
        assert self.probability_check_func(board_advantage),"{}".format(board_advantage)
        tempo_advantage = (field.cost[player_num]-field.remain_cost[player_num])/field.cost[player_num]
        assert self.probability_check_func(tempo_advantage),"{}".format(tempo_advantage)
        value = \
            life_advantage*self.hyper_parameter[0]+\
                hand_advantage*self.hyper_parameter[1]+\
                    (1-before_evo_turn)*board_advantage*self.hyper_parameter[2]+before_evo_turn*board_advantage*self.hyper_parameter[3]+\
                        accumulated_damage_ratio*self.hyper_parameter[4]+\
                            tempo_advantage*self.hyper_parameter[5]
        
        max_value = sum(self.hyper_parameter)-((1-before_evo_turn)*self.hyper_parameter[3]+before_evo_turn*self.hyper_parameter[2])
        
        min_value = 0


            
        value = max(value,0.001)
        assert (max_value-min_value)*self.eta>1,"{} {}".format(max_value,min_value)
        #probability=np.log(max(1+EPSILON,(value-min_value)*self.eta))/np.log(max(1+EPSILON,(max_value-min_value)*self.eta))
        probability=value/max_value
        

        assert self.probability_check_func(probability),"probability:{}".format(probability)
        probability=max(0,probability)
        #mylogger.info("value:{} max_value:{} probability:{}".format(value,max_value,probability))
        return probability
    def decide(self,player,opponent,field):
        if self.current_node==None:
            
            self.exp3_search(player,opponent,field)
            #self.current_node.print_tree()
            
            if len(self.current_node.child_nodes)==0:
                #mylogger.info("End")
                self.current_node=None
                return 0,0,0 
            else:
                #mylogger.info("roulette(0)")
                #mylogger.info("action_value:{}".format(self.current_node.action_value_dict))
                #self.current_node.print_estimated_action_value()
                #mylogger.info("distribution:{}".format(self.exp3(self.current_node)))
                next_node,action,_=self.roulette(self.current_node)
                self.next_node=next_node
                self.prev_node=self.current_node
                #self.current_node=next_node
                if action==(0,0,0):self.current_node=None
                return action

        else:
            self.current_node=self.next_node
            if self.current_node.finite_state_flg==True or len(self.current_node.children_moves)==1:
                self.current_node=None
                return 0,0,0
            elif len(self.current_node.child_nodes)==0:

                self.current_node=None
                self.exp3_search(player,opponent,field)
                #mylogger.info("action_value:{}".format(self.current_node.action_value_dict))
                #self.current_node.print_estimated_action_value()
                #mylogger.info("distribution:{}".format(self.exp3(self.current_node)))
                if self.current_node.get_able_action_list()==[(0,0,0)]:
                    self.current_node = None
                    return 0,0,0
                next_node,action,_=self.roulette(self.current_node)
                self.next_node=next_node
                self.prev_node=self.current_node
                #self.current_node=next_node
                if action==(0,0,0):self.current_node=None
                return action
            else:
                #mylogger.info("roulette(1)")
                #self.current_node.print_estimated_action_value()
                #mylogger.info("distribution:{}".format(self.exp3(self.current_node)))

                #mylogger.info("action_value:{}".format(self.current_node.action_value_dict))
                next_node,action,_=self.roulette(self.current_node)
                self.next_node=next_node
                self.prev_node=self.current_node
                if action==(0,0,0):
                    self.current_node=None
                return action

            
            
            

    def exp3_search(self,player,opponent,field):
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
            #mylogger.info("check:{}".format(self.current_node==self.starting_node))
            return
        for i in range(ITERATION):

            node,probability = self.tree_policy(starting_node,player_num=player_num)
            value = self.default_policy(node,probability,player_num=player_num)
            self.back_up(node,value,player_num=player_num)
            if starting_node.max_child_visit_num[0]!=None and starting_node.max_child_visit_num[1]!=None:
                if starting_node.max_child_visit_num[1].visit_num-starting_node.max_child_visit_num[0].visit_num>ITERATION-i:
                    break

            elif starting_node.max_child_visit_num[1]!=None:
                if starting_node.max_child_visit_num[1].visit_num>50:
                    break

        #if end_flg==True:
        #    while True:
        #        if mark_node.parent_node==None:
        #            break
        #        for child in mark_node.parent_node.child_nodes:
        #            if child[1]==mark_node:
        #                self.action_seq.append(child[0])
        #                break
        #        mark_node=mark_node.parent_node
        #    self.action_seq=self.action_seq[::-1]
        #    self.initial_seq=self.action_seq[:]
        assert self.starting_node!=None and self.current_node !=None,"{},{}".format(self.starting_node,self.current_node)
        #mylogger.info("check:{}".format(self.current_node==self.starting_node))
        assert len(self.current_node.field.card_location[0])==len(field.card_location[0])
        assert len(self.current_node.field.card_location[1])==len(field.card_location[1])
        assert len(self.current_node.field.players[player.player_num].hand)==len(field.players[player.player_num].hand)
        #mylogger.info("action_value:{}".format(self.current_node.action_value_dict))
        return #move

    def tree_policy(self,node,player_num=0):

        length_of_children=len(node.child_nodes)
        check=self.fully_expand(node,player_num=player_num)
        if length_of_children== 0 and check==False:
            return self.expand(node,player_num=player_num)
        count=0
        probability=0.01
        while node.finite_state_flg==False:
            if length_of_children>0:
                
                if random.uniform(0,1)<.5:
                    node,action,probability=self.roulette(node,player_num=player_num)
                    #node, _ = self.best(node,player_num=player_num)
                    #if node==None:
                    #    return select_expand(self,node,action,player_num=0)
                else:
                    check=self.fully_expand(node,player_num=player_num)
                    if check == False:
                        return self.expand(node,player_num=player_num)
                    else:
                        node,action,probability=self.roulette(node,player_num=player_num)
                        #node, _ = self.best(node,player_num=player_num)
                length_of_children=len(node.child_nodes)
            else:
                return self.expand(node,player_num=player_num)


            count+=1
            if count>100:
                field=node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                raise Exception("infinite loop!")

        return node,probability
    
    def default_policy(self,node,probability,player_num=0):
        if node.finite_state_flg==True:
            action=None
            for cell in node.parent_node.child_nodes:
                if cell[-1]==node:
                    action=cell[0]
            if node.field.check_game_end():
                #mylogger.info("action:{} depth:{} value:{} probability:{}".format(action,node.depth,self.state_value(node.field,player_num),probability))
                node.parent_node.action_value_dict[action]=(node.visit_num+1)*self.state_value(node.field,player_num)/probability
                #mylogger.info("action_value:{}".format(node.parent_node.action_value_dict[action]))
            else:
                node.parent_node.action_value_dict[action]=-1000*(node.parent_node.visit_num+1)*(1/node.depth)#0
            #mylogger.info("{}:{},visit_num:{} depth:{}".format(action,self.state_value(node.field,player_num),node.visit_num,node.depth))
            return self.state_value(node.field,player_num)
        sum_of_value=0
        end_flg=False
        for i in range(10):
            if node.finite_state_flg==False:
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)

                current_field.players[0].deck.shuffle()#デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                #current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])

                action_count=0
                while True:
                    current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                    (action_num,card_id,target_id)=self.play_out_policy.decide(current_field.players[player_num],current_field.players[1-player_num],\
                        current_field)
 
                    end_flg=current_field.players[player_num].execute_action(current_field,current_field.players[1-player_num],\
                        action_code=(action_num,card_id,target_id),virtual=True)

                    if current_field.check_game_end()==True or end_flg==True:
                        break

                    #current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                    action_count+=1
                    if action_count>100:
                        current_field.players[player_num].show_hand()
                        current_field.show_field()
                        assert False

                if current_field.check_game_end()==True:
                    sum_of_value += 1.0
                    action=None
                    for cell in node.parent_node.child_nodes:
                        if cell[-1]==node:
                            action=cell[0]
                    node.parent_node.action_value_dict[action]=self.state_value(current_field,player_num)*(node.visit_num+1)
                    return (sum_of_value/(i+1))
                current_field.end_of_turn(player_num,virtual=True)
                if current_field.check_game_end()==True:
                    action=None
                    for cell in node.parent_node.child_nodes:
                        if cell[-1]==node:
                            action=cell[0]
                    node.parent_node.action_value_dict[action]=self.state_value(current_field,player_num)*(node.visit_num+1)
                    sum_of_value += 1.0
                    return (sum_of_value/(i+1))
                else:
                    assert self.state_value(current_field,player_num)>0,"{},{}".format(self.state_value(current_field,player_num),current_field.check_game_end())
                    sum_of_value += self.state_value(current_field,player_num)
            else:
                assert False,"finite:True"
        result=sum_of_value/10
        if node.parent_node!=[]:
            action=None
            for cell in node.parent_node.child_nodes:
                if cell[-1]==node:
                    action=cell[0]
            assert action!=None,"child_nodes:{}".format(node.parent_node.child_nodes)
            if action not in node.parent_node.action_value_dict:
                node.parent_node.action_value_dict[action]=0.0
                #mylogger.info("append {} to action_value_dict:{}".format(action,node.parent_node.action_value_dict))
            assert result/probability!=0.0,"result:{} probability:{} sum_of_value:{}".format(result,probability,sum_of_value)
            node.parent_node.action_value_dict[action]+=result/probability
            #mylogger.info("{}:{},visit_num:{} depth:{}".format(action,result,node.visit_num,node.depth))
            #mylogger.info("now_value:{}".format(node.parent_node.action_value_dict[action]))
        return result

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
            next_field.end_of_turn(player_num,virtual=True)
            #if (0,0,0) not in node.action_value_dict:
            #node.action_value_dict[move]=self.state_value(next_field,player_num)
            next_node=Node(field=next_field,player_num=player_num,finite_state_flg=True,depth=node.depth+1)
        else:
            if move[0]==1 and move[2]==None and node.regal_targets[move[1]]!=[]:
                mylogger.info("in_node:{}".format(node.regal_targets[move[1]]))
                raise Exception("Null target!")

            next_field.players[player_num].execute_action(next_field,next_field.players[1-player_num],action_code=move,virtual=True)
            flg =(next_field.check_game_end()==True)
            if flg==True:
                #node.action_value_dict[move]=self.state_value(next_field,player_num)
                assert any(self.state_value(next_field,player_num)==i for i in [0,1]),"{}".format((self.state_value(next_field,player_num)))
            next_node=Node(field=next_field,player_num=player_num,\
                finite_state_flg= flg==True,depth=node.depth+1)
        next_node.parent_node=node
        node.child_nodes.append((move,next_node))
        next_node.current_probability=1/len(new_choices)
        #if move==(0,0,0):   
        #   node.action_value_dict[move]=self.state_value(next_field,player_num)

        #node.action_value_dict[move]=0.0
        #mylogger.info("append {} to action_value_dict:{}".format(move,node.action_value_dict))
        return next_node,1#/len(new_choices)

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
    
    def exp3(self,node,player_num=0):
        over_all_n = node.visit_num
        #A = len(node.child_nodes)
        dict_key_list=list(node.action_value_dict.keys())
        A = len(dict_key_list)

        assert A>0,"child_nodes:{} child_moves:{} able_to_actions:{}".format(node.child_nodes,node.children_moves,node.get_exist_action())


        e = np.e
        #value = (A*np.log(A))/((e-1)*over_all_n)
        value = (A*np.log(A))/((e-1)*over_all_n)
        value = np.sqrt(value)
        gamma = min(1,value)
        #gamma = 0.01
        #mylogger.info("gamma:{}".format(gamma))
        #gamma = 0.5
        #eta = gamma / over_all_n
        #eta = 10000*gamma/np.sqrt(over_all_n)
        eta = 100*gamma/A
        #mylogger.info("eta:{}".format(eta))
        #distribution = [0 for i in range(A)]
        #value_list=[0 for i in range(A)]
        distribution=[0.0]*len(dict_key_list)
        #value_list=[node.action_value_dict[child[0]] for child in node.child_nodes]
        value_list=[node.action_value_dict[key]/over_all_n for key in dict_key_list]
        #assert all(ele<=1 for ele in value_list),"{},over_all_n:{}".format(value_list,over_all_n)
        first_term = gamma / A
        max_value=max(value_list)
        #value_list=[(value_list[i]-max_value)*eta for i in range(len(value_list))]
        
        value_list=[(value_list[i]-max_value)*eta for i in range(len(value_list))]
        value_list=np.exp(value_list)/np.sum(np.exp(value_list))
        for i in range(len(distribution)):
            second_term = (1-gamma)*(value_list[i])
            distribution[i]=first_term + second_term
        
        """
        for i in range(len(distribution)):
            tmp=sum(np.exp(eta*(value_list[j] - value_list[i])) for j in range(len(value_list)))
            #mylogger.info("tmp[({})]:{}".format(dict_key_list[i],tmp))
            second_term=(1-gamma)/tmp
            distribution[i]=first_term+second_term
        """
        #mylogger.info("action_value_dict:{}".format(node.action_value_dict))
        #mylogger.info("distribution:{}".format(distribution))
        #assert len(dict_key_list)<5
        return distribution#,index
        
    def roulette(self,node,player_num=0,show_flg=False):
        distribution=self.exp3(node,player_num=player_num)
        if show_flg==True:
            mylogger.info("distribution:{}".format(distribution))
        assert len(distribution)>0
        population=[key for key in list(node.action_value_dict.keys())]
        assert len(distribution)==len(population),"{},{}".format(len(distribution),len(population))
        assert np.nan not in distribution,"{}".format(distribution)
        decision=random.choices(population,weights=distribution,k=1)
        for cell in node.child_nodes:
            if cell[0]==decision[0]:
                decision_node=cell[-1]
                action=cell[0]
                index=population.index(cell[0])
                #mylogger.info("probability:{}".format(distribution[index]))
                decision_node.current_probability=distribution[index]
                return decision_node,action,distribution[index]
        assert decision[0]!=None,"population:{},distribution:{}".format(population,distribution)
        assert False,"{},{} {}".format(decision[0],node.action_value_dict,node.child_nodes)
        
    def select_expand(self,node,action,player_num=0):

        field=node.field

        new_choices = node.get_able_action_list()
        assert action in new_choices
        move = action
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
            
            next_node=Node(field=next_field,player_num=player_num,\
                finite_state_flg= flg==True,depth=node.depth+1)
        next_node.parent_node=node
        node.child_nodes.append((move,next_node))
        return next_node,1/len(new_choices)


    def back_up(self,last_visited,reward,player_num=0):
        current=last_visited
        probabilities=[]
        while True:
            current.visit_num += 1
            #current.value += reward
            


            if current.is_root==True:
                break
            probabilities.append(current.current_probability)
            if current!=last_visited:
                target_action = [cell[0] for i, cell in enumerate(current.parent_node.child_nodes) if cell[1] == current][0]
                p=0
                #mylogger.info("probabilities:{}".format(probabilities))
                if len(probabilities)==1:
                    p=probabilities[0]
                else:
                    p = np.prod(np.array(probabilities))
                current.parent_node.action_value_dict[target_action]+=reward/p

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
        return 'EXP3_MCTSPolicy'

"""
class Information_Set_MCTSPolicy(MCTSPolicy):
    def __init__(self):

"""

class Information_Set():
    def __init__(self):
        self.field=None


