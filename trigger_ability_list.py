import random
import card_setting
from my_moduler import get_module_logger
mylogger = get_module_logger(__name__)
from util_ability import *
from my_enum import * 
class trigger_ability_001:
    def __init__(self):
        self.count=0
    def __call__(self,field,player,opponent,virtual,target,itself,state_log=None):
        #実装途中
        if state_log!=None and state_log[0]==State_Code.SET.value and state_log[1][0]==opponent.player_num:
            get_damage_to_player(player,virtual,num=1)

class trigger_ability_002:
    def __init__(self):
        self.count=0
    def __call__(self,field,player,opponent,virtual,target,itself,state_log=None):
        #実装途中
        if state_log!=None and state_log[0]==State_Code.RESTORE_PLAYER_LIFE.value and state_log[1]==player.player_num:
            for creature_id in field.get_creature_location()[player.player_num]:
                creature=field.card_location[player.player_num][creature_id]
                buff_creature(creature,params=[1,1])
                if virtual==False:
                    mylogger.info("player{}'s {} get +1/+1".format(player.player_num+1,creature.name))

class trigger_ability_003:
    def __init__(self):
        self.count=0
    def __call__(self,field,player,opponent,virtual,target,itself,state_log=None):
        #実装途中
        if state_log!=None and state_log[0]==State_Code.RESTORE_PLAYER_LIFE.value and state_log[1]==player.player_num:
            itself.down_count(num=1,virtual=virtual)

class trigger_ability_004:
    def __init__(self):
        self.count=0
    def __call__(self,field,player,opponent,virtual,target,itself,state_log=None):
        if state_log!=None and state_log[0]==State_Code.RESTORE_PLAYER_LIFE.value and state_log[1]==player.player_num:
            get_damage_to_player(player,virtual,num=2)

class trigger_ability_005:
    def __init__(self):
        self.count=0
    def __call__(self,field,player,opponent,virtual,target,itself,state_log=None):
        #実装途中
        if state_log!=None and state_log[0]==State_Code.RESTORE_PLAYER_LIFE.value and state_log[1]==player.player_num:
            damage=1+int(itself.evolved)
            for creature_id in field.get_creature_location()[opponent.player_num]:
                get_damage_to_creature(field,opponent,virtual,creature_id,num=damage)


class trigger_ability_006:
    def __init__(self):
        self.count=0
    def __call__(self,field,player,opponent,virtual,target,itself,state_log=None):
        #実装途中
        if state_log!=None and state_log[0]==State_Code.SET.value and state_log[1][0]==player.player_num:

            if state_log[1][1]=="Creature":
                if card_setting.creature_list[state_log[1][2]][4][-1]==1:
                    buff_creature_until_end_of_turn(itself,params=[1,0])
            elif state_log[1][1]=="Amulet":
                if card_setting.amulet_list[state_log[1][2]][2][-1]==1:
                    buff_creature_until_end_of_turn(itself,params=[1,0])
            



        

trigger_ability_dict={1:trigger_ability_001,2:trigger_ability_002,3:trigger_ability_003,4:trigger_ability_004,5:trigger_ability_005,\
                    6:trigger_ability_006}