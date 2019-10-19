
import random
from my_moduler import get_module_logger
import card_setting
mylogger = get_module_logger(__name__)
from util_ability import *
#0:strike 1:follower-strike 2:leader-strike 3:clash
def creature_battle_ability_001(field,player,opponent,itself,target,situation_num=[],virtual=False):#仮仕様
    def until_end_of_turn(field,player,opponent,virtual,target,itself):
        itself.ability.remove(2)
        itself.turn_end_ability.remove(until_end_of_turn)
    if 1 in situation_num and type(target)==card_setting.Creature:
        if target.power>=5:
            itself.ability.append(2)
            itself.turn_end_ability.append(until_end_of_turn)
            if virtual==False:
                mylogger.info("{} get bane until end of this turn".format(itself.name))
        

def creature_battle_ability_002(field,player,opponent,itself,target,situation_num=[],virtual=False):#仮仕様
    if 0 in situation_num:
        get_damage_to_player(opponent,virtual,num=2)
        for creature_id in field.get_creature_location()[opponent.player_num]:
            get_damage_to_creature(field,opponent,virtual,creature_id,num=2)
        field.check_death(player_num=player.player_num,virtual=virtual)

battle_ability_dict={1:creature_battle_ability_001,2:creature_battle_ability_002}