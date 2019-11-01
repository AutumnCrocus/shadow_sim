import random
from my_moduler import get_module_logger
import card_setting
from my_enum import *
mylogger = get_module_logger(__name__)

def cost_change_ability_001(itself,field,player):
    if player.check_vengeance()==True:
        itself.cost=max(0,itself.origin_cost-4)
    else:
        itself.cost=max(0,itself.origin_cost)

def cost_change_ability_002(itself,field,player):
    for card in field.card_location[player.player_num]:
        if card.is_tapped==True and  card.trait.value==Trait.COMMANDER.value and card.origin_cost==3:
            itself.cost=itself.origin_cost-1
            return
        else:
            itself.cost=itself.origin_cost

cost_change_ability_dict={1:cost_change_ability_001,2:cost_change_ability_002}