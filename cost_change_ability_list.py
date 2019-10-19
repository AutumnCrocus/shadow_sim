import random
from my_moduler import get_module_logger
import card_setting
mylogger = get_module_logger(__name__)

def cost_change_ability_001(itself,field,player):
    if player.check_vengence()==True:
        itself.cost=max(0,itself.origin_cost-4)
    else:
        itself.cost=max(0,itself.origin_cost)

cost_change_ability_dict={1:cost_change_ability_001}