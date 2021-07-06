import random
from my_moduler import get_module_logger
import card_setting

mylogger = get_module_logger(__name__)
from util_ability import *
from my_enum import *

def ability_condition_001(field,player_num,itself):
    for card in field.card_location[player_num]:
        if card.card_class.value == LeaderClass.NEUTRAL.value:
            return True
    return False

def ability_condition_002(field,player_num,itself):
    return field.players[player_num].check_overflow()


creature_ability_condition_dict={
    1:ability_condition_001,
    2:ability_condition_002
}
