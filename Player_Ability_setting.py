import random
import card_setting
from my_moduler import get_module_logger
from util_ability import *
mylogger = get_module_logger(__name__)
from my_enum import *

PRIORITY_MAX_LIMIT = 10


class reduce_damage_to_zero():
    def __init__(self):
        self.priority = PRIORITY_MAX_LIMIT
        self.ability_id = 1
    def __call__(argument=None, state_code=None):
        if state_code[0] == State_Code.GET_DAMAGE.value and type(argument) == int:
            return 0
        else:
            return argument


class can_not_take_more_than_x_damage():
    def __init__(self, num=0):
        self.priority = PRIORITY_MAX_LIMIT - 1
        self.damage_upper_bound = num

    def __call__(self, argument=None, state_code=None):
        if state_code[0] == State_Code.GET_DAMAGE.value and type(argument) == int:
            return (argument, self.damage_upper_bound)[argument > self.damage_upper_bound]
        else:
            return argument

class restore_1_defense_to_all_allies():
    def __init__(self):
        self.name = "restore_1_defense_to_all_allies"
        self.ability_id = 2

    def __call__(self,field, player, virtual, state_log=None):
        if state_log[0] != State_Code.END_OF_TURN.value:
            return
        if state_log[1] != player.player_num:
            return
        if not virtual:
            mylogger.info("De La Fille, Gem Princess's leader effect is actived")
        field.restore_player_life(player=player, num=1, virtual=virtual)
        for follower in field.card_location[player.player_num]:
            if follower.card_category == "Creature":
                field.restore_follower_toughness(follower=follower, num=1, virtual=virtual, at_once=True)

class  search_three_followers():
    def __init__(self):
        self.name = "search_three_followers"
        self.ability_id = 3
    def __call__(self,field, player, virtual, state_log=None):
        if state_log is None: return
        if state_log[0] != State_Code.START_OF_TURN.value:
            return
        if state_log[1] != player.player_num:
            return
        if not virtual:
            mylogger.info("Staircase to Paradise's ability is actived")
            mylogger.info("Search three followers from deck")
        condition = lambda card: card.card_category == "Creature"

        search_cards(player, condition, virtual, num=3)
        while True:
            #if search_three_followers in field.player_ability[player.player_num]:
            #    field.player_ability[player.player_num].remove(search_three_followers)
            if self.ability_id in field.player_ability[player.player_num]:
                field.player_ability[player.player_num].remove(self.ability_id)
            else:
                break

player_ability_name_dict = {id(restore_1_defense_to_all_allies):"restore_1_defense_to_all_allies"}
player_ability_id_2_func = {
    2:restore_1_defense_to_all_allies(),
    3:search_three_followers()}