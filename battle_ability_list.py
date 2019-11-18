import random
from my_moduler import get_module_logger
import card_setting

mylogger = get_module_logger(__name__)
from util_ability import *


# 0:strike 1:follower-strike 2:leader-strike 3:clash
def creature_battle_ability_001(field, player, opponent, itself, target, situation_num=[], virtual=False):  # 仮仕様
    def until_end_of_turn(field, player, opponent, virtual, target, it_self):
        it_self.ability.remove(2)
        it_self.turn_end_ability.remove(until_end_of_turn)

    if 1 in situation_num and type(target) == card_setting.Creature:
        if target.power >= 5:
            itself.ability.append(2)
            itself.turn_end_ability.append(until_end_of_turn)
            if not virtual:
                mylogger.info("{} get bane until end of this turn".format(itself.name))


def creature_battle_ability_002(field, player, opponent, itself, target, situation_num=[], virtual=False):  # 仮仕様
    if 0 in situation_num:
        get_damage_to_player(opponent, virtual, num=2)
        for creature_id in field.get_creature_location()[opponent.player_num]:
            get_damage_to_creature(field, opponent, virtual, creature_id, num=2)
        field.check_death(player_num=player.player_num, virtual=virtual)


def creature_battle_ability_003(field, player, opponent, itself, target, situation_num=[], virtual=False):  # 仮仕様
    """
    Strike: Put a Fairy into your hand.
    """
    if 0 in situation_num:
        put_card_in_hand(field, player, virtual, name="Fairy", card_category="Creature")


def creature_battle_ability_004(field, player, opponent, itself, target, situation_num=[], virtual=False):  # 仮仕様
    """
    Strike: Gain +2/+0 until the end of the turn.
    """
    if 0 in situation_num:
        buff_creature_until_end_of_turn(itself, params=[2, 0])
        if virtual == False:
            mylogger.info("{} get +2/0 until end of this turn".format(itself.name))


battle_ability_dict = {1: creature_battle_ability_001, 2: creature_battle_ability_002,
                       3: creature_battle_ability_003, 4: creature_battle_ability_004}
