import random
from my_moduler import get_module_logger
import card_setting

mylogger = get_module_logger(__name__)
from util_ability import *
import Player_Ability_setting

def amulet_ability_001(field, player, opponent, virtual, target, itself):
    if field.turn_player_num != player.player_num: return
    tmp = field.get_creature_location()[player.player_num]
    # mylogger.info("tmp:{}".format(tmp))
    if tmp != []:
        target_id = random.choice(tmp)
        creature = field.card_location[player.player_num][target_id]
        buff_creature(creature, params=[1, 1])
        if not virtual:
            mylogger.info("Player {}'s {} get +1/+1".format(player.player_num + 1, creature.name))


def amulet_ability_002(field, player, opponent, virtual, target, itself):
    if field.turn_player_num == player.player_num:
        summon_creature(field, player, virtual, name="Windblast Dragon")


def amulet_ability_003(field, player, opponent, virtual, target, itself):
    if field.graveyard.shadows[player.player_num] < 30:
        return
    if field.turn_player_num != player.player_num:
        return
    if not virtual:
        mylogger.info("Deal 6 damage to all opponent")
    get_damage_to_player(opponent,virtual,num=6)
    #opponent.get_damage(6)
    if opponent.life <= 0:
        return
    get_damage_to_all_enemy_follower(field, opponent, virtual, num=6)
    #mylogger.info("opponent.life:{}".format(opponent.life))
    #for i, thing in enumerate(field.card_location[opponent.player_num]):
    #    if thing.card_category == "Creature":
    #        damage = thing.get_damage(6)
    #        # mylogger.info("Player {}'s {} get {} damage".format(opponent.player_num+1,thing.name,damage))

    while True:
        break_flg = True
        continue_flg = False
        for i, creature in enumerate(field.card_location[opponent.player_num]):
            if creature.is_in_graveyard:
                field.remove_card([opponent.player_num, i], virtual=virtual)
                break_flg = False
                continue_flg = True
                break
            if continue_flg:
                break

        if break_flg:
            break


def amulet_ability_004(field, player, opponent, virtual, target, itself):
    draw_cards(player, virtual, num=2)


def amulet_ability_005(field, player, opponent, virtual, target, itself):
    opponent_side = field.get_creature_location()[opponent.player_num]
    if len(opponent_side) > 0:
        tmp = random.choice(opponent_side)
        destroy_opponent_creature(field, opponent, virtual, tmp)


def amulet_ability_006(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Holy Falcon")


def amulet_ability_007(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Holyflame Tiger")


def amulet_ability_008(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Regal Falcon")


def amulet_ability_009(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Barong")


def amulet_ability_010(field, player, opponent, virtual, target, itself):
    if field.turn_player_num == player.player_num:
        get_damage_to_player(player, virtual, num=1)
        get_damage_to_player(opponent, virtual, num=1)


def amulet_ability_011(field, player, opponent, virtual, target, itself):
    """Whitefang Temple's end_of_turn ability"""
    if field.turn_player_num == player.player_num:
        if not virtual:
            mylogger.info("Whitefang Temple's end_of_turn ability is actived")
        field.restore_player_life(player=player, num=1, virtual=virtual)


def amulet_ability_012(field, player, opponent, virtual, target, itself):
    """
    Last Words: Summon a Holywing Dragon.
    """
    summon_creature(field, player, virtual, name="Holywing Dragon", num=1)


def amulet_ability_013(field, player, opponent, virtual, target, itself):
    """Moriae Encomium's fanfare ability"""
    draw_cards(player, virtual, num=1)


def amulet_ability_014(field, player, opponent, virtual, target, itself):
    destroy_random_creature(field, opponent, virtual)


def amulet_ability_015(field, player, opponent, virtual, target, itself):
    destroy_opponent_creature(field, opponent, virtual, target)


def amulet_ability_016(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Scrap Golem")


def amulet_ability_017(field, player, opponent, virtual, target, itself):
    if random.random() < 0.5:
        summon_creature(field, player, virtual, name="Clay Golem")
    else:
        summon_creature(field, player, virtual, name="Zombie")


def amulet_ability_018(field, player, opponent, virtual, target, itself):
    """
    Countdown (7)
    Fanfare: Enhance (5) - Subtract 7 from this amulet's Countdown.
    """
    if itself.active_enhance_code[0] == True:
        itself.down_count(num=7, virtual=virtual)


def amulet_ability_019(field, player, opponent, virtual, target, itself):
    """
    Last Words: At the start of your next turn, put 3 random followers from your deck into your hand.
    """


    if not virtual:
        mylogger.info(
            "Player{} get ability:'At the start of your next turn, put 3 random followers from your deck into your "
            "hand'".format(
                player.player_num+1))
    #field.player_ability[player.player_num].append(Player_Ability_setting.search_three_followers())
    field.player_ability[player.player_num].append(Player_Ability_setting.search_three_followers().ability_id)


def amulet_ability_020(field, player, opponent, virtual, target, itself):
    """
    Countdown (4)
    Last Words: Summon a Pegasus.
    """
    summon_creature(field, player, virtual, name="Pegasus")


def amulet_ability_021(field, player, opponent, virtual, target, itself):
    """
    Countdown (2)
    Last Words: Summon 2 Holyflame Tigers.
    """
    summon_creature(field, player, virtual, name="Holyflame Tiger", num=2)


def amulet_ability_022(field, player, opponent, virtual, target, itself):
    restore_player_life(player,virtual,num=1)


def amulet_ability_023(field, player, opponent, virtual, target, itself):
    """
    At the end of your turn, draw a card if at least 3 cards have been played this turn.
    """
    if field.turn_player_num == player.player_num:
        if field.players_play_num>=3:
            draw_cards(player,virtual)

amulet_ability_dict = \
    {1: amulet_ability_001, 2: amulet_ability_002, 3: amulet_ability_003, 4: amulet_ability_004,
     5: amulet_ability_005, 6: amulet_ability_006, 7: amulet_ability_007, 8: amulet_ability_008,
     9: amulet_ability_009, 10: amulet_ability_010, 11: amulet_ability_011, 12: amulet_ability_012,
     13: amulet_ability_013, 14: amulet_ability_014, 15: amulet_ability_015, 16: amulet_ability_016,
     17: amulet_ability_017, 18: amulet_ability_018, 19: amulet_ability_019, 20: amulet_ability_020,
     21: amulet_ability_021, 22:amulet_ability_022, 23:amulet_ability_023}
