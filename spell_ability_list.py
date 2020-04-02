import random
from my_moduler import get_module_logger
import card_setting

mylogger = get_module_logger(__name__)
from util_ability import *


def spell_ability_001(field, player, opponent, virtual, target, itself):
    get_damage_to_creature(field, opponent, virtual, target, num=3)
    put_card_in_hand(field, player, virtual, name="Earth Essence", card_category="Amulet")


def spell_ability_002(field, player, opponent, virtual, target, itself):
    draw_cards(player, virtual, num=1)


def spell_ability_003(field, player, opponent, virtual, target, itself):
    if virtual == False:
        mylogger.info("All creatures get 2 damage")
    get_damage_to_all_creature(field, virtual, num=2)


def spell_ability_004(field, player, opponent, virtual, target, itself):
    if virtual == False:
        mylogger.info("Player {}'s {} get +2/+2".format(player.player_num + 1,
                                                        field.card_location[player.player_num][target].name))
    creature = field.card_location[player.player_num][target]
    buff_creature(creature, params=[2, 2])


def spell_ability_005(field, player, opponent, virtual, target, itself):
    if virtual == False:
        mylogger.info("All creatures are destroyed!")
    for j in range(2):
        side_num = (player.player_num + j) % 2
        # side=field.get_creature_location()[side_num]
        side = field.card_location[side_num]
        for thing in side:
            # thing=field.card_location[side_num][i]
            if thing.card_category == "Creature" and KeywordAbility.CANT_BE_DESTROYED_BY_EFFECTS.value not in thing.ability:
                thing.is_in_field = False
                thing.is_in_graveyard = True
    field.check_death(player_num=player.player_num, virtual=virtual)

    """
    while True:
        break_flg=True
        continue_flg=False
        for i,side in enumerate(field.card_location):
            for j,creature in enumerate(side):
                if creature.card_category=="Creature":
                    field.remove_card([i,j],virtual=virtual)
                    break_flg=False
                    continue_flg=True
                    break
            if continue_flg==True:
                break

        if break_flg==True:
            break
    """


def spell_ability_006(field, player, opponent, virtual, target, itself):
    destroy_opponent_creature(field, opponent, virtual, target)
    get_damage_to_player(opponent, virtual, num=2)


def spell_ability_007(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Pirate")
    summon_creature(field, player, virtual, name="Viking")


def spell_ability_008(field, player, opponent, virtual, target, itself):
    if field.cost[player.player_num] >= 7:
        draw_cards(player, virtual, num=1)
    gain_max_pp(field, player, virtual, num=1)


def spell_ability_009(field, player, opponent, virtual, target, itself):
    if target == -1:
        get_damage_to_player(opponent, virtual, num=1)
    else:
        # mylogger.info("target_id:{}".format(target))
        # field.show_field()
        get_damage_to_creature(field, opponent, virtual, target, num=1)
    draw_cards(player, virtual, num=1)


def spell_ability_010(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Clay Golem")


def spell_ability_011(field, player, opponent, virtual, target, itself):
    get_damage_to_creature(field, opponent, virtual, target, num=itself.spell_boost + 1)


def spell_ability_012(field, player, opponent, virtual, target, itself):
    # mylogger.info("2 draw")
    draw_cards(player, virtual, num=2)


def spell_ability_013(field, player, opponent, virtual, target, itself):
    destroy_opponent_creature(field, opponent, virtual, target)


def spell_ability_014(field, player, opponent, virtual, target, itself):
    for i in range(itself.spell_boost + 1):
        tmp = field.get_creature_location()[opponent.player_num]
        if tmp != []:
            target_id = random.choice(tmp)
            # mylogger.info("target_id:{}".format(target))
            # field.show_field()
            get_damage_to_creature(field, opponent, virtual, target_id, num=1)
        else:
            break


def spell_ability_015(field, player, opponent, virtual, target, itself):
    if target == -1:
        get_damage_to_player(opponent, virtual, num=1)
    else:
        # mylogger.info("target_id:{}".format(target))
        # field.show_field()
        get_damage_to_creature(field, opponent, virtual, target, num=1)


def spell_ability_016(field, player, opponent, virtual, target, itself):
    if not virtual:
        mylogger.info("Player {} get extra turn".format(player.player_num + 1))
    field.ex_turn_count[player.player_num] += 1


def spell_ability_017(field, player, opponent, virtual, target, itself):
    # mylogger.info("target:{}".format(target))
    return_card_to_hand(field, target, virtual)
    draw_cards(player, virtual, num=1)


def spell_ability_018(field, player, opponent, virtual, target, itself):
    get_damage_to_player(player, virtual, num=2)
    if field.check_game_end():
        return
    draw_cards(player, virtual, num=2)


def spell_ability_019(field, player, opponent, virtual, target, itself):
    get_damage_to_player(player, virtual, num=2)
    if field.check_game_end():
        return
    if target == -1:
        get_damage_to_player(opponent, virtual, num=3)
    else:
        get_damage_to_creature(field, opponent, virtual, target, num=3)


def spell_ability_020(field, player, opponent, virtual, target, itself):
    get_damage_to_creature(field, opponent, virtual, target, num=4)
    field.restore_player_life(player=player, num=2, virtual=virtual)


def spell_ability_021(field, player, opponent, virtual, target, itself):
    get_damage_to_all_creature(field, virtual, num=8)


def spell_ability_022(field, player, opponent, virtual, target, itself):
    if necromancy(field, player, num=2, virtual=virtual) == True:
        get_damage_to_creature(field, opponent, virtual, target, num=4)
    else:
        get_damage_to_creature(field, opponent, virtual, target, num=2)


def spell_ability_023(field, player, opponent, virtual, target, itself):
    while len(field.card_location[player.player_num]) < field.max_field_num:
        if necromancy(field, player, num=1, virtual=virtual) == True:
            summon_creature(field, player, virtual, name="Ghost")
        else:
            break


def spell_ability_024(field, player, opponent, virtual, target, itself):
    summon_creature(field, player, virtual, name="Zombie", num=3)
    if necromancy(field, player, num=6, virtual=virtual) == True:
        if virtual == False:
            mylogger.info("Give +0/+1 and Ward to allied Zombies.")
        for creature_id in field.get_creature_location()[player.player_num]:
            creature = field.card_location[player.player_num][creature_id]
            if creature.name == "Zombie":
                buff_creature(creature, params=[0, 1])
                if 3 not in creature.ability:
                    add_ability_to_creature(field, player, creature, virtual, add_ability=[KeywordAbility.WARD.value])
                    # creature.ability.append(KeywordAbility.WARD.value)


def spell_ability_025(field, player, opponent, virtual, target, itself):
    if field.card_location[opponent.player_num][target].get_current_toughness() > 3:
        raise Exception("Illegal Target!")

    field.banish_card([opponent.player_num, target], virtual=virtual)


def spell_ability_026(field, player, opponent, virtual, target, itself):
    return_card_to_hand(field, [player.player_num, target], virtual)
    draw_cards(player, virtual)


def spell_ability_027(field, player, opponent, virtual, target, itself):
    put_card_in_hand(field, player, virtual, name="Fairy", card_category="Creature", num=2)


def spell_ability_028(field, player, opponent, virtual, target, itself):
    return_card_to_hand(field, [player.player_num, target[0]], virtual)
    get_damage_to_creature(field, opponent, virtual, target[1], num=3)


def spell_ability_029(field, player, opponent, virtual, target, itself):
    get_damage_to_creature(field, opponent, virtual, target, num=2)
    put_card_in_hand(field, player, virtual, name="Fairy", card_category="Creature")


def spell_ability_030(field, player, opponent, virtual, target, itself):
    return_card_to_hand(field, [player.player_num, target], virtual)
    choices = field.get_creature_location()[opponent.player_num]
    if choices == []:
        return
    target_id = random.choice(choices)
    return_card_to_hand(field, [opponent.player_num, target_id], virtual)


def spell_ability_031(field, player, opponent, virtual, target, itself):
    hand_len = len(player.hand)
    for i in range(hand_len):
        get_damage_to_random_creature(field, opponent, virtual, num=1)


def spell_ability_032(field, player, opponent, virtual, target, itself):
    get_damage_to_creature(field, opponent, virtual, target, num=2)


def spell_ability_033(field, player, opponent, virtual, target, itself):
    get_damage_to_creature(field, opponent, virtual, target, num=3)
    if not field.card_location[opponent.player_num][target].is_in_field:
        field.remove_card([opponent.player_num, target], virtual=virtual)
    if itself.active_enhance_code[0]:
        for creature_id in field.get_creature_location()[opponent.player_num]:
            get_damage_to_creature(field, opponent, virtual, creature_id, num=2)


def spell_ability_034(field, player, opponent, virtual, target, itself):
    gain_max_pp(field, player, virtual, num=1)
    draw_cards(player, virtual, num=2)
    field.restore_player_life(player=player, num=3, virtual=virtual)


def spell_ability_035(field, player, opponent, virtual, target, itself):
    if not itself.active_enhance_code[0]:
        field.banish_card([opponent.player_num, target], virtual=virtual)
    else:
        index = 0
        while index < len(field.card_location[opponent.player_num]):
            before = len(field.card_location[opponent.player_num])
            field.banish_card([opponent.player_num, index], virtual=virtual)
            after = len(field.card_location[opponent.player_num])
            if after == before:
                index += 1


def spell_ability_036(field, player, opponent, virtual, target, itself):
    if not itself.active_enhance_code[0]:
        if field.card_location[opponent.player_num][target].origin_cost > 2:
            player.show_hand()
            field.show_field()
            raise Exception("Over Cost Error:target_name={}"
                            .format(field.card_location[opponent.player_num][target].name))

    field.remove_card([opponent.player_num, target], virtual=virtual)


def spell_ability_037(field, player, opponent, virtual, target, itself):
    """
    Summon a Club Soldier, a Heart Guardian, and a Spade Raider.
    """
    summon_creature(field, player, virtual, name="Crab Soldier")
    summon_creature(field, player, virtual, name="Heart Guardian")
    summon_creature(field, player, virtual, name="Spade Raider")


def spell_ability_038(field, player, opponent, virtual, target, itself):
    """
    Put 2 random Commanders that cost 3 play points from your deck into play.
    """
    condition = lambda card: card.card_category == "Creature" and card.trait.value == Trait[
        "COMMANDER"].value and card.origin_cost == 3
    put_card_from_deck_in_play(field, player, virtual, condition=condition)
    put_card_from_deck_in_play(field, player, virtual, condition=condition)


def spell_ability_039(field, player, opponent, virtual, target, itself):
    """
    Destroy an allied follower.
    Draw 2 cards.
    """
    destroy_opponent_creature(field, player, virtual, target)
    draw_cards(player, virtual, num=2)


def spell_ability_040(field, player, opponent, virtual, target, itself):
    """
    Deal X damage to an enemy follower.
    X equals the number of allied followers in play.
    """
    damage = len(field.get_creature_location()[player.player_num])
    get_damage_to_creature(field, opponent, virtual, target, num=damage)


def spell_ability_041(field, player, opponent, virtual, target, itself):
    """
    Summon 1 Snowman.
    Spellboost: Summon 1 more.
    """
    summon_creature(field, player, virtual, name="Snowman", num=1 + itself.spell_boost)


def spell_ability_042(field, player, opponent, virtual, target, itself):
    """
    Summon 2 Clay Golems.
    """
    summon_creature(field, player, virtual, name="Clay Golem", num=2)


def spell_ability_043(field, player, opponent, virtual, target, itself):
    """
    Deal 6 damage to an enemy follower.
    """
    assert target is not None, "Non-target"
    get_damage_to_creature(field, opponent, virtual, target, num=6)


def spell_ability_044(field, player, opponent, virtual, target, itself):
    """
    Destroy an enemy follower.
    Necromancy (4): Summon a Zombie.
    """
    assert target is not None, "Non-target"
    destroy_opponent_creature(field, opponent, virtual, target)
    if necromancy(field, player, num=4, virtual=virtual):
        summon_creature(field, player, virtual, name="Zombie")


def spell_ability_045(field, player, opponent, virtual, target, itself):
    """
    Deal 2 damage to your leader.
    Destroy an enemy follower.
    """
    assert target is not None, "non-target-error"
    get_damage_to_player(player, virtual, num=2)
    if field.check_game_end():
        return
    destroy_opponent_creature(field, opponent, virtual, target)


def spell_ability_046(field, player, opponent, virtual, target, itself):
    """
    Subtract 2 from the Countdown of an allied amulet.
    Draw a card.
    """
    assert target < field.card_num[player.player_num], "out-of-range target:{}".format(target)
    target_amulet = field.card_location[player.player_num][target]
    assert target_amulet.card_category == "Amulet", "illegal target error name:{}".format(target_amulet.name)
    target_amulet.down_count(num=2, virtual=virtual)
    draw_cards(player, virtual)


def spell_ability_047(field, player, opponent, virtual, target, itself):
    """
    Banish an enemy follower.
    """
    assert target < field.card_num[opponent.player_num], "out-of-range target:{}".format(target)
    field.banish_card([opponent.player_num, target], virtual=virtual)


def spell_ability_048(field, player, opponent, virtual, target, itself):
    """
    Deal 2 damage to an enemy follower.
    If Resonance is active for you, deal 4 damage instead.
    """
    if player.check_resonance():
        get_damage_to_creature(field, opponent, virtual, target, num=4)
    else:
        get_damage_to_creature(field, opponent, virtual, target, num=2)


def spell_ability_049(field, player, opponent, virtual, target, itself):
    """
    Put 2 Puppets into your hand.
    Deal 1 damage to all enemy followers.
    """
    put_card_in_hand(field, player, virtual, name="Puppet",
                     card_category="Creature", num=2)
    get_damage_to_all_enemy_follower(field, opponent, virtual, num=1)


def spell_ability_050(field, player, opponent, virtual, target, itself):
    """
    Put a Conjure Guardian into your hand.
    Enhance (6): Put 2 more into your hand.
    Earth Rite: Subtract 1 from the cost of all Conjure Guardians in your hand.
    """
    put_card_in_hand(field, player, virtual, name="Conjure Guardian", card_category="Spell")
    if itself.active_enhance_code[0]:
        put_card_in_hand(field, player, virtual, name="Conjure Guardian", card_category="Spell", num=2)

    if earth_rite(field, player, virtual):
        if not virtual:
            mylogger.info("Subtract 1 from the cost of all Conjure Guardians in Player{}'s hand."
                          .format(player.player_num + 1))
        for card in player.hand:
            if card.name == "Conjure Guardian":
                card.cost = max(0, card.cost - 1)


def spell_ability_051(field, player, opponent, virtual, target, itself):
    """
    Draw 2 cards.
    Earth Rite: Draw 3 cards instead. Then restore 1 defense to your leader.
    """
    if earth_rite(field, player, virtual):
        draw_cards(player, virtual, num=3)
        restore_player_life(player, virtual, num=1)
    else:
        draw_cards(player, virtual, num=2)


def spell_ability_052(field, player, opponent, virtual, target, itself):
    """
    Return all followers to the players' hands.
    """
    for i in range(2):
        side_id = (i + player.player_num) % 2
        side = field.card_location[side_id]
        location_id = 0
        while location_id < len(side):
            if side[location_id].card_category == "Creature":
                before = len(side)
                return_card_to_hand(field, [side_id, location_id], virtual)
                after = len(side)
                location_id += int(before == after)
            else:
                location_id += 1


def spell_ability_053(field, player, opponent, virtual, target, itself):
    """
    Discard your hand. Draw a card for each card you discarded.
    """
    if not virtual:
        mylogger.info("Discard Player {}'s hand.".format(player.player_num+1))
    hand_len = len(player.hand)
    while len(player.hand)>0:
        field.discard_card(player,0)
    draw_cards(player,virtual,num=hand_len)


def spell_ability_054(field, player, opponent, virtual, target, itself):
    """
    Deal 3 damage to an enemy follower.
    Enhance (7): Then summon 3 Zombies.
    """
    get_damage_to_creature(field,opponent,virtual,target,num=3)
    if itself.active_enhance_code[0]:
        summon_creature(field,player,virtual,name="Zombie",num=3)


def token_spell_ability_001(field, player, opponent, virtual, target, itself):
    get_damage_to_enemy(field, opponent, virtual, target, num=3)
    """
    if target == -1:
        get_damage_to_player(opponent, virtual, num=3)
    else:
        get_damage_to_creature(field, opponent, virtual, target, num=3)
    """
    if field.check_game_end():
        return
    if earth_rite(field, player, virtual):
        draw_cards(player, virtual, num=1)


def token_spell_ability_002(field, player, opponent, virtual, target, itself):
    num = 0
    for card in field.card_location[player.player_num]:
        if card.trait.value == Trait["EARTH_SIGIL"].value:
            num += 1
    for i in range(num):
        earth_rite(field, player, virtual)
        tmp = random.random()
        if tmp < 1 / 3:
            summon_creature(field, player, virtual, name="Clay Golem")
        elif tmp < 2 / 3:
            put_card_in_hand(field, player, virtual, name="Earth Essence", card_category="Amulet")
        else:
            get_damage_to_player(player, virtual, num=2)

        if field.check_game_end():
            break


def token_spell_ability_003(field, player, opponent, virtual, target, itself):
    """
    Accelerate (2): Summon an Earth Essence. Put a Veridic Ritual into your hand.
    """
    set_amulet(field, player, virtual, name="Earth Essence")
    put_card_in_hand(field, player, virtual, name="Veridic Ritual", card_category="Spell")


def token_spell_ability_004(field, player, opponent, virtual, target, itself):
    """
    Summon a Guardian Golem.
    """
    summon_creature(field, player, virtual, name="Guardian Golem")


def token_spell_ability_005(field, player, opponent, virtual, target, itself):
    """
    Deal 2 damage to an enemy.
    """
    get_damage_to_enemy(field, opponent, virtual, target, num=2)


def token_spell_ability_006(field, player, opponent, virtual, target, itself):
    """
    Give +2/+0 to an allied follower.
    """
    target_follower = field.card_location[player.player_num][target]
    if not virtual:
        mylogger.info("Player{}'s {} get  +2/0".format(player.player_num+1,target_follower.name))
    buff_creature(target_follower,params=[2,0])


spell_ability_dict = {1: spell_ability_001, 2: spell_ability_002, 3: spell_ability_003, 4: spell_ability_004,
                      5: spell_ability_005,
                      6: spell_ability_006, 7: spell_ability_007, 8: spell_ability_008, 9: spell_ability_009,
                      10: spell_ability_010,
                      11: spell_ability_011, 12: spell_ability_012, 13: spell_ability_013, 14: spell_ability_014,
                      15: spell_ability_015,
                      16: spell_ability_016, 17: spell_ability_017, 18: spell_ability_018, 19: spell_ability_019,
                      20: spell_ability_020,
                      21: spell_ability_021, 22: spell_ability_022, 23: spell_ability_023, 24: spell_ability_024,
                      25: spell_ability_025,
                      26: spell_ability_026, 27: spell_ability_027, 28: spell_ability_028, 29: spell_ability_029,
                      30: spell_ability_030,
                      31: spell_ability_031, 32: spell_ability_032, 33: spell_ability_033, 34: spell_ability_034,
                      35: spell_ability_035,
                      36: spell_ability_036, 37: spell_ability_037, 38: spell_ability_038, 39: spell_ability_039,
                      40: spell_ability_040, 41: spell_ability_041, 42: spell_ability_042, 43: spell_ability_043,
                      44: spell_ability_044, 45: spell_ability_045, 46: spell_ability_046, 47: spell_ability_047,
                      48: spell_ability_048, 49: spell_ability_049, 50: spell_ability_050, 51: spell_ability_051,
                      52: spell_ability_052, 53: spell_ability_053, 54: spell_ability_054,
                      -1: token_spell_ability_001, -2: token_spell_ability_002, -3: token_spell_ability_003,
                      -4: token_spell_ability_004, -5: token_spell_ability_005 ,-6: token_spell_ability_006}
