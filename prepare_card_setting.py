# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import random
import math
from creature_ability_list import creature_ability_dict
from creature_ability_conditions import creature_ability_condition_dict
from spell_ability_list import spell_ability_dict
from amulet_ability_list import amulet_ability_dict
from cost_change_ability_list import cost_change_ability_dict
from battle_ability_list import battle_ability_dict
from trigger_ability_list import trigger_ability_dict
#from numba import jit
from collections import deque
from my_moduler import get_module_logger

mylogger = get_module_logger(__name__)
from my_enum import *
import csv
import pandas as pd
import warnings

# warnings.simplefilter('ignore', NumbaWarning)


def tsv_to_card_list(tsv_name):
    card_list = {}
    card_category = tuple(tsv_name.split("_"))[1]
    with open("Card_List_TSV/" + tsv_name) as f:
        reader = csv.reader(f, delimiter='\t', lineterminator='\n')
        for row in reader:
            #mylogger.info("row:{}".format(row))
            card_id = int(row[0])
            # card_cost=int(row[1])
            card_cost = int(row[2])
            # assert card_category in ["Creature","Spell","Amulet"]
            if card_id not in card_list: card_list[card_id] = []

            card_name = row[1]

            card_class = None
            card_list[card_id].append(card_cost)
            card_traits = None
            has_count = None
            if card_category == "Creature":
                card_class = LeaderClass[row[-2]].value
                card_traits = Trait[row[-1]].value
                power = int(row[3])
                toughness = int(row[4])
                ability = []
                if row[5] != "":
                    txt = list(row[5].split(","))
                    ability = [int(ele) for ele in txt]
                card_list[card_id].extend([power, toughness, ability])
            elif card_category == "Amulet":
                # mylogger.info("row_contents:{}".format(row))
                card_traits = Trait[row[-2]].value
                card_class = LeaderClass[row[-3]].value
                has_count = False
                if row[-1] != "False":
                    has_count = int(row[-1])
                ability = []
                if row[3] != "":
                    txt = tuple(row[3].split(","))
                    ability = [int(ele) for ele in txt]
                card_list[card_id].append(ability)

            elif card_category == "Spell":
                card_traits = Trait[row[-1]].value
                card_class = LeaderClass[row[-2]].value
            else:
                assert False, "{}".format(card_category)
            if card_class == LeaderClass["RUNE"].value:
                spell_boost = tuple(row[-3 - int(card_category == "Amulet")].split(","))
                check_spellboost = [bool(int(cell)) for cell in spell_boost]
                card_list[card_id].append([card_class, check_spellboost, card_traits])
            else:
                card_list[card_id].append([card_class, card_traits])
            if has_count != None:
                card_list[card_id].append(has_count)
            card_list[card_id].append(card_name)
    return card_list


def tsv_to_dataframe(tsv_name):
    card_category = tuple(tsv_name.split("_"))[1]
    my_columns = []
    sample = []
    assert card_category in ["Creature", "Spell", "Amulet"]
    if card_category == "Creature":
        my_columns = ["Card_id", "Card_name", "Cost", "Power", "Toughness", "Ability", "Class", "Trait", "Spell_boost"]
        sample = [0, "Sample", 0, 0, 0, [], "NEUTRAL", "NONE", "None"]
    elif card_category == "Spell":
        my_columns = ["Card_id", "Card_name", "Cost", "Class", "Trait", "Spell_boost"]
        sample = [0, "Sample", 0, "NEUTRAL", "NONE", "None"]
    elif card_category == "Amulet":
        my_columns = ["Card_id", "Card_name", "Cost", "Ability", "Class", "Trait", "Spell_boost", "Count_down"]
        sample = [0, "Sample", 0, [], "NEUTRAL", "NONE", "None", "None"]

    df = pd.DataFrame([sample], columns=my_columns)
    with open("Card_List_TSV/" + tsv_name) as f:
        # with open(tsv_name) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data = []
            # [1,"Goblin",1,1,2,[],"NEUTRAL","NONE","NONE"]
            card_id = int(row[0])
            card_name = row[1]
            card_cost = int(row[2])
            data.append(card_id)
            data.append(card_name)
            data.append(card_cost)

            card_class = None
            card_trait = None
            has_count = None
            if card_category == "Creature":
                card_class = LeaderClass[row[-2]].name
                card_trait = Trait[row[-1]].name
                power = int(row[3])
                toughness = int(row[4])
                ability = []
                if row[5] != "":
                    txt = tuple(row[5].split(","))
                    ability = [KeywordAbility(int(ele)).value for ele in txt]
                data.extend([power, toughness, ability])
            elif card_category == "Amulet":
                card_trait = Trait[row[-2]].name
                card_class = LeaderClass[row[-3]].name
                has_count = False
                if row[-1] != "False":
                    has_count = int(row[-1])
                ability = []
                if row[3] != "":
                    txt = tuple(row[3].split(","))
                    ability = [KeywordAbility(int(ele)).value for ele in txt]
                data.append(ability)

            elif card_category == "Spell":
                card_trait = Trait[row[-1]].name
                card_class = LeaderClass[row[-2]].name
            else:
                assert False, "{}".format(card_category)
            if card_class == LeaderClass["RUNE"].name:
                spell_boost = tuple(row[-3 - int(card_category == "Amulet")].split(","))
                check_spellboost = [bool(int(spell_boost[i])) for i in range(2)]
                spell_boost_type = "None"
                if check_spellboost[0]:
                    if check_spellboost[1]:
                        spell_boost_type = "Costdown"
                    else:
                        spell_boost_type = "Normal"

                data.extend([card_class, card_trait, spell_boost_type])
            else:
                data.extend([card_class, card_trait, "None"])
            if has_count is not None:
                data.append(has_count)
            new_df = pd.DataFrame([data], columns=my_columns)
            df = pd.concat([df, new_df])

    return df


def tsv_2_ability_dict(file_name, name_to_id=None):
    ability_dict = {}
    with open("Card_List_TSV/" + file_name) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            assert row[0] in name_to_id,"name_to_id:{}".format(name_to_id)
            ability_dict[name_to_id[row[0]]] = int(row[1])

    return ability_dict


creature_list = tsv_to_card_list("New-All_Creature_Card_List.tsv")

creature_name_to_id = {}
for key in tuple(creature_list.keys()):
    creature_name_to_id[creature_list[key][-1]] = key
creature_fanfare_ability = tsv_2_ability_dict("All_fanfare_list.tsv",
                                              name_to_id=creature_name_to_id)
creature_lastword_ability = tsv_2_ability_dict("All_lastword_list.tsv",
                                               name_to_id=creature_name_to_id)
creature_end_of_turn_ability = tsv_2_ability_dict("All_end_of_turn_list.tsv",
                                                  name_to_id=creature_name_to_id)
creature_start_of_turn_ability = tsv_2_ability_dict("All_start_of_turn_list.tsv",
                                                    name_to_id=creature_name_to_id)
creature_has_target = tsv_2_ability_dict("All_fanfare_target_list.tsv",
                                         name_to_id=creature_name_to_id)
creature_evo_effect = tsv_2_ability_dict("All_evo_effect_list.tsv",
                                         name_to_id=creature_name_to_id)
creature_has_evo_effect_target = {
    #29: 1,41: 1, 83: 2, 96: 1,
    creature_name_to_id["Dragon Warrior"]: Target_Type.ENEMY_FOLLOWER.value,
    creature_name_to_id["Wardrobe Raider"]: Target_Type.ENEMY_FOLLOWER.value,
    creature_name_to_id["Wind Reader Zell"]:Target_Type.ALLIED_FOLLOWER.value,
    creature_name_to_id["Maisy, Red Riding Hood"]:Target_Type.ENEMY_FOLLOWER.value,
    creature_name_to_id["Lyrial, Celestial Archer"]:Target_Type.ENEMY.value}
creature_target_regulation = {
    creature_name_to_id["Tsubaki"]: lambda target,card: target.power >= 5,
    creature_name_to_id["Princess Vanguard"]: lambda target,card: target.origin_cost == 1,
    creature_name_to_id["Sahaquiel"]: lambda
        target,card: target.card_category == "Creature" and target.card_class.name == "NEUTRAL",
    creature_name_to_id["Little Soulsquasher"]: lambda target,card: target.evolved,
    creature_name_to_id["White General"]: lambda target,card: target.trait.name == "OFFICER",
    creature_name_to_id["Big Knuckle Bodyguard"]: lambda target,card: target.get_current_toughness() <= 3}
another_target_func = lambda target, itself: id(target) != id(itself)
evo_target_regulation = {
    creature_name_to_id["Wind Reader Zell"]: another_target_func}
creature_ability_condition = {
    creature_name_to_id["Maisy, Red Riding Hood"]: 1,
    creature_name_to_id["Wind Reader Zell"]: 2
}
player_attack_regulation = \
    {16: lambda player: len(player.field.get_creature_location()[1 - player.player_num]) < 2}
creature_in_battle_ability_list = \
    {creature_name_to_id["Young Ogrehunter Momo"]: 1,
     creature_name_to_id["Israfil"]: 2,
     creature_name_to_id["Dark Elf Faure"]: 3,
     creature_name_to_id["Disaster Dragon"]: 4,
     creature_name_to_id["Spawn of the Abyss"]: 5}
creature_cost_change_ability_list = {97: 2}
can_only_attack_check = lambda field, player: field.check_ward()[1 - player.player_num]
creature_can_only_attack_list = {#49: can_only_attack_check,
                                 creature_name_to_id["Lurching Corpse"]:can_only_attack_check,
                                 creature_name_to_id["Attendant of Night"]:can_only_attack_check}
creature_trigger_ability_dict = {# 60: 1, 63: 4, 64: 5, 79: 6,
                                 creature_name_to_id["Yurius, Levin Duke"]:1,
                                 creature_name_to_id["Holy Bowman Kel"]:4,
                                 creature_name_to_id["Kel, Holy Marksman"]:5,
                                 creature_name_to_id["Fervid Soldier"]:6,
                                 creature_name_to_id["Bladed Hedgehog"]: 7,
                                 creature_name_to_id["Ephemera, Angelic Slacker"]: 8,
                                 creature_name_to_id["Prime Dragon Keeper"]: 10,
                                 creature_name_to_id["Shadow Reaper"]: 11,
                                 creature_name_to_id["Okami"]: 12,
                                 creature_name_to_id["Toy Soldier"]: 13,
                                 creature_name_to_id["Dragonrider"]: -1,
                                 creature_name_to_id["Tove"]: 14}
special_evo_stats_id = {26: 1, 27: 3, 29: 1, 41: 1, 52: 1, 66: 1, 77: 1,
                        creature_name_to_id["Puppeteer"]: 2}
evo_stats = {1: [1, 1], 2: [0, 0], 3: [3, 1]}
creature_earth_rite_list = [67, 68, 71, 90]
# 1:相手のフォロワー,2:自分のフォロワー,3:相手のフォロワーと相手リーダー,
# 4:自分と相手のフォロワー,5:自分と相手の全てのカード,6:自分の場のカード,7:自分の場のカードと相手の場のフォロワー,8:自分の他の手札
# 9:相手の場の全てのカード 10:自分のフォロワーと自分リーダー
creature_enhance_list = {
                        creature_name_to_id["Ax Fighter"]: [6],
                        creature_name_to_id["War Dog"]: [6],
                        creature_name_to_id["Grimnir, War Cyclone"]: [10],
                        creature_name_to_id["Albert, Levin Saber"]: [9],
                        creature_name_to_id["Tender Rabbit Healer"]: [7],
                        creature_name_to_id["Baphomet"]: [5]}
creature_enhance_target_list = {}
creature_enhance_target_regulation_list = {}

creature_accelerate_list = {  # 90: [1],
    creature_name_to_id["Orichalcum Golem"]: [1],
    creature_name_to_id["Clarke, Knowledge Seeker"]: [2]}
creature_accelerate_card_id_list = {  # 90: {1: -2},
    creature_name_to_id["Orichalcum Golem"]: {1: -2},
    creature_name_to_id["Clarke, Knowledge Seeker"]: {2: -3}}
creature_accelerate_target_list = {}
creature_accelerate_target_regulation_list = {}
# spell_list=tsv_to_card_list("ALL_Spell_Card_List.tsv")
creature_active_ability_card_id_list = {
    creature_name_to_id["Dark Dragoon Forte"]: Active_Ability_Check_Code.OVERFLOW.value,
    creature_name_to_id["Prime Dragon Keeper"]: Active_Ability_Check_Code.OVERFLOW.value,
    creature_name_to_id["Firstborn Dragon"]: Active_Ability_Check_Code.OVERFLOW.value,
    creature_name_to_id["Dragonguard"]: Active_Ability_Check_Code.OVERFLOW.value,
    creature_name_to_id["Bahamut"]: Active_Ability_Check_Code.BAHAMUT.value}

creature_active_ability_list = {
    creature_name_to_id["Dark Dragoon Forte"]: [KeywordAbility.CANT_BE_ATTACKED.value],
    creature_name_to_id["Prime Dragon Keeper"]: [KeywordAbility.CANT_BE_ATTACKED.value],
    creature_name_to_id["Firstborn Dragon"]: [KeywordAbility.WARD.value],
    creature_name_to_id["Dragonguard"]: [KeywordAbility.WARD.value],
    creature_name_to_id["Bahamut"]: [KeywordAbility.CANT_ATTACK_TO_PLAYER.value]
}
active_ability_check_func_list = {
    Active_Ability_Check_Code.OVERFLOW.value: lambda player: player.check_overflow(),
    Active_Ability_Check_Code.VENGEANCE.value: lambda player: player.check_vengeance(),
    Active_Ability_Check_Code.RESONANCE.value: lambda player: player.check_resonance(),
    Active_Ability_Check_Code.BAHAMUT.value: lambda player: len(
        player.field.get_creature_location()[1 - player.player_num]) >= 2}

spell_list = tsv_to_card_list("New-All_Spell_Card_List.tsv")
spell_name_to_id = {}
for key in tuple(spell_list.keys()):
    spell_name_to_id[spell_list[key][-1]] = key
spell_triggered_ability = tsv_2_ability_dict("All_spell_effect_list.tsv", name_to_id=spell_name_to_id)
spell_has_target = tsv_2_ability_dict("All_spell_target_list.tsv", name_to_id=spell_name_to_id)
# 1:相手のフォロワー,2:自分のフォロワー,3:相手のフォロワーと相手リーダー,
# 4:自分と相手のフォロワー,5:自分と相手の全てのカード,6:自分の場のカード,7:自分の場のカードと相手の場のフォロワー,8:自分の他の手札
# 9:相手の場の全てのカード
spell_target_regulation = {
    spell_name_to_id["Kaleidoscopic Glow"]: lambda target,card: target.origin_cost <= 2,
    spell_name_to_id["Blackened Scripture"]: lambda target,card: target.get_current_toughness() <= 3,
    spell_name_to_id["Seraphic Blade"]: lambda target,card: target.origin_cost <= 2}
spell_cost_change_ability_list = {
    # 20: 1,
    spell_name_to_id["Diabolic Drain"]: 1,
    # 21: 1
    spell_name_to_id["Revelation"]: 1}
spell_earth_rite_list = []
spell_enhance_list = {
    spell_name_to_id["Breath of the Salamander"]: [6],
    spell_name_to_id["Lightning Blast"]: [10],
    spell_name_to_id["Seraphic Blade"]: [6],
    spell_name_to_id["Golem Assault"]: [6],
    spell_name_to_id["Zombie Party"]: [7]}
spell_enhance_target_list = {
    spell_name_to_id["Breath of the Salamander"]: 1,
    spell_name_to_id["Seraphic Blade"]: 9,
    spell_name_to_id["Zombie Party"]: 1}
spell_enhance_target_regulation_list = {}

spell_accelerate_list = {}
spell_accelerate_card_id_list = {}
spell_accelerate_target_list = {}
spell_accelerate_target_regulation_list = {}
# amulet_list=tsv_to_card_list("ALL_Amulet_Card_List.tsv")
amulet_list = tsv_to_card_list("New-All_Amulet_Card_List.tsv")
amulet_name_to_id = {}
for key in tuple(amulet_list.keys()):
    amulet_name_to_id[amulet_list[key][-1]] = key
Earth_sigil_list = [-1, 15, 16]
amulet_start_of_turn_ability = {
    amulet_name_to_id["Well of Destiny"]: 1,
    amulet_name_to_id["Polyphonic Roar"]: 2}
amulet_end_of_turn_ability = {
    amulet_name_to_id["Path to Purgatory"]: 3,
    amulet_name_to_id["Bloodfed Flowerbed"]: 10,
    amulet_name_to_id["Whitefang Temple"]: 11,
    amulet_name_to_id["Harvest Festival"]: 23}
amulet_fanfare_ability = {
    amulet_name_to_id["Forgotten Sanctuary"]: 9,
    amulet_name_to_id["Moriae Encomium"]: 13,
    amulet_name_to_id["Tribunal of Good and Evil"]: 15,
    amulet_name_to_id["Scrap Iron Smelter"]: 16,
    amulet_name_to_id["Silent Laboratory"]: 17,
    amulet_name_to_id["Staircase to Paradise"]: 18,
    amulet_name_to_id["Golden Bell"]: 22}
amulet_lastword_ability = {
   amulet_name_to_id["Sacred Plea"]: 4,
   amulet_name_to_id["Heretical Inquiry"]: 5,
   amulet_name_to_id["Pinion Prayer"]: 6,
   amulet_name_to_id["Beastly Vow"]: 7,
   amulet_name_to_id["Divine Birdsong"]: 8,
   amulet_name_to_id["Forgotten Sanctuary"]: 9,
   amulet_name_to_id["Whitefang Temple"]: 12,
   amulet_name_to_id["Moriae Encomium"]: 14,
   amulet_name_to_id["Tribunal of Good and Evil"]: 14,
   amulet_name_to_id["Witch's Cauldron"]: 13,
   amulet_name_to_id["Staircase to Paradise"]: 19,
   amulet_name_to_id["Summon Pegasus"]: 20,
   amulet_name_to_id["Dual Flames"]: 21,
   amulet_name_to_id["Golden Bell"]: 4
   }
amulet_has_target = {
    amulet_name_to_id["Tribunal of Good and Evil"]: 1
}
amulet_trigger_ability_dict = {11: 2, 12: 3,
                               amulet_name_to_id["Staircase to Paradise"]: 9}
# amulet_countdown_list={4:3,5:1,6:2,7:2,8:2,9:3}
amulet_target_regulation = {}
amulet_cost_change_ability_list = {}
amulet_earth_rite_list = []
amulet_enhance_list = {amulet_name_to_id["Staircase to Paradise"]: [5]}
amulet_enhance_target_list = {}
amulet_enhance_target_regulation_list = {}

amulet_accelerate_list = {}
amulet_accelerate_card_id_list = {}
amulet_accelerate_target_list = {}
amulet_accelerate_target_regulation_list = {}

amulet_active_ability_card_id_list = {}

amulet_active_ability_list = {}
class_card_list = {}
for i in range(9):
    class_card_list[i] = {"Creature": {}, "Spell": {}, "Amulet": {}}
for i in tuple(creature_list):
    class_num = creature_list[i][4][0]
    class_card_list[class_num]["Creature"][i] = creature_list[i]
for i in tuple(spell_list):
    class_num = spell_list[i][1][0]
    class_card_list[class_num]["Spell"][i] = spell_list[i]
for i in tuple(amulet_list):
    class_num = amulet_list[i][2][0]
    class_card_list[class_num]["Amulet"][i] = amulet_list[i]
