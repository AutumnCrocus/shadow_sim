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
from numba import jit
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
creature_fanfare_ability = tsv_2_ability_dict("All_fanfare_list.tsv", name_to_id=creature_name_to_id)
creature_lastword_ability = tsv_2_ability_dict("All_lastword_list.tsv", name_to_id=creature_name_to_id)
creature_end_of_turn_ability = tsv_2_ability_dict("All_end_of_turn_list.tsv", name_to_id=creature_name_to_id)
creature_start_of_turn_ability = tsv_2_ability_dict("All_start_of_turn_list.tsv", name_to_id=creature_name_to_id)
creature_has_target = tsv_2_ability_dict("All_fanfare_target_list.tsv", name_to_id=creature_name_to_id)
creature_evo_effect = tsv_2_ability_dict("All_evo_effect_list.tsv", name_to_id=creature_name_to_id)
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
creature_trigger_ability_dict = {60: 1, 63: 4, 64: 5, 79: 6,
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


class Card:
    def __init__(self, card_id):
        assert False

    def get_copy(self):
        assert False

    def get_damage(self, amount):
        assert False

    def get_current_toughness(self):
        assert False

    def can_attack_to_follower(self):
        assert False

    def can_attack_to_player(self):
        assert False

    def can_be_attacked(self):
        assert False

    def __str__(self):
        assert False

    def untap(self):
        return

    def down_count(self, num=1, virtual=False):
        return


class Creature(Card):
    #@jit
    def __init__(self, card_id):
        self.time_stamp = 0
        self.card_id = card_id  # カードid
        self.card_category = "Creature"
        target_follower_data = creature_list[card_id]
        #self.cost = creature_list[card_id][0]  # カードのコスト
        #self.origin_cost = creature_list[card_id][0]  # カードの元々のコスト
        #self.power = creature_list[card_id][1]  # カードの攻撃力
        #self.toughness = creature_list[card_id][2]  # カードの体力
        #self.ability = creature_list[card_id][3][:]
        self.cost,self.power,self.toughness,_, card_data = target_follower_data[0:5]
        self.ability = target_follower_data[3][:]
        self.origin_cost = int(self.cost)
        self.card_class = LeaderClass(card_data[0])#LeaderClass(creature_list[card_id][4][0])
        self.trait = Trait(card_data[-1])
        if self.card_class.name == "RUNE":
            self.spell_boost = None
            #if creature_list[card_id][4][1][0]:
            if card_data[1][0]:
                self.spell_boost = 0
                self.cost_down = card_data[1][1]#creature_list[card_id][4][1][1]
        self.name = target_follower_data[-1]#creature_list[card_id][-1]

        self.buff = [0, 0]  # スタッツ上昇量
        self.until_turn_end_buff = [0, 0]  # ターン終了時までのスタッツ上昇量
        self.target_regulation = card_id
        #if card_id in creature_target_regulation:
        #    self.target_regulation = card_id#creature_target_regulation[card_id]
        self.player_attack_regulation = None
        if card_id in player_attack_regulation:
            self.player_attack_regulation = card_id#player_attack_regulation[card_id]
        self.evo_stat = evo_stats[special_evo_stats_id[card_id]] if card_id in special_evo_stats_id else [2,2]

        self.tmp_keyword_ability = [[[],[]],[[],[]]]
        self.have_active_ability = card_id in creature_active_ability_card_id_list
        if self.have_active_ability:
            func_id = creature_active_ability_card_id_list[card_id]
            self.func_id = func_id
            #self.active_ability_check_func = active_ability_check_func_list[func_id]
            self.active_ability = creature_active_ability_list[card_id]
        self.fanfare_ability = None
        if card_id in creature_fanfare_ability:
            self.fanfare_ability = creature_fanfare_ability[card_id]
            #creature_ability_dict[creature_fanfare_ability[card_id]]

        self.lastword_ability = [creature_lastword_ability[card_id]] if card_id in creature_lastword_ability else []
        #self.lastword_ability = []
        #if card_id in creature_lastword_ability:
        #    self.lastword_ability.append(creature_ability_dict[creature_lastword_ability[card_id]])

        self.have_target = 0
        if card_id in creature_has_target:
            self.have_target = creature_has_target[card_id]

        self.evo_effect = None
        self.evo_target = None
        if card_id in creature_evo_effect:
            self.evo_effect = creature_evo_effect[card_id]
            #creature_ability_dict[creature_evo_effect[card_id]]
            if card_id in creature_has_evo_effect_target:
                self.evo_target = creature_has_evo_effect_target[card_id]
            self.evo_target_regulation = None
            if card_id in evo_target_regulation:
                self.evo_target_regulation = evo_target_regulation[card_id]

        self.turn_start_ability = [creature_start_of_turn_ability[card_id]] if card_id in creature_start_of_turn_ability else []
        #self.turn_start_ability = []
        #if card_id in creature_start_of_turn_ability:
        #    self.turn_start_ability.append(creature_ability_dict[creature_start_of_turn_ability[card_id]])
        self.turn_end_ability = [creature_end_of_turn_ability[card_id]] if card_id in creature_end_of_turn_ability else []
        #self.turn_end_ability = []
        #if card_id in creature_end_of_turn_ability:
        #    self.turn_end_ability.append(creature_ability_dict[creature_end_of_turn_ability[card_id]])
        self.trigger_ability = [creature_trigger_ability_dict[card_id]] if card_id in creature_trigger_ability_dict else []
        self.trigger_ability_stack = []
        #self.trigger_ability = []
        #if card_id in creature_trigger_ability_dict:
        #    self.trigger_ability.append(trigger_ability_dict[creature_trigger_ability_dict[card_id]]())
        #    self.trigger_ability_stack = []

        #self.name = creature_list[card_id][-1]
        self.is_in_field = False
        self.is_in_graveyard = False
        self.damage = 0
        self.is_tapped = True
        self.can_attack_num = 1
        self.current_attack_num = 0
        self.evolved = False
        self.can_only_attack_target = None
        if card_id in creature_can_only_attack_list:
            self.can_only_attack_target = creature_can_only_attack_list[card_id]
        self.in_battle_ability = [creature_in_battle_ability_list[card_id]] if card_id in creature_in_battle_ability_list else []
        #self.in_battle_ability = []
        #if card_id in creature_in_battle_ability_list:
        #    self.in_battle_ability.append(battle_ability_dict[creature_in_battle_ability_list[card_id]])
        self.cost_change_ability = None
        if card_id in creature_cost_change_ability_list:
            self.cost_change_ability = cost_change_ability_dict[creature_cost_change_ability_list[card_id]]


        self.is_earth_rite = card_id in creature_earth_rite_list
        self.have_enhance = card_id in creature_enhance_list
        self.active_enhance_code = [False, 0]
        if self.have_enhance:
            self.enhance_cost = creature_enhance_list[card_id]
            self.active_enhance_code = [False, 0]
            self.enhance_target = 0
            self.enhance_target_regulation = None
            if card_id in creature_enhance_target_list:
                self.enhance_target = creature_enhance_target_list[card_id]
                if card_id in creature_enhance_target_regulation_list:
                    self.enhance_target_regulation = creature_enhance_target_regulation_list[card_id]

        self.have_accelerate = card_id in creature_accelerate_list
        self.active_accelerate_code = [False, 0]
        if self.have_accelerate:
            self.accelerate_cost = creature_accelerate_list[card_id]
            self.accelerate_card_id = creature_accelerate_card_id_list[card_id]
            self.active_accelerate_code = [False, 0]
            self.accelerate_target = 0
            self.accelerate_target_regulation = None
            if card_id in creature_accelerate_target_list:
                self.accelerate_target = creature_accelerate_target_list[card_id]
                if card_id in creature_accelerate_target_regulation_list:
                    self.accelerate_target_regulation = creature_accelerate_target_regulation_list[card_id]
        self.current_target = None
        return
    #@jit
    def get_copy(self):
        creature = Creature(self.card_id)
        creature.cost = int(self.cost)  # カードのコスト
        creature.power = int(self.power)  # カードの攻撃力
        creature.toughness = int(self.toughness)  # カードの体力
        creature.buff = self.buff[:]  # スタッツ上昇量
        creature.until_turn_end_buff = self.until_turn_end_buff[:]  # ターン終了時までのスタッツ上昇量
        creature.ability = list(self.ability[:])
        creature.tmp_keyword_ability = [[list(self.tmp_keyword_ability[0][0][:]),list(self.tmp_keyword_ability[0][1][:])],
                                        [list(self.tmp_keyword_ability[1][0][:]),list(self.tmp_keyword_ability[1][1][:])]]
        creature.lastword_ability = self.lastword_ability[:]
        """
        if len(creature.turn_start_ability) != len(self.turn_start_ability):
            creature.turn_start_ability = copy.deepcopy(self.turn_start_ability)
        if len(creature.turn_end_ability) != len(self.turn_end_ability):
            creature.turn_end_ability = copy.deepcopy(self.turn_end_ability)
        if len(creature.trigger_ability) != len(self.trigger_ability):
            creature.trigger_ability = copy.deepcopy(self.trigger_ability)
        """
        creature.turn_start_ability = self.turn_start_ability[:]
        creature.turn_end_ability = self.turn_end_ability[:]
        creature.trigger_ability = self.trigger_ability[:]

        creature.is_in_field = self.is_in_field
        creature.is_in_graveyard = self.is_in_graveyard
        creature.damage = int(self.damage)
        creature.is_tapped = self.is_tapped
        creature.current_attack_num = int(self.current_attack_num)
        creature.can_attack_num = int(self.can_attack_num)
        creature.evolved = self.evolved
        #if len(creature.in_battle_ability) != len(self.in_battle_ability):
        #    creature.in_battle_ability = copy.deepcopy(self.in_battle_ability)
        creature.in_battle_ability = self.in_battle_ability[:]
        if self.card_class.name == "RUNE":
            creature.spell_boost = None
            if creature_list[self.card_id][4][1][0]:
                creature.spell_boost = int(self.spell_boost)
                creature.cost_down = self.cost_down
        if self.current_target is not None:
            creature.current_target = int(self.current_target)
        return creature

    def untap(self):
        self.is_tapped = False
        # self.attacked_flg=False
        self.current_attack_num = 0

    def evolve(self, field, target, player_num=0, virtual=False, auto=False):
        self.evolved = True
        self.power += self.evo_stat[0]
        self.toughness += self.evo_stat[1]
        if auto: return
        if self.evo_effect is not None:
            creature_ability_dict[self.evo_effect](field, field.players[player_num], field.players[1 - player_num], virtual, target, self)
            #self.evo_effect(field, field.players[player_num], field.players[1 - player_num], virtual, target, self)

    def get_damage(self, amount):
        if KeywordAbility.REDUCE_DAMAGE_TO_ZERO.value not in self.ability:
            self.damage += amount
            if self.toughness - self.damage <= 0:
                self.is_in_field = False
                self.is_in_graveyard = True
            return amount
        else:
            return 0

    def restore_toughness(self, amount):
        tmp = int(self.damage)
        self.damage = max(0, self.damage - amount)
        return tmp - self.damage

    def get_current_toughness(self):
        return self.toughness - self.damage

    def can_attack_to_follower(self):
        if KeywordAbility.CANT_ATTACK.value in self.ability:
            return False
        if self.current_attack_num >= self.can_attack_num:
            return False
        if self.evolved: return True
        if any((i in self.ability) for i in [KeywordAbility.STORM.value, KeywordAbility.RUSH.value]): return True
        if not self.is_tapped: return True
        return False

    def can_attack_to_player(self):
        if KeywordAbility.CANT_ATTACK.value in self.ability:
            return False
        if self.current_attack_num >= self.can_attack_num:
            return False
        if KeywordAbility.CANT_ATTACK_TO_PLAYER.value in self.ability: return False
        if KeywordAbility.STORM.value in self.ability: return True
        if not self.is_tapped: return True
        return False

    def can_be_targeted(self):
        if KeywordAbility.CANT_BE_TARGETED.value in self.ability: return False
        if KeywordAbility.AMBUSH.value in self.ability: return False
        return True

    def can_be_attacked(self):
        if KeywordAbility.CANT_BE_ATTACKED.value in self.ability: return False
        if KeywordAbility.AMBUSH.value in self.ability: return False
        return True

    def get_active_ability(self):
        assert self.have_active_ability, "invalid acitve ability error"
        for ability_id in self.active_ability:
            self.ability.append(ability_id)
        self.ability = list(set(self.ability))

    def lose_active_ability(self):
        assert self.have_active_ability, "invalid acitve ability error"
        for ability_id in self.active_ability:
            if ability_id in self.ability:
                self.ability.remove(ability_id)

    def eq(self, other):
        """
        :param other:
        :return:
        """
        if other is None:
            return False
        if self.name != other.name:
            return False
        if self.power != other.power:
            return False
        if self.get_current_toughness() != other.get_current_toughness():
            return False
        if self.cost != other.cost:
            return False
        if self.evolved != other.evolved:
            return False
        if self.ability != other.ability:
            return False
        if self.can_be_attacked() != other.can_be_attacked():
            return False
        if self.can_be_targeted() != other.can_be_targeted():
            return False
        if self.current_attack_num != other.current_attack_num or self.can_attack_num != other.can_attack_num:
            return False
        if self.can_attack_to_follower() != other.can_attack_to_follower():
            return False
        if self.can_attack_to_player() != other.can_attack_to_player():
            return False
        if self.tmp_keyword_ability != other.tmp_keyword_ability:
            return False

        return True

    def __str__(self):
        text = ""
        default_color = "\033[0m"
        if self.is_in_field:
            if self.can_attack_to_player():
                CYAN = "\033[36m"
                text += CYAN
                default_color = CYAN
            elif self.can_attack_to_follower():
                YELLOW = "\033[33m"
                text += YELLOW
                default_color = YELLOW
        if self.is_in_field:
            text += "name:{:<25} {}/{}/{}".format(self.name, str(self.origin_cost), str(self.power),
                                                  str(self.toughness - self.damage))
        else:
            text += "name:{:<25} {}/{}/{}".format(self.name, str(self.cost), str(self.power), str(self.toughness))
            if self.have_enhance and self.active_enhance_code[0]:
                text += " enhance:{}".format(self.active_enhance_code[1])
            elif self.have_accelerate and self.active_accelerate_code[0]:
                text += " accelerate:{}".format(self.active_accelerate_code[1])
        if self.card_class.name == "RUNE" and self.spell_boost is not None and self.is_in_field == False:
            text += " spell_boost:{:<2}".format(self.spell_boost)
        if self.ability != [] and self.is_in_field:
            text += " ability={}".format([KeywordAbility(i).name for i in self.ability])
        if self.is_in_field:
            if len(self.lastword_ability) > 0:
                RED = '\033[31m'
                text += RED + " ■"
            if len(self.trigger_ability) > 0 or len(self.in_battle_ability) > 0:
                GREEN = '\033[32m'
                text += GREEN + " ◆"
        text += "\033[0m"
        return text


class Spell(Card):
    def __init__(self, card_id):
        self.time_stamp = 0
        self.card_id = card_id  # カードid
        self.card_category = "Spell"
        self.cost = spell_list[self.card_id][0]  # カードのコスト
        self.origin_cost = spell_list[self.card_id][0]  # カードの元々のコスト
        self.target_regulation = card_id#None
        #if card_id in spell_target_regulation:
        #    self.target_regulation = card_id#spell_target_regulation[card_id]

        self.triggered_ability = [spell_ability_dict[spell_triggered_ability[card_id]]]

        self.have_target = 0
        if card_id in spell_has_target:
            self.have_target = spell_has_target[card_id]
        self.name = spell_list[self.card_id][-1]
        self.is_in_graveyard = False
        self.cost_change_ability = None
        if card_id in spell_cost_change_ability_list:
            self.cost_change_ability = cost_change_ability_dict[spell_cost_change_ability_list[card_id]]

        self.card_class = LeaderClass(spell_list[card_id][1][0])
        self.trait = Trait(spell_list[card_id][1][-1])
        if self.card_class.name == "RUNE":
            self.spell_boost = None
            if spell_list[card_id][1][1][0]:
                self.spell_boost = 0
                self.cost_down = spell_list[card_id][1][1][1]

        self.is_earth_rite = card_id in spell_earth_rite_list
        self.have_enhance = card_id in spell_enhance_list
        self.active_enhance_code = [False, 0]
        if self.have_enhance:
            self.enhance_cost = spell_enhance_list[card_id]
            self.active_enhance_code = [False, 0]
            self.enhance_target = 0
            self.enhance_target_regulation = None
            if card_id in spell_enhance_target_list:
                self.enhance_target = spell_enhance_target_list[card_id]
                if card_id in spell_enhance_target_regulation_list:
                    self.enhance_target_regulation = spell_enhance_target_regulation_list[card_id]

        self.have_accelerate = card_id in spell_accelerate_list
        self.active_accelerate_code = [False, 0]
        if self.have_accelerate:
            self.accelerate_cost = spell_accelerate_list[card_id]
            self.accelerate_card_id = spell_accelerate_card_id_list[card_id]
            self.active_accelerate_code = [False, 0]
            self.accelerate_target = 0
            self.accelerate_target_regulation = None
            if card_id in spell_accelerate_target_list:
                self.accelerate_target = spell_accelerate_target_list[card_id]
                if card_id in spell_accelerate_target_regulation_list:
                    self.accelerate_target_regulation = spell_accelerate_target_regulation_list[card_id]
        self.current_target = None

    def get_copy(self):
        spell = Spell(self.card_id)
        spell.cost = int(self.cost)
        if self.card_class.name == "RUNE":
            spell.spell_boost = None
            if spell_list[self.card_id][1][1][0]:
                spell.spell_boost = int(self.spell_boost)
                spell.cost_down = self.cost_down
        if self.current_target is not None:
            spell.current_target = int(self.current_target)
        return spell

    def can_be_attacked(self):
        mylogger.info("card_name:{}".format(self.name))
        assert False

    def eq(self, other):
        """
        :param other:
        :return:
        """
        if other is None:
            return False

        if self.name != other.name:
            return False
        if self.cost != other.cost:
            return False
        if self.card_class.name == "RUNE":
            if self.spell_boost != other.spell_boost:
                return False

        return True

    def __str__(self):
        text = "name:" + '{:<25}'.format(self.name) + " cost: " + '{:<2}'.format(str(self.cost))
        if self.card_class.name == "RUNE" and self.spell_boost is not None:
            text += " spell_boost:{:<2}".format(self.spell_boost)
        if self.have_enhance  and self.active_enhance_code[0]:
            text += " enhance:{}".format(self.active_enhance_code[1])
        return text


class Amulet(Card):
    def __init__(self, card_id):
        self.time_stamp = 0
        self.card_id = card_id  # カードid
        self.card_category = "Amulet"
        self.cost = amulet_list[self.card_id][0]  # カードのコスト
        self.origin_cost = amulet_list[self.card_id][0]  # カードの元々のコスト
        self.can_not_be_targeted = 6 in amulet_list[card_id][1]  # 能力の対象にならないを持つか
        self.ability = amulet_list[card_id][1][:]
        self.have_active_ability = card_id in amulet_active_ability_card_id_list
        if self.have_active_ability:
            func_id = amulet_active_ability_card_id_list[card_id]
            self.active_ability_check_func = active_ability_check_func_list[func_id]
            self.active_ability = amulet_active_ability_list[card_id]
        self.trigger_ability = [amulet_trigger_ability_dict[card_id]] if card_id in amulet_trigger_ability_dict else []
        self.trigger_ability_stack = []
        #self.trigger_ability = []
        #if card_id in amulet_trigger_ability_dict:
        #    self.trigger_ability.append(trigger_ability_dict[amulet_trigger_ability_dict[card_id]]())
        #    self.trigger_ability_stack = []
        self.target_regulation = card_id#None
        #if card_id in amulet_target_regulation:
        #    self.target_regulation = amulet_target_regulation[card_id]

        self.fanfare_ability = None
        if card_id in amulet_fanfare_ability:
            self.fanfare_ability = amulet_fanfare_ability[card_id]
            #amulet_ability_dict[amulet_fanfare_ability[card_id]]
        self.lastword_ability = [amulet_lastword_ability[card_id]] if card_id in amulet_lastword_ability else []
        #self.lastword_ability = []
        #if card_id in amulet_lastword_ability:
        #    self.lastword_ability.append(amulet_ability_dict[amulet_lastword_ability[card_id]])

        self.have_target = 0
        if card_id in amulet_has_target:
            self.have_target = amulet_has_target[card_id]

        #self.turn_start_ability = []
        #if card_id in amulet_start_of_turn_ability:
        #    self.turn_start_ability.append(amulet_ability_dict[amulet_start_of_turn_ability[card_id]])
        #    # mylogger.info("ability exist")
        self.turn_start_ability = [amulet_start_of_turn_ability[card_id]] if card_id in amulet_start_of_turn_ability else []

        #self.turn_end_ability = []
        #if card_id in amulet_end_of_turn_ability:
        #    self.turn_end_ability.append(amulet_ability_dict[amulet_end_of_turn_ability[card_id]])
        #    # mylogger.info("ability exist")
        self.turn_end_ability = [amulet_end_of_turn_ability[card_id]] if card_id in amulet_end_of_turn_ability else []
        self.name = amulet_list[self.card_id][-1]
        self.is_in_graveyard = False
        self.is_in_field = False
        self.countdown = False
        self.ini_count = 0
        self.current_count = 0
        if amulet_list[card_id][3]:
            self.countdown = True
            self.ini_count = amulet_list[card_id][3]
            self.current_count = amulet_list[card_id][3]
        self.cost_change_ability = None
        if card_id in amulet_cost_change_ability_list:
            self.cost_change_ability = cost_change_ability_dict[amulet_cost_change_ability_list[card_id]]

        self.card_class = LeaderClass(amulet_list[card_id][2][0])
        self.trait = Trait(amulet_list[card_id][2][-1])
        if self.card_class.name == "RUNE":
            self.spell_boost = None
            if amulet_list[card_id][2][1][0]:
                self.spell_boost = 0
                self.cost_down = amulet_list[card_id][2][1][1]

        self.is_earth_sigil = self.trait.name == "EARTH_SIGIL"
        self.is_earth_rite = card_id in amulet_earth_rite_list

        self.have_enhance = card_id in amulet_enhance_list
        self.active_enhance_code = [False, 0]
        if self.have_enhance:
            self.enhance_cost = amulet_enhance_list[card_id]
            self.active_enhance_code = [False, 0]
            self.enhance_target = 0
            self.enhance_target_regulation = None
            if card_id in amulet_enhance_target_list:
                self.enhance_target = amulet_enhance_target_list[card_id]
                if card_id in amulet_enhance_target_regulation_list:
                    self.enhance_target_regulation = amulet_enhance_target_regulation_list[card_id]

        self.have_accelerate = card_id in amulet_accelerate_list
        self.active_accelerate_code = [False, 0]
        if self.have_accelerate:
            self.accelerate_cost = amulet_accelerate_list[card_id]
            self.accelerate_card_id = amulet_accelerate_card_id_list[card_id]
            self.active_accelerate_code = [False, 0]
            self.accelerate_target = 0
            self.accelerate_target_regulation = None
            if card_id in amulet_accelerate_target_list:
                self.accelerate_target = amulet_accelerate_target_list[card_id]
                if card_id in amulet_accelerate_target_regulation_list:
                    self.accelerate_target_regulation = amulet_accelerate_target_regulation_list[card_id]
        self.current_target = None

    def get_copy(self):
        amulet = Amulet(self.card_id)
        amulet.cost = int(self.cost)
        amulet.is_in_field = self.is_in_field
        amulet.current_count = int(self.current_count)
        amulet.turn_start_ability = self.turn_start_ability[:]
        amulet.turn_end_ability = self.turn_end_ability[:]
        amulet.trigger_ability = self.trigger_ability[:]
        if self.card_class.name == "RUNE":
            # self.spell_boost = None
            if amulet_list[self.card_id][2][1][0]:
                amulet.spell_boost = int(self.spell_boost)
                amulet.cost_down = self.cost_down
        if self.current_target is not None:
            amulet.current_target = int(self.current_target)
        return amulet

    def can_be_attacked(self):
        mylogger.info("card_name:{}".format(self.name))
        assert False

    def can_be_targeted(self):
        if KeywordAbility.CANT_BE_TARGETED.value in self.ability: return False
        if KeywordAbility.AMBUSH.value in self.ability: return False
        return True
        # return not any(i in self.ability for i in [KeywordAbility.CANT_BE_TARGETED.value,KeywordAbility.AMBUSH.value])

    def down_count(self, num=1, virtual=False):
        if not self.countdown: return
        self.current_count -= num
        if not virtual: mylogger.info("{}'s count down by {}".format(self.name, num))
        if self.current_count <= 0:
            self.is_in_graveyard = True
            self.is_in_field = False
            self.current_count = self.ini_count

    def eq(self, other):
        """
        :param other:
        :return:
        """
        if other is None:
            return False
        if self.name != other.name:
            return False
        if self.cost != other.cost:
            return False
        if self.countdown != other.countdown:
            return False
        if self.countdown:
            if self.current_count != other.current_count:
                return False
        if self.ability != other.ability:
            return False

        return True

    def __str__(self):
        tmp = ""
        if not self.is_in_field:
            tmp = "name:" + '{:<25}'.format(self.name) + " cost: " + '{:<2}'.format(str(self.cost))
        else:
            tmp = "name:" + '{:<25}'.format(self.name) + " cost: " + '{:<2}'.format(str(self.origin_cost))

        if self.countdown:
            tmp = tmp + " count:{:<2}".format(self.current_count)
        if self.have_enhance and self.active_enhance_code[0]:
            tmp += " enhance:{}".format(self.active_enhance_code[1])
        if self.is_in_field:
            if len(self.lastword_ability) > 0:
                RED = '\033[31m'
                tmp += RED + " ◇"
            if len(self.trigger_ability) > 0:
                GREEN = '\033[32m'
                tmp += GREEN + " ◆"
        tmp += "\033[0m"
        return tmp


class Deck:
    def __init__(self):
        self.deck = deque()
        self.remain_num = 0
        self.mean_cost = 0
        self.deck_type = None
        self.leader_class = None
    #@jit(nopython=True)
    def append(self, card, num=1):
        self.remain_num += num
        for i in range(num):
            self.deck.append(card.get_copy())

    def show_all(self):
        print("Deck contents")
        print("==============================================")
        for i in range(self.remain_num):
            print(self.deck[self.remain_num - i - 1])  # 引くのはlistの最後尾から
        print("==============================================")

    def draw(self):
        self.remain_num -= 1
        return self.deck.pop()

    def shuffle(self):
        random.shuffle(self.deck)

    def get_mean_cost(self):
        sum_of_cost = 0
        for card in self.deck:
            sum_of_cost += card.origin_cost
        return sum_of_cost / len(self.deck)

    def set_deck_type(self, type_num):
        self.deck_type = DeckType(type_num).value
        #mylogger.info("Deck_Type:{}".format(self.deck_type.name))
        # 1はAggro,2はMid,3はControl,4はCombo

    def set_leader_class(self, leader_class):
        assert leader_class in LeaderClass.__members__, "invalid class name!"
        self.leader_class = LeaderClass[leader_class]

    def get_name_set(self):
        name_list = {}
        for card in self.deck:
            if card.name not in name_list:
                name_list[card.name] = {"used_num": 0, "sum_of_turn_when_used": 0, "win_num": 0, "drawn_num": 0,
                                        "win_num_when_drawn": 0}
                if card.have_accelerate:
                    for accelerate_cost in card.accelerate_cost:
                        name_list["{}(Accelerate {})".format(card.name, accelerate_cost)] = \
                            {"used_num": 0, "sum_of_turn_when_used": 0, "win_num": 0, "drawn_num": 0,
                             "win_num_when_drawn": 0}

        return name_list

    def get_remain_card_set(self):
        remain_card_set = {}
        for card in self.deck:
            if card.name not in remain_card_set:
                remain_card_set[card.name] = 0
            remain_card_set[card.name] += 1

        return remain_card_set

    def show_remain_card_set(self):
        remain_card_set = self.get_remain_card_set()
        print("remain_cards_in_deck")
        for key in sorted(tuple(remain_card_set.keys())):
            print("{}:{}".format(key, remain_card_set[key]))
        print("")
    """
    def get_cost_histgram(self):
        histgram_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        for card in self.deck:
            if card.origin_cost <= 1:
                histgram_dict[1] += 1
            elif card.origin_cost >= 8:
                histgram_dict[8] += 1
            else:
                histgram_dict[card.cost] += 1
        histgram = list(histgram_dict.values())
        height_histgram = ["" for i in range(max(histgram))]
        for i in range(max(histgram)):
            for j in range(len(histgram)):
                if histgram[j] >= i:
                    height_histgram[i] += " ■ "
                else:
                    height_histgram[i] += "    "

        for i in range(len(height_histgram)):
            print(height_histgram[len(height_histgram) - 1 - i])

        print("~1   2   3   4   5   6   7   8+")
    """
