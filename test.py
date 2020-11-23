# -*- coding: utf-8 -*-
# +
from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock

#     try:
#         set_start_method('spawn')
#         print("spawn is run.")
#         #set_start_method('fork') GPU使用時CUDA initializationでerror
#         #print('fork')
#     except RuntimeError:
#         pass
#import random
#random.seed(247165)
import sys
import numpy as np
import random
import math
import copy
from card_setting import *
from Field_setting import Field
from Player_setting import *
from Game_setting import Game
from my_moduler import get_module_logger#, get_state_logger
from mulligan_setting import *
import logging
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
mylogger = get_module_logger(__name__)
from my_enum import *
import argparse
import csv
from tqdm import tqdm
value_history = []
# -



def tsv_to_deck(tsv_name):
    deck = Deck()
    with open("Deck_TSV/" + tsv_name) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            card_category = row[0]
            # mylogger.info(row)
            if card_category == "Creature":
                deck.append(Creature(creature_name_to_id[row[1]]), num=int(row[2]))
            elif card_category == "Spell":
                deck.append(Spell(spell_name_to_id[row[1]]), num=int(row[2]))
            elif card_category == "Amulet":
                deck.append(Amulet(amulet_name_to_id[row[1]]), num=int(row[2]))
            else:
                assert False, "{} {}".format(card_category)
    return deck


def game_play(Player1, Player2, D1, D2, win, lose, lib_num, virtual_flg=False, deck_name_list=None):
    assert Player1.player_num != Player2.player_num, "same error"
    f = Field(5)
    f.players[0] = Player1
    f.players[0].field = f
    f.players[1] = Player2
    f.players[1].field = f
    f.players[0].deck = Deck()
    f.players[0].deck.set_leader_class(D1.leader_class.name)
    f.players[0].deck.set_deck_type(D1.deck_type)
    for card in D1.deck:
        f.players[0].deck.deck.append(card.get_copy())
    f.players[1].deck = Deck()
    f.players[1].deck.set_leader_class(D2.leader_class.name)
    f.players[1].deck.set_deck_type(D2.deck_type)
    for card in D2.deck:
        f.players[1].deck.deck.append(card.get_copy())

    f.players[0].deck.shuffle()
    f.players[1].deck.shuffle()
    f.players[0].draw(f.players[0].deck, 3)
    f.players[1].draw(f.players[1].deck, 3)
    G = Game()
    #virtual_flg=False
    print("virtual:{}".format(virtual_flg))
    (w, l, lib, turn) = G.start(f, virtual_flg=virtual_flg)
    win += w
    lose += l
    first = w
    lib_num += lib
    if not virtual_flg:
        mylogger.info("Game end")
        mylogger.info("Player1 life:{} Player2 life:{}".format(f.players[0].life, f.players[1].life))
        f.show_field()
    player1_win_turn = False
    player2_win_turn = False
    if w == 1:
        player1_win_turn = turn
    else:
        player2_win_turn = turn

    f.players[0].life = 20
    f.players[0].hand.clear()
    f.players[0].deck = None
    f.players[0].lib_out_flg = False

    f.players[1].life = 20
    f.players[1].hand.clear()
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
    if f.players[0].policy.policy_type == 3:
        f.players[0].policy.current_node = None
    if f.players[1].policy.policy_type == 3:
        f.players[1].policy.current_node = None
    if not virtual_flg:
        f.graveyard.show_graveyard()
        f.play_cards.show_play_list()
    if deck_name_list is not None:
        f.play_cards.play_cards_set()
        win_flg = [w, l]

        for i in range(2):
            for cost_key in list(f.play_cards.name_list[i].keys()):
                for category_key in list(f.play_cards.name_list[i][cost_key].keys()):
                    for name_key in list(f.play_cards.name_list[i][cost_key][category_key].keys()):
                        if name_key in deck_name_list[f.players[i].name]:
                            deck_name_list[f.players[i].name][name_key]["used_num"] += 1
                            turn_list = f.play_cards.played_turn_dict[i][name_key]
                            deck_name_list[f.players[i].name][name_key]["sum_of_turn_when_used"] \
                                += sum(turn_list) / len(turn_list)
                            deck_name_list[f.players[i].name][name_key]["win_num"] += win_flg[i]
            for card_name in f.drawn_cards.name_list[i]:
                if card_name in deck_name_list[f.players[i].name]:
                    deck_name_list[f.players[i].name][card_name]["drawn_num"] += 1
                    deck_name_list[f.players[i].name][card_name]["win_num_when_drawn"] += win_flg[i]
    return win, lose, lib_num, turn, first, (player1_win_turn, player2_win_turn)


def demo_game_play(Player1, Player2, D1, D2, win, lose, lib_num, virtual_flg=False, deck_name_list=None,
                   history_flg=False):
    assert Player1.player_num != Player2.player_num, "same error"
    f = Field(5)
    f.players[0] = Player1
    f.players[0].field = f
    f.players[1] = Player2
    f.players[1].field = f
    f.players[0].deck = Deck()
    f.players[0].deck.set_leader_class(D1.leader_class.name)
    f.players[0].deck.set_deck_type(D1.deck_type)
    for card in D1.deck:
        f.players[0].deck.deck.append(card.get_copy())
    f.players[1].deck = Deck()
    f.players[1].deck.set_leader_class(D2.leader_class.name)
    f.players[1].deck.set_deck_type(D2.deck_type)
    for card in D2.deck:
        f.players[1].deck.deck.append(card.get_copy())
    G = Game()
    f.players[0].deck.shuffle()
    f.players[1].deck.shuffle()
    f.players[0].draw(f.players[0].deck, 3)
    f.players[1].draw(f.players[1].deck, 3)
    # f.evo_point = [1,2]

    if history_flg:
        (w, l, lib, turn, history) = G.start(f, virtual_flg=virtual_flg, history_flg=history_flg)
        value_history.append(history)
    else:
        (w, l, lib, turn) = G.start(f, virtual_flg=virtual_flg)
    win += w
    lose += l
    first = w
    lib_num += lib
    if not virtual_flg:
        mylogger.info("Game end")
        mylogger.info("Player1 life:{} Player2 life:{}".format(f.players[0].life, f.players[1].life))
        f.show_field()
    player1_win_turn = False
    player2_win_turn = False
    if w == 1:
        player1_win_turn = turn
    else:
        player2_win_turn = turn

    f.players[0].life = 20
    f.players[0].hand.clear()
    f.players[0].deck = None
    f.players[0].lib_out_flg = False
    f.players[0].effect.clear()
    f.players[1].life = 20
    f.players[1].hand.clear()
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
    f.players[1].effect.clear()
    if f.players[0].policy.policy_type == 3:
        f.players[0].policy.current_node = None
    if f.players[1].policy.policy_type == 3:
        f.players[1].policy.current_node = None

    if deck_name_list is not None:
        f.play_cards.play_cards_set()
        win_flg = [w, l]

        for i in range(2):
            for cost_key in list(f.play_cards.name_list[i].keys()):
                for category_key in list(f.play_cards.name_list[i][cost_key].keys()):
                    for name_key in list(f.play_cards.name_list[i][cost_key][category_key].keys()):
                        if name_key in deck_name_list[f.players[i].name]:
                            deck_name_list[f.players[i].name][name_key]["used_num"] += 1
                            turn_list = f.play_cards.played_turn_dict[i][name_key]
                            deck_name_list[f.players[i].name][name_key]["sum_of_turn_when_used"] \
                                += sum(turn_list) / len(turn_list)
                            deck_name_list[f.players[i].name][name_key]["win_num"] += win_flg[i]
            for card_name in f.drawn_cards.name_list[i]:
                if card_name in deck_name_list[f.players[i].name]:
                    deck_name_list[f.players[i].name][card_name]["drawn_num"] += 1
                    deck_name_list[f.players[i].name][card_name]["win_num_when_drawn"] += win_flg[i]

        # mylogger.info(deck_name_list)

    return win, lose, lib_num, turn, first, (player1_win_turn, player2_win_turn)


def demo_game_play_with_pairwise(Player1, Player2, D1, D2, win, lose, lib_num, virtual_flg=False,deck_name_list=None,pairwise_dict=None):
    assert Player1.player_num != Player2.player_num, "same error"
    f = Field(5)
    f.players[0] = Player1
    f.players[0].field = f
    f.players[1] = Player2
    f.players[1].field = f
    f.players[0].deck = Deck()
    f.players[0].deck.set_leader_class(D1.leader_class.name)
    f.players[0].deck.set_deck_type(D1.deck_type)
    for card in D1.deck:
        f.players[0].deck.deck.append(card.get_copy())
    f.players[1].deck = Deck()
    f.players[1].deck.set_leader_class(D2.leader_class.name)
    f.players[1].deck.set_deck_type(D2.deck_type)
    for card in D2.deck:
        f.players[1].deck.deck.append(card.get_copy())

    f.players[0].deck.shuffle()
    f.players[1].deck.shuffle()
    f.players[0].draw(f.players[0].deck, 3)
    f.players[1].draw(f.players[1].deck, 3)
    # f.evo_point = [1,2]
    G = Game()
    (w, l, lib, turn) = G.start(f, virtual_flg=virtual_flg)
    win += w
    lose += l
    first = w
    lib_num += lib
    if not virtual_flg:
        mylogger.info("Game end")
        mylogger.info("Player1 life:{} Player2 life:{}".format(f.players[0].life, f.players[1].life))
        f.show_field()
    player1_win_turn = False
    player2_win_turn = False
    if w == 1:
        player1_win_turn = turn
    else:
        player2_win_turn = turn

    f.players[0].life = 20
    f.players[0].hand.clear()
    f.players[0].deck = None
    f.players[0].lib_out_flg = False
    f.players[0].effect.clear()
    f.players[1].life = 20
    f.players[1].hand.clear()
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
    f.players[1].effect.clear()
    f.play_cards.plain_play_cards_set()
    if f.players[0].policy.policy_type == 3:
        f.players[0].policy.current_node = None
    if f.players[1].policy.policy_type == 3:
        f.players[1].policy.current_node = None
    win_flg = [w, l]
    #for i in range(2):
    #    sorted_name_list = sorted(list(f.play_cards.plain_name_list[i].keys()))
    #    for key in list(sorted_name_list.keys()):
    #        if key in deck_name_list[f.players[i].name]:
    #            deck_name_list[key]["used_num"] += 1
    #            deck_name_list[key]["win_num"] += win_flg[i]

    for i in range(2):
        sorted_name_list = sorted(list(f.play_cards.plain_name_list[i].keys()))
        for first_id,first_card_name in enumerate(sorted_name_list):
            for second_id,second_card_name in enumerate(sorted_name_list,start=first_id+1):
                if (first_card_name,second_card_name) in pairwise_dict[f.players[i].name]:
                    target_pair_wise = pairwise_dict[f.players[i].name][(first_card_name,second_card_name)]
                    target_pair_wise["both_played_num"] += 1
                    target_pair_wise["win_num_when_both_played"] += win_flg[i]
        if deck_name_list is not None:
            for key in sorted_name_list:
                if key in deck_name_list[f.players[i].name]:
                    deck_name_list[f.players[i].name][key]["used_num"] += 1
                    deck_name_list[f.players[i].name][key]["win_num"] += win_flg[i]



    return win, lose, lib_num, turn, first, (player1_win_turn, player2_win_turn)

def execute_demo(Player_1, Player_2, iteration, virtual_flg=False, deck_type=None,graph=False):
    #Player1 = copy.deepcopy(Player_1)
    #Player2 = copy.deepcopy(Player_2)

    mylogger.info("d1:{}".format(Player_1.policy.name))
    mylogger.info("d2:{}".format(Player_2.policy.name))
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = [Deck(), Deck()]
    if deck_type is None:
        deck_type = [5, 5]
    else:
        mylogger.info("deck_type:{}".format(deck_type))
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic", -2: "Sword_Basic",
                      -3: "Rune_Basic",
                      -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
                      100: "Test",
                      -9: "Spell-Rune",11:"PtP-Forest",12:"Mid-Shadow",13:"Neutral-Blood",100:"TEST"}

    key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                      3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                      6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                      9: ["SpellBoost-Rune.tsv", "RUNE"],10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                      11: ["PtP_Forest.tsv", "FOREST"],12: ["Mid_Shadow.tsv", "SHADOW"],
                      13: ["Neutral_Blood.tsv", "BLOOD"],100: ["TEST.tsv", "SHADOW"]}
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    class_pool = [0, 0]
    for i, d in enumerate(D):
        if deck_type[i] in key_2_tsv_name:
            D[i] = tsv_to_deck(key_2_tsv_name[deck_type[i]][0])
            D[i].set_leader_class(key_2_tsv_name[deck_type[i]][1])
            D[i].set_deck_type(Embedd_Network_model.deck_id_2_deck_type(deck_type[i]))
            continue

        if deck_type[i] == -1:
            D[i] = tsv_to_deck("Forest_Basic.tsv")
            D[i].set_leader_class("FOREST")
        elif deck_type[i] == -2:
            D[i] = tsv_to_deck("Sword_Basic.tsv")
            D[i].set_leader_class("SWORD")
        elif deck_type[i] == -3:
            D[i] = tsv_to_deck("Rune_Basic.tsv")
            D[i].set_leader_class("RUNE")
        elif deck_type[i] == -4:
            D[i] = tsv_to_deck("Dragon_Basic.tsv")
            D[i].set_leader_class("DRAGON")
        elif deck_type[i] == -5:
            D[i] = tsv_to_deck("Shadow_Basic.tsv")
            D[i].set_leader_class("SHADOW")
        elif deck_type[i] == -6:
            D[i] = tsv_to_deck("Blood_Basic.tsv")
            D[i].set_leader_class("BLOOD")
        elif deck_type[i] == -7:
            D[i] = tsv_to_deck("Haven_Basic.tsv")
            D[i].set_leader_class("HAVEN")
        elif deck_type[i] == -8:
            D[i] = tsv_to_deck("Portal_Basic.tsv")
            D[i].set_leader_class("PORTAL")
        elif deck_type[i] == -9:
            D[i] = tsv_to_deck("SpellBoost-Rune.tsv")
            D[i].set_leader_class("RUNE")
        elif deck_type[i] == -10:
            D[i] = tsv_to_deck("Test-Haven.tsv")
            D[i].set_leader_class("HAVEN")
        """
        elif deck_type[i] == 100:

            D[i].set_leader_class("NEUTRAL")
            D[i].append(Creature(creature_name_to_id["Goblin"]), num=4)
            D[i].append(Creature(creature_name_to_id["Fighter"]), num=6)
            D[i].append(Creature(creature_name_to_id["Unicorn Dancer Unica"]), num=3)
            D[i].append(Creature(creature_name_to_id["Ax Fighter"]), num=3)
            D[i].append(Creature(creature_name_to_id["Mercenary Drifter"]), num=3)
            D[i].append(Creature(creature_name_to_id["Healing Angel"]), num=3)
            D[i].append(Creature(creature_name_to_id["Shield Angel"]), num=3)
            D[i].append(Creature(creature_name_to_id["Golyat"]), num=3)
            D[i].append(Creature(creature_name_to_id["Angelic Sword Maiden"]), num=9)
            D[i].append(Creature(creature_name_to_id["Gilgamesh"]), num=3)
        """

    Player1.class_num = class_pool[0]
    Player2.class_num = class_pool[1]
    mylogger.info("Alice's deck mean cost:{:<4}".format(D[0].get_mean_cost()))
    mylogger.info("Bob's deck mean cost:{:<4}".format(D[1].get_mean_cost()))
    D[0].mean_cost = D[0].get_mean_cost()
    D[1].mean_cost = D[1].get_mean_cost()
    mylogger.info("deck detail")
    D[0].show_remain_card_set()
    D[1].show_remain_card_set()
    mylogger.info("")
    assert len(D[0].deck) == 40 and len(D[1].deck) == 40, "deck_len:{},{}" \
        .format(len(D[0].deck), len(D[1].deck))
    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]
    win_lose = [win, lose]
    epoc_len = span
    epoc_win_lose = [0, 0]
    epoc_lib_num = 0
    first_num = [0, 0]
    epoch_first_num = [0,0]
    win_turns = [0, 0]
    deck_name_list = {"Alice": D[0].get_name_set(), "Bob": D[1].get_name_set()}
    for i in range(iteration):
        if (i + 1) % epoc_len == 1 or epoc_len == 1:
            epoc_win_lose = [int(win_lose[0]), int(win_lose[1])]
            epoc_lib_num = int(lib_num)
            epoch_first_num = [int(first_num[0]),int(first_num[1])]
        if not virtual_flg:
            mylogger.info("Game {}".format(i + 1))
        # mylogger.info("name:{}".format(Turn_Players[i%2].name))
        Turn_Players[i % 2].is_first = True
        Turn_Players[i % 2].player_num = 0
        Turn_Players[(i + 1) % 2].is_first = False
        Turn_Players[(i + 1) % 2].player_num = 1
        assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
            Turn_Players[0].player_num)
        (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, (player1_win_turn, player2_win_turn)) \
            = demo_game_play(Turn_Players[i % 2],
                             Turn_Players[(i + 1) % 2],
                             D[i % 2], D[(i + 1) % 2],
                             win_lose[i % 2],
                             win_lose[(i + 1) % 2], lib_num,
                             virtual_flg=virtual_flg, deck_name_list=deck_name_list, history_flg=graph)
        first_num[i % 2] += first
        sum_of_turn += end_turn
        if player1_win_turn is not False:
            win_turns[i % 2] += player1_win_turn
        elif player2_win_turn is not False:
            win_turns[(i + 1) % 2] += player2_win_turn

        # mylogger.info("{}\n{}".format(deck_name_list[i % 2], deck_name_list[(i + 1) % 2]))
        if (i + 1) % span == 0:
            mylogger.info(
                "Halfway {}:win={}, lose={}, libout_num={}, win_rate:{:.3f}".format(i + 1, win_lose[0], win_lose[1],

                                                                                    lib_num, win_lose[0] / (i + 1)))
            mylogger.info("epoc_win_rate:{:.3%}".format((win_lose[0] - epoc_win_lose[0]) / (epoc_len)))
            mylogger.info("first_win_rate:Player1:{:.3%},Player2:{:.3%}".format((first_num[0] - epoch_first_num[0]) / max(1,(epoc_len//2)),
                          (first_num[1] - epoch_first_num[1]) / max(1,(epoc_len//2))))
        if (i + 1) % epoc_len == 0:
            if Player1.policy.name.split("_")[0] == "Genetic":
                Player1.policy.set_fitness((win_lose[0] - epoc_win_lose[0]) / (epoc_len))

    mylogger.info("Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                 win_lose[0] / iteration))
    mylogger.info("deck size:{} mean_end_turn {:<3}".format(len(D[0].deck), sum_of_turn / iteration))
    if iteration>1:
        mylogger.info("first_win_rate:Player1:{:<3},Player2:{:<3}".format(first_num[0] / (iteration // 2),
                                                                      first_num[1] / (iteration // 2)))
    if win_lose[0] == 0:
        win_lose[0] = 1
    if win_lose[1] == 0:
        win_lose[1] = 1
    mylogger.info("mean_win_turn:{:.3f},{:.3f}".format(win_turns[0] / win_lose[0], win_turns[1] / win_lose[1]))

    #import itertools
    #if Player1.mulligan_policy.data_use_flg:
    #    mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player1.mulligan_policy.mulligan_data,
    #                                                                        Player1.mulligan_policy.win_data)))))
    #if Player2.mulligan_policy.data_use_flg:
    #    mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player2.mulligan_policy.mulligan_data,
    #                                                                        Player2.mulligan_policy.win_data)))))
    mylogger.info("deck_type:{}".format(deck_type))
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    contribution_list = {"Alice": [], "Bob": []}
    drawn_win_rate_list = {"Alice": [], "Bob": []}
    for i, player_key in enumerate(list(deck_name_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for key in list(deck_name_list[player_key].keys()):
            target_cell = deck_name_list[player_key][key]
            if target_cell["used_num"] > 0:
                mylogger.info("{}'s contribution(used_num:{}):{:.3f}".format(key, target_cell["used_num"],
                                                                             target_cell["win_num"] / target_cell[
                                                                                 "used_num"]))
                contribution_list[player_key].append((key, target_cell["win_num"] / target_cell["used_num"]))
            if target_cell["drawn_num"] > 0:
                mylogger.info("{}'s drawn_win_rate(drawn_num:{}):{:.3f}".format(key, target_cell["drawn_num"],
                                                                                target_cell["win_num_when_drawn"] /
                                                                                target_cell["drawn_num"]))
                drawn_win_rate_list[player_key].append(
                    (key, target_cell["win_num_when_drawn"] / target_cell["drawn_num"]))

    contribution_list["Alice"].sort(key=lambda element: -element[1])
    contribution_list["Bob"].sort(key=lambda element: -element[1])
    drawn_win_rate_list["Alice"].sort(key=lambda element: -element[1])
    drawn_win_rate_list["Bob"].sort(key=lambda element: -element[1])
    mylogger.info("played_win_rate_rank")
    for i, player_key in enumerate(list(contribution_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for j, cell in enumerate(contribution_list[player_key]):
            mylogger.info("No.{} {}:{:.3f}".format(j + 1, cell[0], cell[1]))
        print("")
    mylogger.info("drawn_win_rate_rank")
    for i, player_key in enumerate(list(drawn_win_rate_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for j, cell in enumerate(drawn_win_rate_list[player_key]):
            mylogger.info("No.{} {}:{:.3f}".format(j + 1, cell[0], cell[1]))
        print("")



def execute_demo_with_pairwise(Player_1, Player_2, iteration, virtual_flg=False, deck_type=None,output=False,directory_name=None):
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = [Deck(), Deck()]
    if deck_type is None:
        deck_type = [5, 5]
    else:
        mylogger.info("deck_type:{}".format(deck_type))
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic", -2: "Sword_Basic",
                      -3: "Rune_Basic",
                      -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
                      100: "Test",
                      -9: "Spell-Rune",11:"PtP-Forest",12:"Mid-Shadow",13:"Neutral-Blood",100:"TEST"}

    key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                      3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                      6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                      9: ["SpellBoost-Rune.tsv", "RUNE"],10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                      11: ["PtP_Forest.tsv", "FOREST"],12: ["Mid_Shadow.tsv", "SHADOW"],
                      13: ["Neutral_Blood.tsv", "BLOOD"],100: ["TEST.tsv", "SHADOW"]}
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    class_pool = [0, 0]
    for i, d in enumerate(D):
        if deck_type[i] in key_2_tsv_name:
            D[i] = tsv_to_deck(key_2_tsv_name[deck_type[i]][0])
            D[i].set_leader_class(key_2_tsv_name[deck_type[i]][1])
            D[i].set_deck_type(Embedd_Network_model.deck_id_2_deck_type(deck_type[i]))

    Player1.class_num = class_pool[0]
    Player2.class_num = class_pool[1]
    mylogger.info("Alice's deck mean cost:{:<4}".format(D[0].get_mean_cost()))
    mylogger.info("Bob's deck mean cost:{:<4}".format(D[1].get_mean_cost()))
    D[0].mean_cost = D[0].get_mean_cost()
    D[1].mean_cost = D[1].get_mean_cost()
    assert len(D[0].deck) == 40 and len(D[1].deck) == 40, "deck_len:{},{}" \
        .format(len(D[0].deck), len(D[1].deck))
    mylogger.info("deck detail")
    D[0].show_remain_card_set()
    D[1].show_remain_card_set()
    mylogger.info("")
    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]
    win_lose = [win, lose]
    epoc_len = span
    epoc_win_lose = [0, 0]
    epoc_lib_num = 0
    first_num = [0, 0]
    epoch_first_num = [0,0]
    win_turns = [0, 0]
    deck_contents = {"Alice": D[0].get_name_set(), "Bob": D[1].get_name_set()}
    #mylogger.info("deck_contents")
    #mylogger.info(deck_contents)
    pairwise_dict = {}
    for i,player_name in enumerate(["Alice","Bob"]):
        pairwise_dict[player_name] = {}
        deck_name_list = D[i].get_name_set()
        card_name_keys=sorted(list(deck_name_list.keys()))
        for first_id,first_key in enumerate(card_name_keys):
            for second_id,second_key in enumerate(card_name_keys[first_id+1:],start=first_id+1):
                pairwise_dict[player_name][(first_key,second_key)] = {"both_played_num":0,"win_num_when_both_played":0}

    for i in range(iteration):
        if (i + 1) % epoc_len == 1 or epoc_len == 1:
            epoc_win_lose = [int(win_lose[0]), int(win_lose[1])]
            epoc_lib_num = int(lib_num)
            epoch_first_num = [int(first_num[0]),int(first_num[1])]
        if not virtual_flg:
            mylogger.info("Game {}".format(i + 1))
        # mylogger.info("name:{}".format(Turn_Players[i%2].name))
        Turn_Players[i % 2].is_first = True
        Turn_Players[i % 2].player_num = 0
        Turn_Players[(i + 1) % 2].is_first = False
        Turn_Players[(i + 1) % 2].player_num = 1
        assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
            Turn_Players[0].player_num)
        (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, (player1_win_turn, player2_win_turn)) \
            = demo_game_play_with_pairwise(Turn_Players[i % 2],
                             Turn_Players[(i + 1) % 2],
                             D[i % 2], D[(i + 1) % 2],
                             win_lose[i % 2],
                             win_lose[(i + 1) % 2], lib_num,
                             virtual_flg=virtual_flg, pairwise_dict=pairwise_dict,deck_name_list=deck_contents)
        first_num[i % 2] += first
        sum_of_turn += end_turn
        if player1_win_turn is not False:
            win_turns[i % 2] += player1_win_turn
        elif player2_win_turn is not False:
            win_turns[(i + 1) % 2] += player2_win_turn

        # mylogger.info("{}\n{}".format(deck_name_list[i % 2], deck_name_list[(i + 1) % 2]))
        if (i + 1) % span == 0:
            mylogger.info(
                "Halfway {}:win={}, lose={}, libout_num={}, win_rate:{:.3f}".format(i + 1, win_lose[0], win_lose[1],

                                                                                    lib_num, win_lose[0] / (i + 1)))
            mylogger.info("epoc_win_rate:{:.3%}".format((win_lose[0] - epoc_win_lose[0]) / (epoc_len)))
            mylogger.info("first_win_rate:Player1:{:.3%},Player2:{:.3%}".format((first_num[0] - epoch_first_num[0]) / max(1,(epoc_len//2)),
                          (first_num[1] - epoch_first_num[1]) / max(1,(epoc_len//2))))

    mylogger.info("Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                 win_lose[0] / iteration))
    mylogger.info("deck size:{} mean_end_turn {:<3}".format(len(D[0].deck), sum_of_turn / iteration))
    if iteration>1:
        mylogger.info("first_win_rate:Player1:{:<3},Player2:{:<3}".format(first_num[0] / (iteration // 2),
                                                                      first_num[1] / (iteration // 2)))
    #if win_lose[0] == 0:
    #    win_lose[0] = 1
    #if win_lose[1] == 0:
    #    win_lose[1] = 1
    if win_lose[0]>0 and win_lose[1]>0:
        mylogger.info("mean_win_turn:{:.3f},{:.3f}".format(win_turns[0] / win_lose[0], win_turns[1] / win_lose[1]))

    import itertools
    mylogger.info("deck_type:{}".format(deck_type))
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    contribution_list = {"Alice": [], "Bob": []}
    footcut_contribution_list = {"Alice": [], "Bob": []}
    single_contribution_list = {"Alice": [], "Bob": []}
    pairwise_contribution_list = {"Alice": [], "Bob": []}
    resulting_win_rate = [win_lose[0]/iteration,win_lose[1]/iteration]
    #drawn_win_rate_list = {"Alice": [], "Bob": []}
    for i, player_key in enumerate(list(deck_contents.keys())):
        for key in list(deck_contents[player_key].keys()):
            target_cell = deck_contents[player_key][key]
            if target_cell["used_num"] > 0:
                played_num = target_cell["used_num"]
                win_rate = target_cell["win_num"]/played_num
                single_contribution_list[player_key].append((key,win_rate,played_num))
    for i, player_key in enumerate(list(pairwise_dict.keys())):
        mylogger.info("Player{}".format(i + 1))
        for key in list(pairwise_dict[player_key].keys()):
            target_cell = pairwise_dict[player_key][key]
            if target_cell["both_played_num"] > 0:
                both_played_num = target_cell["both_played_num"]
                win_rate = target_cell["win_num_when_both_played"] / target_cell["both_played_num"]
                #mylogger.info("{}'s pairwise WRP(both_played_num:{}):{:.3f}".format(key, both_played_num,win_rate))
                contribution_list[player_key].append((key, win_rate,both_played_num))
                if both_played_num > (iteration//10):
                    footcut_contribution_list[player_key].append((key, win_rate, both_played_num))

                    first_cell = list(filter(lambda x: x[0] == key[0], single_contribution_list[player_key]))[0]
                    second_cell = list(filter(lambda x: x[0] == key[1], single_contribution_list[player_key]))[0]
                    first_single_win_rate = first_cell[1]
                    second_single_win_rate = second_cell[1]
                    pairwise_contribution = win_rate
                    persentages = (first_single_win_rate,second_single_win_rate)
                    #pairwise_contribution = win_rate - ((first_single_win_rate + second_single_win_rate)/2)
                    #persentage = both_played_num/iteration
                    #first_term = first_single_win_rate*(first_cell[2]/iteration)
                    #second_term = second_single_win_rate * (second_cell[2] / iteration)
                    #pairwise_contribution = win_rate*persentage - ((first_term+second_term)/2)

                    pairwise_contribution_list[player_key].append((key, pairwise_contribution,persentages))


    rank_range = 10
    contribution_list["Alice"].sort(key=lambda element: -element[1])
    contribution_list["Bob"].sort(key=lambda element: -element[1])
    contribution_list["Alice"] = contribution_list["Alice"][:rank_range]
    contribution_list["Bob"] = contribution_list["Bob"][:rank_range]
    footcut_contribution_list["Alice"].sort(key=lambda element: -element[1])
    footcut_contribution_list["Bob"].sort(key=lambda element: -element[1])
    single_contribution_list["Alice"].sort(key=lambda element: -element[1])
    single_contribution_list["Bob"].sort(key=lambda element: -element[1])
    #pairwise_contribution_list["Alice"].sort(key=lambda element: -element[1]*element[2])
    #pairwise_contribution_list["Bob"].sort(key=lambda element: -element[1]*element[2])
    pairwise_contribution_list["Alice"].sort(key=lambda element: -element[1])
    pairwise_contribution_list["Bob"].sort(key=lambda element: -element[1])
    if len(footcut_contribution_list["Alice"])<rank_range:
        rank_range = len(footcut_contribution_list["Alice"])
    footcut_contribution_list["Alice"] = footcut_contribution_list["Alice"][:10]
    rank_range = 10
    if len(footcut_contribution_list["Bob"])<rank_range:
        rank_range = len(footcut_contribution_list["Bob"])
    footcut_contribution_list["Bob"] = footcut_contribution_list["Bob"][:10]
    #drawn_win_rate_list["Alice"].sort(key=lambda element: -element[1])
    #drawn_win_rate_list["Bob"].sort(key=lambda element: -element[1])
    """
    mylogger.info("pairwise-WRP_rank")
    for i, player_key in enumerate(list(contribution_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for j, cell in enumerate(contribution_list[player_key]):
            txt = "No.{} ({},{})(both_played_num:{})".format(j + 1, cell[0][0],cell[0][1],cell[2])
            txt = "{:<80}:{:.3%}".format(txt, cell[1])
            mylogger.info(txt)
        print("")

    mylogger.info("pairwise-WRP_rank(footcut by threthold:{})".format(iteration//10))
    for i, player_key in enumerate(list(footcut_contribution_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for j, cell in enumerate(footcut_contribution_list[player_key]):
            txt = "No.{} ({},{})(both_played_num:{})".format(j + 1, cell[0][0],cell[0][1],cell[2])
            txt = "{:<80}:{:.3%}".format(txt, cell[1])
            mylogger.info(txt)
            #covariance_cell = list(filter(lambda x:x[0] == cell[0],pairwise_contribution_list[player_key]))[0]
            #mylogger.info("covariance:{:.3%}".format(covariance_cell[1]))

            mylogger.info("")
        print("")
    mylogger.info("Single WRP")
    for i, player_key in enumerate(list(single_contribution_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for j, cell in enumerate(single_contribution_list[player_key]):
            txt = "No.{} {}(played_num:{})".format(j + 1, cell[0],cell[2])
            txt = "{:<80}:{:.3%}".format(txt,cell[1])
            mylogger.info(txt)
        print("")
    """
    mylogger.info("WRP_result")
    for i,player_key in enumerate(list(pairwise_contribution_list.keys())):
        mylogger.info("Player{}".format(i+1))
        for j,cell in enumerate(pairwise_contribution_list[player_key]):
            txt = "({},{})".format(cell[0][0],cell[0][1])
            mylogger.info("{:<80}:{:.3%} | {:.3%} {:.3%}".format(txt,cell[1],cell[2][0],cell[2][1]))

    title = "{}".format(deck_id_2_name[deck_type[0]])
    result_txt = "Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                win_lose[0] / iteration)
    first_win_rate_txt = "first_win_rate Player1:{:.3f},Player2:{:.3f}" \
        .format(first_num[0] / (iteration // 2), first_num[1] / (iteration // 2))
    if output:
        file_name = title+".txt"
        path =  file_name
        if directory_name is not None:
            path = directory_name + "/" +path
        with open(path, mode="w") as f:
            f.write(title+"\n")
            f.write(result_txt+"\n")
            f.write(first_win_rate_txt + "\n")
            f.write("pairwise_WRP\n")
            for i, player_key in enumerate(list(contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(contribution_list[player_key]):
                    pairwise = "No.{} ({},{})(both_played_num:{})".format(j + 1, cell[0][0],cell[0][1],cell[2])
                    txt = "{:<80}:{:.3%}\n".format(pairwise,cell[1])
                    f.write(txt)
                f.write("\n")
            f.write("\n")
            f.write("pairwise-WRP_rank(footcut by threthold:{})\n".format(iteration//10))
            for i, player_key in enumerate(list(footcut_contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(footcut_contribution_list[player_key]):
                    pairwise = "No.{} ({},{})(both_played_num:{})".format(j + 1, cell[0][0],cell[0][1],cell[2])
                    txt = "{:<80}:{:.3%}\n".format(pairwise,cell[1])
                    f.write(txt)
                f.write("\n")
            f.write("\n")
            f.write("WRP_rank\n")
            for i, player_key in enumerate(list(single_contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(single_contribution_list[player_key]):
                    pairwise = "No.{} {}(played_num:{})".format(j + 1, cell[0],cell[2])
                    txt = "{:<40}:{:.3%}\n".format(pairwise,cell[1])
                    f.write(txt)
                f.write("\n")
            f.write("\n")
        path = "{}_Result.tsv".format(deck_id_2_name[deck_type[0]])
        path = directory_name + "/" + path
        with open(path,mode = "w") as w:
            writer = csv.writer(w, delimiter='\t', lineterminator='\n')

            writer.writerow(["pairwise_contribution"])
            for i, player_key in enumerate(list(pairwise_contribution_list.keys())):
                writer.writerow(["Player{}".format(i + 1),"first","second","pairwise-WRP","first_WRP","second_WRP"])
                for j, cell in enumerate(pairwise_contribution_list[player_key]):
                    row = []
                    row.append("No.{}".format(j+1))
                    #txt = "({},{})".format(cell[0][0], cell[0][1])
                    #row.append(txt)
                    #row.extend([cell[0][0],cell[0][1]])
                    row.append(cell[0][0])
                    row.append(cell[0][1])

                    #row.append("{:.3%}".format(cell[1]))
                    #row.append("{:.3%}".format(cell[2]))

                    #row.append("{:.3%}".format(cell[1]*cell[2]))
                    #row.append("{:.3%}".format(cell[1]))
                    #row.append("{:.3%}".format(cell[1]*cell[2]))
                    #row.append("{:.3%}".format(cell[2]))
                    row.append("{:.3%}".format(cell[1]))
                    row.append("{:.3%}".format(cell[2][0]))
                    row.append("{:.3%}".format(cell[2][1]))
                    writer.writerow(row)
                writer.writerow([])


def random_match(Player_1, Player_2, iteration, virtual_flg=False):
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = [Deck(), Deck()]
    #deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
    #                  6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic",
    #                  -2: "Sword_Basic",
    #                  -3: "Rune_Basic",
    #                  -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
    #                  100: "Test",
    #                  -9: "Spell-Rune", 11: "PtP-Forest", 12: "Mid-Shadow", 13: "Neutral-Blood"}

    key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                      3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                      6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                      9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                      11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"],
                      13: ["Neutral_Blood.tsv", "BLOOD"]}


    class_pool = [0, 0]

    Player1.class_num = class_pool[0]
    Player2.class_num = class_pool[1]

    mylogger.info("")
    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]
    win_lose = [win, lose]
    epoc_len = span
    epoc_win_lose = [0, 0]
    epoc_lib_num = 0
    first_num = [0, 0]
    epoch_first_num = [0, 0]
    win_turns = [0, 0]
    deck_keys = list(key_2_tsv_name.keys())
    for i in tqdm(range(iteration)):
        deck_type = [random.choice(deck_keys),random.choice(deck_keys)]
        for j, d in enumerate(D):
            if deck_type[j] in key_2_tsv_name:
                D[j] = tsv_to_deck(key_2_tsv_name[deck_type[j]][0])
                D[j].set_leader_class(key_2_tsv_name[deck_type[j]][1])
            else:
                assert False
        D[0].mean_cost = D[0].get_mean_cost()
        D[1].mean_cost = D[1].get_mean_cost()

        if (i + 1) % epoc_len == 1 or epoc_len == 1:
            epoc_win_lose = [int(win_lose[0]), int(win_lose[1])]
            epoc_lib_num = int(lib_num)
            epoch_first_num = [int(first_num[0]), int(first_num[1])]
        Turn_Players[i % 2].is_first = True
        Turn_Players[i % 2].player_num = 0
        Turn_Players[(i + 1) % 2].is_first = False
        Turn_Players[(i + 1) % 2].player_num = 1
        assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
            Turn_Players[0].player_num)
        (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, (player1_win_turn, player2_win_turn)) \
            = demo_game_play(Turn_Players[i % 2],
                             Turn_Players[(i + 1) % 2],
                             D[i % 2], D[(i + 1) % 2],
                             win_lose[i % 2],
                             win_lose[(i + 1) % 2], lib_num,
                             virtual_flg=True, deck_name_list=None, history_flg=False)
        first_num[i % 2] += first
        sum_of_turn += end_turn
        if player1_win_turn is not False:
            win_turns[i % 2] += player1_win_turn
        elif player2_win_turn is not False:
            win_turns[(i + 1) % 2] += player2_win_turn


    mylogger.info("Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                 win_lose[0] / iteration))
    if iteration > 1:
        mylogger.info("first_win_rate:Player1:{:<3},Player2:{:<3}".format(first_num[0] / (iteration // 2),
                                                                          first_num[1] / (iteration // 2)))
    if win_lose[0] == 0:
        win_lose[0] = 1
    if win_lose[1] == 0:
        win_lose[1] = 1
    mylogger.info("mean_win_turn:{:.3f},{:.3f}".format(win_turns[0] / win_lose[0], win_turns[1] / win_lose[1]))


def get_contributions(Player_1, Player_2, iteration, player1_deck_num=None, directory_name=None):
    assert player1_deck_num is not None
    assert directory_name is not None
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    deck_id_2_name = {0: "Aggro-Sword", 1: "Earth-RUne", 2: "Midrange-Sword", 3: "Midrange-Shadow", 4: "PDK-Dragon",
                      5: "Elana-Haven",
                      6: "Control-Blood", 7: "Ramp-Dragon", 8: "Forest", 9: "Spell-Rune"}
    D = [Deck()] * len(deck_id_2_name)
    assert player1_deck_num < len(D), "player1_deck_num:{}".format(player1_deck_num)
    class_pool = [0, 0]
    for i in range(len(D)):
        if i == 0:
            D[i] = tsv_to_deck("Sword_Aggro.tsv")
            # Aggro
        elif i == 1:
            D[i] = tsv_to_deck("Rune_Earth.tsv")
            # Aggro
        elif i == 2:
            D[i] = tsv_to_deck("Sword.tsv")
            # Mid
        elif i == 3:
            # D[i]=tsv_to_deck("Shadow.tsv")
            D[i] = tsv_to_deck("New-Shadow.tsv")
            # Mid
        elif i == 4:
            D[i] = tsv_to_deck("Dragon_PDK.tsv")
            # Mid
        elif i == 5:
            D[i] = tsv_to_deck("Haven.tsv")
            # Control
        elif i == 6:
            D[i] = tsv_to_deck("Blood.tsv")
            # Control
        elif i == 7:
            D[i] = tsv_to_deck("Dragon.tsv")
            # Control
        elif i == 8:
            D[i] = tsv_to_deck("Forest.tsv")
            # Combo
        elif i == 9:
            D[i] = tsv_to_deck("Rune.tsv")
            # Combo

    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]

    for deck_id in range(len(D)):
        current_decks = [D[player1_deck_num], D[deck_id]]
        win_lose = [0, 0]
        win_turns = [0, 0]
        first_num = [0,0]
        deck_name_list = {"Alice": {}, "Bob": {}}
        deck_name_list["Alice"] = current_decks[0].get_name_set()
        deck_name_list["Bob"] = current_decks[1].get_name_set()

        for i in range(iteration):
            Turn_Players[i % 2].is_first = True
            Turn_Players[i % 2].player_num = 0
            Turn_Players[(i + 1) % 2].is_first = False
            Turn_Players[(i + 1) % 2].player_num = 1
            assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
                Turn_Players[0].player_num)
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, (player1_win_turn, player2_win_turn)) \
                = demo_game_play(Turn_Players[i % 2],
                                 Turn_Players[(i + 1) % 2],
                                 current_decks[i % 2], current_decks[(i + 1) % 2],
                                 win_lose[i % 2],
                                 win_lose[(i + 1) % 2], lib_num,
                                 virtual_flg=True, deck_name_list=deck_name_list)
            first_num[i%2] += first
            sum_of_turn += end_turn
            if player1_win_turn is not False:
                win_turns[i % 2] += player1_win_turn
            elif player2_win_turn is not False:
                win_turns[(i + 1) % 2] += player2_win_turn
        result_txt = "Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                            win_lose[0] / iteration)
        first_win_rate_txt="first_win_rate[Player1:{:.3f},Player2:.3f}"\
            .format(first_num[0]/(iteration//2),first_num[1]/(iteration//2))
        mylogger.info(result_txt)
        if win_lose[0] == 0:
            win_lose[0] = 1
        if win_lose[1] == 0:
            win_lose[1] = 1

        # mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[player1_deck_num], Player_2.policy.name,
        #                                       deck_id_2_name[deck_id]))
        contribution_list = {"Alice": [], "Bob": []}
        for i, player_key in enumerate(list(deck_name_list.keys())):
            # mylogger.info("Player{}".format(i+1))
            for key in list(deck_name_list[player_key].keys()):
                target_cell = deck_name_list[player_key][key]
                if target_cell["used_num"] > 0:
                    # mylogger.info("{}'s contribution(used_num:{}):{:.3f}".format(key,target_cell["used_num"],target_cell["win_num"]/target_cell["used_num"]))
                    contribution_list[player_key].append(
                        (key, target_cell["win_num"] / target_cell["used_num"], target_cell["used_num"]))

        contribution_list["Alice"].sort(key=lambda element: -element[1])
        contribution_list["Bob"].sort(key=lambda element: -element[1])
        file_name = ""
        title = "{}({})vs {}({})({} matchup)".format(Player_1.policy.name, deck_id_2_name[player1_deck_num],
                                                     Player_2.policy.name,
                                                     deck_id_2_name[deck_id], iteration)
        mylogger.info(title)
        title += "\n"
        if player1_deck_num != deck_id:
            file_name = "contribution_{}_vs_{}.txt".format(deck_id_2_name[player1_deck_num], deck_id_2_name[deck_id])
        else:
            file_name = "contribution_{}_mirror.txt".format(deck_id_2_name[deck_id])
        with open(directory_name + "/" + file_name, mode="w") as f:

            f.write(title)
            f.write(result_txt + "\n")
            f.write(first_win_rate_txt + "\n")
            for i, player_key in enumerate(list(contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(contribution_list[player_key]):
                    txt = "No.{} {}(used_num:{}):{:.3f}\n".format(j + 1, cell[0], cell[2], cell[1])
                    f.write(txt)
                    # mylogger.info("No.{} {}:{:.3f}".format(j+1,cell[0],cell[1]))
                f.write("\n")
        mylogger.info("{}/{} complete".format((deck_id + 1), len(D)))


def get_basic_contributions(Player_1, Player_2, iteration, virtual_flg=False, player1_deck_num=None,
                            directory_name=None):
    assert player1_deck_num is not None
    assert directory_name is not None
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = [Deck()] * 8
    assert player1_deck_num < len(D), "player1_deck_num:{}".format(player1_deck_num)
    deck_id_2_name = {0: "Forest", 1: "Sword", 2: "Rune", 3: "Dragon", 4: "Shadow", 5: "Blood",
                      6: "Haven", 7: "Portal"}
    class_pool = [0, 0]
    for i in range(len(D)):
        if i == 0:
            D[i] = tsv_to_deck("Forest_Basic.tsv")
            D[i].set_leader_class("FOREST")
        elif i == 1:
            D[i] = tsv_to_deck("Sword_Basic.tsv")
            D[i].set_leader_class("SWORD")
        elif i == 2:
            D[i] = tsv_to_deck("Rune_Basic.tsv")
            D[i].set_leader_class("RUNE")
        elif i == 3:
            D[i] = tsv_to_deck("Dragon_Basic.tsv")
            D[i].set_leader_class("DRAGON")
        elif i == 4:
            D[i] = tsv_to_deck("Shadow_Basic.tsv")
            D[i].set_leader_class("SHADOW")
        elif i == 5:
            D[i] = tsv_to_deck("Blood_Basic.tsv")
            D[i].set_leader_class("BLOOD")
        elif i == 6:
            D[i] = tsv_to_deck("Haven_Basic.tsv")
            D[i].set_leader_class("HAVEN")
        elif i == 7:
            D[i] = tsv_to_deck("Portal_Basic.tsv")
            D[i].set_leader_class("PORTAL")

    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]

    for deck_id in range(len(D)):
        current_decks = [D[player1_deck_num], D[deck_id]]
        win_lose = [0, 0]
        win_turns = [0, 0]
        first_num = [0,0]
        deck_name_list = {"Alice": current_decks[0].get_name_set(), "Bob": current_decks[1].get_name_set()}
        for i in range(iteration):
            Turn_Players[i % 2].is_first = True
            Turn_Players[i % 2].player_num = 0
            Turn_Players[(i + 1) % 2].is_first = False
            Turn_Players[(i + 1) % 2].player_num = 1
            assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
                Turn_Players[0].player_num)
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, (player1_win_turn, player2_win_turn)) \
                = demo_game_play(Turn_Players[i % 2],
                                 Turn_Players[(i + 1) % 2],
                                 current_decks[i % 2], current_decks[(i + 1) % 2],
                                 win_lose[i % 2],
                                 win_lose[(i + 1) % 2], lib_num,
                                 virtual_flg=True, deck_name_list=deck_name_list)
            first_num[i%2] += first
            sum_of_turn += end_turn
            if player1_win_turn is not False:
                win_turns[i % 2] += player1_win_turn
            elif player2_win_turn is not False:
                win_turns[(i + 1) % 2] += player2_win_turn
        result_txt = "Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                    win_lose[0] / iteration)
        mylogger.info(result_txt)
        first_win_rate_txt="first_win_rate[Player1:{:.3f},Player2:.3f}"\
            .format(first_num[0]/(iteration//2),first_num[1]/(iteration//2))
        result_txt += "\n"
        if win_lose[0] == 0:
            win_lose[0] = 1
        if win_lose[1] == 0:
            win_lose[1] = 1

        # mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[player1_deck_num], Player_2.policy.name,
        #                                       deck_id_2_name[deck_id]))
        contribution_list = {"Alice": [], "Bob": []}
        drawn_win_rate_list = {"Alice": [], "Bob": []}
        for i, player_key in enumerate(list(deck_name_list.keys())):
            for key in list(deck_name_list[player_key].keys()):
                target_cell = deck_name_list[player_key][key]
                if target_cell["used_num"] > 0:
                    contribution_list[player_key].append((key, target_cell["win_num"] / target_cell["used_num"],
                                                          target_cell["used_num"],
                                                          target_cell["sum_of_turn_when_used"] / target_cell[
                                                              "used_num"]))
                if target_cell["drawn_num"] > 0:
                    drawn_win_rate_list[player_key].append(
                        (key, target_cell["win_num_when_drawn"] / target_cell["drawn_num"], target_cell["drawn_num"]))

        contribution_list["Alice"].sort(key=lambda element: -element[1])
        contribution_list["Bob"].sort(key=lambda element: -element[1])
        drawn_win_rate_list["Alice"].sort(key=lambda element: -element[1])
        drawn_win_rate_list["Bob"].sort(key=lambda element: -element[1])
        file_name = ""
        title = "{}({})vs {}({})({} iteration)".format(Player_1.policy.name, deck_id_2_name[player1_deck_num],
                                                       Player_2.policy.name,
                                                       deck_id_2_name[deck_id], iteration)
        mylogger.info(title)
        title += "\n"
        if player1_deck_num != deck_id:
            file_name = "{}_vs_{}'s_WRP_and_WRD.txt".format(deck_id_2_name[player1_deck_num], deck_id_2_name[deck_id])
        else:
            file_name = "{}_mirror_WRP_and_WRD.txt".format(deck_id_2_name[deck_id])
        with open(directory_name + "/" + file_name, mode="w") as f:

            f.write(title)
            f.write(result_txt)
            f.write(first_win_rate_txt + "\n")
            f.write("WRP\n")
            for i, player_key in enumerate(list(contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(contribution_list[player_key]):
                    txt = "No.{} {}(used_num:{},ave_turn:{:.2f}):{:.3f}\n".format(j + 1, cell[0], cell[2], cell[-1],
                                                                                  cell[1])
                    f.write(txt)
                f.write("\n")
            f.write("\nWRD\n")
            for i, player_key in enumerate(list(drawn_win_rate_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(drawn_win_rate_list[player_key]):
                    txt = "No.{} {}(drawn_num:{}):{:.3f}\n".format(j + 1, cell[0], cell[2], cell[1])
                    f.write(txt)
                f.write("\n")

        mylogger.info("{}/{} complete".format((deck_id + 1), len(D)))





def make_deck_table(Player_1, Player_2, iteration, same_flg=False, result_name=None, basic=False,deck_lists=None):
    mylogger.info("{} vs {}".format(Player_1.policy.name, Player_2.policy.name))

    if result_name is None:
        result_name = "{}_vs_{}_{}iteration(deck_list={}).tsv".format(Player_1.policy.name, Player_2.policy.name, iteration,deck_lists)
    #assert Player1 != Player2
    #Player1.deck = None
    #Player2.deck = None
    win = 0
    lose = 0
    lib_num = 0
    D = []

    deck_id_2_name = {}
    if not basic:
        deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                          6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic",
                          -2: "Sword_Basic",
                          -3: "Rune_Basic",
                          -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic",
                          -8: "Portal_Basic",
                          100: "Test",
                          -9: "Spell-Rune", 11: "PtP-Forest", 12: "Mid-Shadow", 13: "Neutral-Blood"}

        key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                          3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"],
                          5: ["Test-Haven.tsv", "HAVEN"],
                          6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                          9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                          11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"],
                          13: ["Neutral_Blood.tsv", "BLOOD"]}

        if deck_lists is None:
            deck_lists = list(deck_id_2_name.keys())
        else:
            assert all(key in deck_id_2_name for key in deck_lists)
        if deck_lists == [0,1,4,5,10,12]:
            deck_lists = [0,1,4,12,5,10]
        mylogger.info("deck_lists:{}".format(deck_lists))
        D = [Deck() for i in range(len(deck_lists))]
        deck_index = 0
        #sorted_keys = sorted(list(deck_id_2_name.keys()))
        #for i in sorted_keys:
        #    if i not in deck_lists:
        #        continue
        for i in deck_lists:
            mylogger.info("{}(deck_id:{}):{}".format(deck_index,i,key_2_tsv_name[i]))
            D[deck_index] = tsv_to_deck(key_2_tsv_name[i][0])
            D[deck_index].set_leader_class(key_2_tsv_name[i][1])
            deck_index += 1

    assert all(len(D[i].deck) == 40 for i in range(len(D)))
    # Turn_Players=[Player1,Player2]
    Results = {}
    mylogger.info("same_flg:{}".format(same_flg))
    list_range = range(len(deck_list))
    #print(list(itertools.product(list_range,list_range)))
    """
    Player1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(model_name=model_name),
                     mulligan=Min_cost_mulligan_policy())

    Player2 = Player(9, True, policy=AggroPolicy(),
                     mulligan=Min_cost_mulligan_policy())
    """
    Player1 = Player_1.get_copy(None)
    Player1.name = "Alice"
    Player2 = Player_2.get_copy(None)
    Player2.name = "Bob"
    """
    iter_data = [(Player1,
                   Player2,(i,j),(deck_lists[i],deck_lists[j]),iteration) for i,j in itertools.product(list_range,list_range)]
    pool = Pool(3)  # 最大プロセス数:8
    # memory = pool.map(preparation, iter_data)
    result = pool.map(multi_battle, iter_data)
    #result = list(tqdm(result, total=len(list_range)**2))
    pool.close()  # add this.
    pool.terminate()  # add this.
    for data in result:
        Results[data[0]] = data[1]
    print(Results)
    """
    #assert  False

    for j in range(len(D)):
        l = 0
        if same_flg:
            l = j
        for k in range(l, len(D)):
            mylogger.info("{} vs {}".format(deck_id_2_name[deck_lists[j]],deck_id_2_name[deck_lists[k]]))
            Turn_Players = [Player1, Player2]
            assert Player1 != Player2
            win_lose = [win, lose]
            first_num = 0
            for i in range(iteration):
                Turn_Players[i % 2].is_first = True
                Turn_Players[i % 2].player_num = 0
                Turn_Players[(i + 1) % 2].is_first = False
                Turn_Players[(i + 1) % 2].player_num = 1
                assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {} name:{} {}" \
                    .format(Turn_Players[0].player_num, Turn_Players[0].name, Turn_Players[1].name)
                (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, _) = demo_game_play(Turn_Players[i % 2],
                                                                                                  Turn_Players[
                                                                                                      (i + 1) % 2],
                                                                                                  D[j],
                                                                                                  D[k],
                                                                                                  win_lose[i % 2],
                                                                                                  win_lose[(i + 1) % 2],
                                                                                                  lib_num,
                                                                                                  virtual_flg=True)
                first_num += first
            Results[(j, k)] = [win_lose[0] / iteration, first_num / iteration]
            mylogger.info("win_rate:{:%} first_win_rate:{:%}".format(Results[(j,k)][0],Results[(j,k)][1]))

        mylogger.info("complete:{}/{}".format(j + 1, len(D)))

    # for key in list(Results.keys()):
    #    mylogger.info("({}):rate:{} first:{}".format(key,Results[key][0],Results[key][1]))
    with open("Battle_Result/" + result_name, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        row = ["{} vs {}".format(Player1.policy.name, Player2.policy.name)]
        mylogger.info("row:{}".format(row))
        deck_names = [deck_id_2_name[deck_lists[i]] for i in range(len(D))]
        mylogger.info("row:{}".format(deck_names))
        row = row + deck_names
        mylogger.info("row:{}".format(row))
        writer.writerow(row)
        if same_flg:
            for i in range(len(D)):
                row = [deck_id_2_name[deck_lists[i]]]
                for j in range(0, i + 1):
                    if (j, i) not in Results:
                        mylogger.info("(i,j):{}".format((j, i)))
                        mylogger.info("Results:{}".format(Results.keys()))
                        assert False
                    row.append(Results[(j, i)][0])
                mylogger.info(row)
                writer.writerow(row)
        else:
            for i in range(len(D)):
                row = [deck_id_2_name[deck_lists[i]]]
                for j in range(len(D)):
                    row.append(Results[(i, j)][0])
                writer.writerow(row)


def test_3(Player_1, Player_2, iteration, same_flg=False, result_name="shadow_result.tsv"):
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = [Deck() for i in range(9)]

    D[0] = tsv_to_deck("Sword_Aggro.tsv")
    D[1] = tsv_to_deck("Rune_Earth.tsv")
    D[2] = tsv_to_deck("Sword.tsv")
    D[3] = tsv_to_deck("Shadow.tsv")
    D[4] = tsv_to_deck("Haven.tsv")
    D[5] = tsv_to_deck("Blood.tsv")
    D[6] = tsv_to_deck("Dragon.tsv")
    D[7] = tsv_to_deck("Forest.tsv")
    D[8] = tsv_to_deck("Rune.tsv")

    # D[0].mean_cost=D[0].get_mean_cost()
    # D[1].mean_cost=D[1].get_mean_cost()
    assert all(len(D[i].deck) == 40 for i in range(8))
    # Turn_Players=[Player1,Player2]
    Results = {}
    mylogger.info("same_flg:{}".format(same_flg))
    l = 0
    j = 3
    if same_flg:
        l = j
    for k in range(l, len(D)):
        Turn_Players = [Player1, Player2]
        assert Player1 != Player2
        win_lose = [win, lose]
        first_num = 0
        for i in range(iteration):
            Turn_Players[i % 2].is_first = True
            Turn_Players[i % 2].player_num = 0
            Turn_Players[(i + 1) % 2].is_first = False
            Turn_Players[(i + 1) % 2].player_num = 1
            assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {} name:{} {}" \
                .format(Turn_Players[0].player_num, Turn_Players[0].name, Turn_Players[1].name)
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, _) = game_play(Turn_Players[i % 2],
                                                                                              Turn_Players[(i + 1) % 2],
                                                                                              D[j], D[k], \
                                                                                              win_lose[i % 2],
                                                                                              win_lose[(i + 1) % 2],
                                                                                              lib_num,
                                                                                              virtual_flg=False)
            first_num += first


def make_policy_table(n, initial_players=None, deck_type=None, same_flg=False, result_name="Policy_table_result.tsv"):
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic", -2: "Sword_Basic",
                      -3: "Rune_Basic",
                      -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
                      100: "Test",
                      -9: "Spell-Rune",11:"PtP-Forest",12:"Mid-Shadow",13:"Neutral-Blood"}

    key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                      3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                      6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                      9: ["SpellBoost-Rune.tsv", "RUNE"],10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                      11: ["PtP_Forest.tsv", "FOREST"],12: ["Mid_Shadow.tsv", "SHADOW"],
                      13: ["Neutral_Blood.tsv", "BLOOD"]}
    mylogger.info("{} vs {}".format(deck_id_2_name[deck_type[0]], deck_id_2_name[deck_type[1]]))
    policy_id_2_name = {}
    for i, target_player in enumerate(initial_players):
        policy_id_2_name[i] = target_player.policy.name
    mylogger.info("players:{}".format(policy_id_2_name))
    iteration = n
    win = 0
    lose = 0
    lib_num = 0
    assert initial_players is not None, "Non-players!"
    assert deck_type is not None, "Non-Deck_type!"
    #players = copy.deepcopy(initial_players)
    players = [player.get_copy(None) for player in initial_players]
    D = [Deck() for i in range(2)]
    for i, d in enumerate(D):
        if deck_type[i] in key_2_tsv_name:
            D[i] = tsv_to_deck(key_2_tsv_name[deck_type[i]][0])
            D[i].set_leader_class(key_2_tsv_name[deck_type[i]][1])
    Results = {}
    for policy1_id, player1 in enumerate(players):
        P1 = player1.get_copy(None)
        P1.name = "Alice"
        last_id = len(players)
        for policy2_id in range(0, last_id):
            player2 = players[policy2_id]
            P2 = player2.get_copy(None)
            P2.name = "Bob"
            Turn_Players = [P1, P2]
            win_lose = [win, lose]
            first_num = 0
            for i in range(iteration):
                Turn_Players[i % 2].is_first = True
                Turn_Players[i % 2].player_num = 0
                Turn_Players[(i + 1) % 2].is_first = False
                Turn_Players[(i + 1) % 2].player_num = 1
                assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {} name:{} {}" \
                    .format(Turn_Players[0].player_num, Turn_Players[0].name, Turn_Players[1].name)
                (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, _) = game_play(Turn_Players[i % 2],
                                                                                                  Turn_Players[
                                                                                                      (i + 1) % 2],
                                                                                                  D[i % 2],
                                                                                                  D[(i + 1) % 2],
                                                                                                  win_lose[i % 2],
                                                                                                  win_lose[(i + 1) % 2],
                                                                                                  lib_num,
                                                                                                  virtual_flg=True)
                first_num += first
            Results[(policy1_id, policy2_id)] = [win_lose[0] / iteration, first_num / iteration]
        mylogger.info("complete:{}/{}".format(policy1_id + 1, len(players)))
    mylogger.info("keys:{}".format(list(Results.keys())))
    with open("Battle_Result/" + result_name, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        row = ["{} vs {}".format(deck_id_2_name[deck_type[0]], deck_id_2_name[deck_type[1]])]
        mylogger.info("row:{}".format(row))
        mylogger.info("row:{}".format([policy_id_2_name[i] for i in range(len(policy_id_2_name))]))
        row = row + [policy_id_2_name[i] for i in range(len(policy_id_2_name))]
        mylogger.info("row:{}".format(row))
        writer.writerow(row)
        for i in range(len(policy_id_2_name)):
            row = [policy_id_2_name[i]]
            for j in range(len(policy_id_2_name)):
                row.append(Results[(i, j)][0])
            writer.writerow(row)


def get_custom_contributions(Player_1, Player_2, iteration, virtual_flg=False, player1_deck_num=None,
                             directory_name=None):
    assert player1_deck_num is not None
    assert directory_name is not None
    Player1 = Player_1.get_copy(None)
    Player2 = Player_2.get_copy(None)
    Player1.name = "Alice"
    Player2.name = "Bob"
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = [Deck()] * 10
    assert player1_deck_num < len(D), "player1_deck_num:{}".format(player1_deck_num)
    deck_id_2_name = {0: "Aggro-Sword", 1: "Rune_Earth(Aggro)", 2: "Mid-Sword", 3: "Mid-Shadow", 4: "Dragon_PDK(Mid)",
                      5: "Elana-Haven(Control)",
                      6: "Control-Blood", 7: "Ramp-Dragon(Combo)", 8: "Combo-Forest", 9: "Spell-Rune(Combo)"}
    class_pool = [0, 0]
    D = [Deck() for i in range(len(D))]
    for i in range(len(D)):
        if i == 0:
            D[i] = tsv_to_deck("Sword_Aggro.tsv")
            D[i].set_leader_class("SWORD")
            # Aggro
        elif i == 1:
            D[i] = tsv_to_deck("Rune_Earth.tsv")
            D[i].set_leader_class("RUNE")
            # Aggro
        elif i == 2:
            D[i] = tsv_to_deck("Sword.tsv")
            D[i].set_leader_class("SWORD")
            # Mid
        elif i == 3:
            # D[i]=tsv_to_deck("Shadow.tsv")
            D[i] = tsv_to_deck("New-Shadow.tsv")
            D[i].set_leader_class("SHADOW")
            # Mid
        elif i == 4:
            D[i] = tsv_to_deck("Dragon_PDK.tsv")
            D[i].set_leader_class("DRAGON")
            # Mid
        elif i == 5:
            D[i] = tsv_to_deck("Test-Haven.tsv")
            D[i].set_leader_class("HAVEN")
            # D[i] = tsv_to_deck("Haven.tsv")
            # D[i].set_leader_class("HAVEN")
            # Control
        elif i == 6:
            D[i] = tsv_to_deck("Blood.tsv")
            D[i].set_leader_class("BLOOD")
            # Control
        elif i == 7:
            D[i] = tsv_to_deck("Dragon.tsv")
            D[i].set_leader_class("DRAGON")
            # Control
        elif i == 8:
            D[i] = tsv_to_deck("Forest.tsv")
            D[i].set_leader_class("FOREST")
            # Combo
        elif i == 9:
            D[i] = tsv_to_deck("Rune.tsv")
            D[i].set_leader_class("RUNE")
            # Combo

    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]

    for deck_id in range(len(D)):
        current_decks = [D[player1_deck_num], D[deck_id]]
        win_lose = [0, 0]
        win_turns = [0, 0]
        first_num = [0,0]
        deck_name_list = {"Alice": current_decks[0].get_name_set(), "Bob": current_decks[1].get_name_set()}
        for i in range(iteration):
            Turn_Players[i % 2].is_first = True
            Turn_Players[i % 2].player_num = 0
            Turn_Players[(i + 1) % 2].is_first = False
            Turn_Players[(i + 1) % 2].player_num = 1
            assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
                Turn_Players[0].player_num)
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first, (player1_win_turn, player2_win_turn)) \
                = demo_game_play(Turn_Players[i % 2],
                                 Turn_Players[(i + 1) % 2],
                                 current_decks[i % 2], current_decks[(i + 1) % 2],
                                 win_lose[i % 2],
                                 win_lose[(i + 1) % 2], lib_num,
                                 virtual_flg=True, deck_name_list=deck_name_list)
            first_num[i%2] += first
            sum_of_turn += end_turn
            if player1_win_turn is not False:
                win_turns[i % 2] += player1_win_turn
            elif player2_win_turn is not False:
                win_turns[(i + 1) % 2] += player2_win_turn
        result_txt = "Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                    win_lose[0] / iteration)
        first_win_rate_txt="first_win_rate[Player1:{:.3f},Player2:{:.3f}"\
            .format(first_num[0]/(iteration//2),first_num[1]/(iteration//2))
        mylogger.info(result_txt)
        result_txt += "\n"
        if win_lose[0] == 0:
            win_lose[0] = 1
        if win_lose[1] == 0:
            win_lose[1] = 1

        contribution_list = {"Alice": [], "Bob": []}
        drawn_win_rate_list = {"Alice": [], "Bob": []}
        for i, player_key in enumerate(list(deck_name_list.keys())):
            for key in list(deck_name_list[player_key].keys()):
                target_cell = deck_name_list[player_key][key]
                if target_cell["used_num"] > 0:
                    # mylogger.info("sum_of_turn_when_used:{},used_num:{}".format(target_cell["sum_of_turn_when_used"],
                    #                                                            target_cell["used_num"]))
                    contribution_list[player_key].append((key, target_cell["win_num"] / target_cell["used_num"],
                                                          target_cell["used_num"],
                                                          target_cell["sum_of_turn_when_used"] / target_cell[
                                                              "used_num"]))
                if target_cell["drawn_num"] > 0:
                    drawn_win_rate_list[player_key].append(
                        (key, target_cell["win_num_when_drawn"] / target_cell["drawn_num"], target_cell["drawn_num"]))

        contribution_list["Alice"].sort(key=lambda element: -element[1])
        contribution_list["Bob"].sort(key=lambda element: -element[1])
        drawn_win_rate_list["Alice"].sort(key=lambda element: -element[1])
        drawn_win_rate_list["Bob"].sort(key=lambda element: -element[1])
        file_name = ""
        title = "{}({})vs {}({})({} iteration)".format(Player_1.policy.name, deck_id_2_name[player1_deck_num],
                                                       Player_2.policy.name,
                                                       deck_id_2_name[deck_id], iteration)
        mylogger.info(title)
        title += "\n"
        if player1_deck_num != deck_id:
            file_name = "{}_vs_{}'s_WRP_and_WRD.txt".format(deck_id_2_name[player1_deck_num], deck_id_2_name[deck_id])
        else:
            file_name = "{}_mirror_WRP_and_WRD.txt".format(deck_id_2_name[deck_id])
        with open(directory_name + "/" + file_name, mode="w") as f:

            f.write(title)
            f.write(result_txt)
            f.write(first_win_rate_txt + "\n")
            f.write("WRP\n")
            for i, player_key in enumerate(list(contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(contribution_list[player_key]):
                    txt = "No.{} {}(used_num:{},ave_turn:{:.2f}):{:.3f}\n".format(j + 1, cell[0], cell[2], cell[-1],
                                                                                  cell[1])
                    f.write(txt)
                f.write("\n")
            f.write("\nWRD\n")
            for i, player_key in enumerate(list(drawn_win_rate_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j, cell in enumerate(drawn_win_rate_list[player_key]):
                    txt = "No.{} {}(drawn_num:{}):{:.3f}\n".format(j + 1, cell[0], cell[2], cell[1])
                    f.write(txt)
                f.write("\n")

        mylogger.info("{}/{} complete".format((deck_id + 1), len(D)))


def make_mirror_match_table(Player_1, Player_2, iteration,deck_lists=None,pairwise=False,out_put=False):
    if deck_lists is None:
        deck_lists = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    directory_name = None
    if out_put:
        tmp = "{}_vs_{}_pairwise_{}_{}".format(Player_1.policy.name,Player_2.policy.name,
                                                                       deck_lists,iteration)
        time = datetime.datetime.now()
        directory_name = "pairwise_{}_{}_{}_{}_{}_{}".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
        os.makedirs(directory_name, exist_ok=True)
        with open(directory_name + "/" + tmp+".txt", mode="w") as f:
            f.write("this directory is {}\n".format(tmp))
    mylogger.info("deck_list:{}".format(deck_lists))
    for deck_id in deck_lists:
        if pairwise:
            execute_demo_with_pairwise(Player_1, Player_2, iteration, virtual_flg=True, deck_type=[deck_id, deck_id],
                                       output=out_put,directory_name=directory_name)
        else:
            execute_demo(Player_1, Player_2, iteration, virtual_flg=True, deck_type=[deck_id,deck_id], graph=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='対戦実行コード')

    parser.add_argument('--N', help='試行回数',default=1)
    parser.add_argument('--playertype1', help='プレイヤー1のAIタイプ',default=1)
    parser.add_argument('--playertype2', help='プレイヤー2のAIタイプ',default=1)
    parser.add_argument('--playertypes', help='プレイヤーのAIタイプ',default="1,1")
    parser.add_argument('--decktype1', help='プレイヤー1のデッキタイプ',default=0)
    parser.add_argument('--decktype2', help='プレイヤー2のデッキタイプ',default=0)
    parser.add_argument('--decktypes', help='プレイヤーのデッキタイプ', default="0,0")
    parser.add_argument('--filename', help='ファイル名')
    parser.add_argument('--playerlist', help='対戦AIタイプリスト')
    parser.add_argument('--decklist', help='対戦デッキリスト')
    parser.add_argument('--time_bound', help='計算時間上限')
    parser.add_argument('--basic', help='ベーシック')
    parser.add_argument('--graph', help='グラフ')
    parser.add_argument('--pairwise', help='ペアワイズ')
    parser.add_argument('--output', help='出力')
    parser.add_argument('--step_num',help='MCTSの繰り返し上限')
    parser.add_argument('--model_name', help='ニューラルネットワークモデルの名前')
    parser.add_argument('--opponent_model_name', help='対戦相手のニューラルネットワークモデルの名前')
    parser.add_argument('--mode', help='実行モード、demoで対戦画面表示,policyでdecktype固定で各AIタイプの組み合わせで対戦')
    parser.add_argument('--cProfile')
    parser.add_argument('--node_num',default=100,type=int)
    args = parser.parse_args()
    mylogger.info("args:{}".format(args))
    step_num = 100
    if args.step_num is not None:
        step_num = int(args.step_num)
    time_bound = 1.0
    if args.time_bound is not None:
        time_bound = float(args.time_bound)
    Players = []
    Players.append(Player(9, True))  # 1
    Players.append(Player(9, True, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy()))  # 2
    Players.append(Player(9, True, policy=GreedyPolicy(), mulligan=Min_cost_mulligan_policy()))  # 3
    Players.append(Player(9, True, policy=FastGreedyPolicy(), mulligan=Min_cost_mulligan_policy()))  # 4
    Players.append(Player(9, True, policy=MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 5
    Players.append(Player(9, True, policy=Aggro_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 6
    Players.append(Player(9, True, policy=New_Aggro_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 7
    Players.append(Player(9, True, policy=Information_Set_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 8
    Players.append(Player(9, True, policy=Flexible_Iteration_MCTSPolicy(N=step_num), mulligan=Min_cost_mulligan_policy()))  # 9
    Players.append(Player(9, True, policy=Flexible_Iteration_Aggro_MCTSPolicy(N=step_num), mulligan=Min_cost_mulligan_policy()))  # 10
    Players.append(Player(9, True, policy=Flexible_Iteration_Information_Set_MCTSPolicy(N=step_num),mulligan=Min_cost_mulligan_policy()))  # 11
    Players.append(Player(9, True, policy=Opponent_Modeling_MCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 12
    Players.append(Player(9, True, policy=Opponent_Modeling_ISMCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 13
    Players.append(Player(9, True, policy=Default_GreedyPolicy(), mulligan=Simple_mulligan_policy()))  # 14
    Players.append(Player(9, True, policy=Default_Aggro_MCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 15
    Players.append(Player(9, True, policy=Non_Rollout_A_MCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 16
    Players.append(Player(9, True, policy=Non_Rollout_ISMCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 17
    Players.append(Player(9, True, policy=Non_Rollout_OM_MCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 18
    Players.append(Player(9, True, policy=Non_Rollout_OM_ISMCTSPolicy(iteration=step_num), mulligan=Min_cost_mulligan_policy()))  # 19
    Players.append(Player(9, True, policy=Simple_value_function_A_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 20
    Players.append(Player(9, True, policy=Simple_value_function_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 21
    Players.append(Player(9, True, policy=Simple_value_function_OM_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 22
    Players.append(Player(9, True, policy=Simple_value_function_OM_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 23
    Players.append(Player(9, True, policy=New_GreedyPolicy(), mulligan=Simple_mulligan_policy()))  # 24
    Players.append(Player(9, True, policy=until_game_end_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 25
    Players.append(Player(9, True, policy=until_game_end_OM_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 26
    Players.append(Player(9, True, policy=Cheating_MO_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 27
    #Players.append(Player(9, True, policy=until_game_end_OM_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 27
    model_name = None
    if args.model_name is not None:
        model_name = args.model_name
        node_num = int(args.node_num)
        cuda = False#torch.cuda.is_available()
        if args.model_name == "ini":
            origin_model = Embedd_Network_model.New_Dual_Net(args.node_num)
            Players.append(
                Player(9, True, policy=Dual_NN_GreedyPolicy(origin_model=origin_model,node_num=node_num), mulligan=Min_cost_mulligan_policy()))  # 28
            Players.append(
                Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=origin_model,\
                                                                               cuda=cuda,node_num=node_num), mulligan=Min_cost_mulligan_policy())) # 29
        else:
            Players.append(
                Player(9, True, policy=Dual_NN_GreedyPolicy(model_name=model_name,node_num=node_num), mulligan=Min_cost_mulligan_policy()))  # 28
            Players.append(
                Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(model_name=model_name,\
                                                                               cuda=cuda,node_num=node_num), mulligan=Min_cost_mulligan_policy())) # 29

    # assert False
    n = 100
    a = 0
    b = 0
    v = False
    deck_flg = False
    p1, p2 = 0, 0
    human_player = HumanPlayer(9, first=True)
    file_name = None
    if args.filename is not None:
        file_name = args.filename

    import cProfile
    import re
    import datetime

    iteration = n
    d1 = None
    d2 = None

    input_players = [Players[0], Players[1], Players[4], Players[8], Players[9], Players[11], Players[13], Players[14]]
    deck_list = None
    if args.playerlist is not None:
        input_players = []
        tmp = list(args.playerlist.split(","))
        player_id_list = [int(ele) for ele in tmp]
        player_id_list = sorted(list(set(player_id_list)))
        for player_id in player_id_list:
            assert player_id >= 0 and player_id <= len(Players)
            input_players.append(Players[player_id - 1])

    if args.decklist is not None:
        deck_list = []
        tmp = list(args.decklist.split(","))
        deck_id_list = [int(ele) for ele in tmp]
        deck_id_list = sorted(list(set(deck_id_list)))
        for deck_id in deck_id_list:
            deck_list.append(deck_id)
    t1 = datetime.datetime.now()
    mylogger.info("{}".format(t1))
    mylogger.info("mode:{}".format(args.mode))
    if args.mode == 'demo' or args.mode == 'background_demo':
        # cProfile.run('execute_demo(d1,d2,n,deck_type=[p1,p2])')
        n = int(args.N)
        #a = int(args.playertype1) - 1
        #b = int(args.playertype2) - 1
        a,b = map(lambda ele:int(ele)-1,args.playertypes.split(","))
        mylogger.info("a,b:{},{}".format(a,b))
        if a == -1:#if args.playertype1 == '0':
            d1 = HumanPlayer(9, first=True)
        else:
            d1 = copy.deepcopy(Players[a])
        if b == -1:#args.playertype2 == '0':
            d2 = HumanPlayer(9, first=True)

        else:
            d2 = copy.deepcopy(Players[b])

        mylogger.info("d1:{}".format(d1.policy))
        mylogger.info("d2:{}".format(d2.policy))
        #p1 = int(args.decktype1)
        #p2 = int(args.decktype2)
        p1, p2 = map(int, args.decktypes.split(","))
        virtual_flg = args.mode == "background_demo"
        graph = args.graph is not None
        if args.pairwise is not None:
            if args.cProfile is not None:
                cProfile.run('execute_demo_with_pairwise(d1,d2,n,deck_type=[p1,p2],virtual_flg=virtual_flg)',sort="tottime",\
                             filename="profiling.stats")
            else:
                execute_demo_with_pairwise(d1,d2,n,deck_type=[p1,p2],virtual_flg=virtual_flg)
        else:
            if args.cProfile is not None:
                cProfile.run('execute_demo(d1, d2, n, deck_type=[p1, p2],virtual_flg = virtual_flg)',sort="tottime",\
                             filename="profiling.stats")
            else:
                execute_demo(d1, d2, n, deck_type=[p1, p2], virtual_flg=virtual_flg,graph=graph)
    # elif sys.argv[-1]=="-shadow":
    elif args.mode == 'shadow':
        test_3(d1, d2, n)
    # elif sys.argv[-1]=="-policy":
    elif args.mode == 'policy_table':
        # file_name=sys.argv[-2]
        n = int(args.N)
        assert args.decktype1 is not None and args.decktype2 is not None, "deck1:{},deck2:{}".format(args.decktype1, args.decktype2)
        a = int(args.decktype1)
        b = int(args.decktype2)
        if file_name is not None:
            make_policy_table(n, initial_players=input_players, deck_type=[a, b], same_flg=a == b,
                              result_name=file_name)
        else:
            make_policy_table(n, initial_players=input_players, deck_type=[a, b], same_flg=a == b)
        # make_policy_table(n,initial_players=input_players,deck_type=[a+1,b+1],same_flg=a==b,result_name=file_name)
    elif args.mode == 'contribution':
        iteration = int(args.N)
        a = int(args.playertype1) - 1
        b = int(args.playertype2) - 1
        player1_deck_id = int(args.decktype1)
        d1 = copy.deepcopy(Players[a])
        d2 = copy.deepcopy(Players[b])
        get_contributions(d1, d2, iteration, virtual_flg=True, player1_deck_num=player1_deck_id, directory_name=file_name)
    elif args.mode == 'basic':
        iteration = int(args.N)
        a = int(args.playertype1) - 1
        b = int(args.playertype2) - 1
        player1_deck_id = int(args.decktype1)
        d1 = copy.deepcopy(Players[a])
        d2 = copy.deepcopy(Players[b])
        get_basic_contributions(d1, d2, iteration, virtual_flg=True, player1_deck_num=player1_deck_id,
                                directory_name=file_name)
    elif args.mode == 'basic_all':
        import os

        deck_id_2_name = {0: "Forest", 1: "Sword", 2: "Rune", 3: "Dragon", 4: "Shadow", 5: "Blood",
                          6: "Haven", 7: "Portal"}

        iteration = int(args.N)
        a = int(args.playertype1) - 1
        b = int(args.playertype2) - 1
        d1 = copy.deepcopy(Players[a])
        d2 = copy.deepcopy(Players[b])
        short_name_1 = list(d1.policy.name.split("Policy"))[0]
        short_name_2 = list(d2.policy.name.split("Policy"))[0]
        path = "{}_vs_{}(basic_all)_{}times_WRP_and_WRD".format(short_name_1, short_name_2, iteration)
        os.makedirs(path)
        for i in range(8):
            player1_deck_id = i
            next_path = "/WRP_and_WRD_{}".format(deck_id_2_name[i])
            os.makedirs(path + next_path)
            file_name = path + next_path
            get_basic_contributions(d1, d2, iteration, virtual_flg=True, player1_deck_num=player1_deck_id,
                                    directory_name=file_name)
    elif args.mode == 'custom_all':
        deck_id_2_name = {0: "Aggro-Sword", 1: "Rune_Earth(Aggro)", 2: "Mid-Sword", 3: "Mid-Shadow", 4: "Dragon_PDK(Mid)",
                          5: "Elana-Haven(Control)",
                          6: "Control-Blood", 7: "Ramp-Dragon(Combo)", 8: "Combo-Forest", 9: "Spell-Rune(Combo)"}

        iteration = int(args.N)
        a = int(args.playertype1) - 1
        b = int(args.playertype2) - 1
        d1 = copy.deepcopy(Players[a])
        d2 = copy.deepcopy(Players[b])
        short_name_1 = list(d1.policy.name.split("Policy"))[0]
        short_name_2 = list(d2.policy.name.split("Policy"))[0]
        path = "{}_vs_{}(custom_all)_{}times_WRP_and_WRD".format(short_name_1, short_name_2, iteration)
        os.makedirs(path,exist_ok=True)
        for i in range(10):
            player1_deck_id = i
            next_path = "/WRP_and_WRD_{}".format(deck_id_2_name[i])
            if os.path.isdir(path + next_path):
                mylogger.info("{} already exists. skip this step.".format(next_path))
                continue
            os.makedirs(path + next_path)
            file_name = path + next_path
            get_custom_contributions(d1, d2, iteration, virtual_flg=True, player1_deck_num=player1_deck_id,
                                     directory_name=file_name)


    elif args.mode == "mirror":
        n = int(args.N)
        a,b = map(lambda ele:int(ele)-1,args.playertypes.split(","))
        if args.playertype1 == '0':
            d1 = copy.deepcopy(human_player)
        else:
            d1 = copy.deepcopy(Players[a])
        if args.playertype2 == '0':
            d2 = copy.deepcopy(human_player)
        else:
            d2 = Player(9, True,
                        policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(
                            model_name=args.opponent_model_name,node_num=node_num),
                        mulligan=Min_cost_mulligan_policy())\
            if args.opponent_model_name is not None else copy.deepcopy(Players[b])
            #d2 = copy.deepcopy(Players[b])

        make_mirror_match_table(d1,d2,n,deck_lists=deck_list,pairwise=args.pairwise is not None,
                                out_put=args.output is not None)
    elif args.mode == "random_match":
        n = int(args.N)
        a = int(args.playertype1) - 1
        b = int(args.playertype2) - 1
        if args.playertype1 == '0':
            d1 = copy.deepcopy(human_player)
        else:
            d1 = copy.deepcopy(Players[a])
        if args.playertype2 == '0':
            d2 = copy.deepcopy(human_player)
        else:
            d2 = copy.deepcopy(Players[b])

        random_match(d1, d2, n, virtual_flg=True)

    elif args.mode == "deck_table":
        if args.N is not None:
            iteration = int(args.N)
            a = int(args.playertype1) - 1
            b = int(args.playertype2) - 1
            d1 = copy.deepcopy(Players[a])
            d2 = copy.deepcopy(Players[b])
            basic_flg = False
            if args.basic is not None:
                basic_flg = True
            #if a == b:
            if False:
                make_deck_table(d1, d2, iteration, same_flg=True, result_name=file_name, basic=basic_flg,deck_lists=deck_list)
            else:
                make_deck_table(d1, d2, iteration, result_name=file_name, basic=basic_flg,deck_lists=deck_list)
    else:
        assert False
    mylogger.info(t1)
    t2 = datetime.datetime.now()
    mylogger.info(t2)
    mylogger.info(t2 - t1)
