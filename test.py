# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import math
import copy
from card_setting import *
from Field_setting import Field
from Player_setting import *
from Game_setting import Game
from my_moduler import get_module_logger, get_state_logger
from mulligan_setting import *
import logging
import matplotlib.pyplot as plt

mylogger = get_module_logger(__name__)
from my_enum import *
import argparse
import csv

value_history = []


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
    for card in D1.deck:
        f.players[0].deck.deck.append(card.get_copy())
    f.players[1].deck = Deck()
    f.players[1].deck.set_leader_class(D2.leader_class.name)
    for card in D2.deck:
        f.players[1].deck.deck.append(card.get_copy())

    f.players[0].deck.shuffle()
    f.players[1].deck.shuffle()
    f.players[0].draw(f.players[0].deck, 3)
    f.players[1].draw(f.players[1].deck, 3)
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
    f.players[1].life = 20
    f.players[1].hand.clear()
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
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
    for card in D1.deck:
        f.players[0].deck.deck.append(card.get_copy())
    f.players[1].deck = Deck()
    f.players[1].deck.set_leader_class(D2.leader_class.name)
    for card in D2.deck:
        f.players[1].deck.deck.append(card.get_copy())

    f.players[0].deck.shuffle()
    f.players[1].deck.shuffle()
    f.players[0].draw(f.players[0].deck, 3)
    f.players[1].draw(f.players[1].deck, 3)
    # f.evo_point = [1,2]
    G = Game()

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


def execute_demo(Player_1, Player_2, iteration, virtual_flg=False, deck_type=None,graph=False):
    Player1 = copy.deepcopy(Player_1)
    Player2 = copy.deepcopy(Player_2)
    Player1 = Player_1
    Player2 = Player_2
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
                      -9: "Spell-Rune",11:"PtP-Forest",12:"Mid-Shadow"}

    key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                      3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                      6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                      9: ["SpellBoost-Rune.tsv", "RUNE"],10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                      11: ["PtP_Forest.tsv", "FOREST"],12: ["Mid_Shadow.tsv", "SHADOW"]}
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    class_pool = [0, 0]
    for i, d in enumerate(D):
        if deck_type[i] in key_2_tsv_name:
            D[i] = tsv_to_deck(key_2_tsv_name[deck_type[i]][0])
            D[i].set_leader_class(key_2_tsv_name[deck_type[i]][1])
        """
        if deck_type[i] == 0:
            D[i] = tsv_to_deck("Sword_Aggro.tsv")
            D[i].set_leader_class("SWORD")
            # Aggro
        elif deck_type[i] == 1:
            D[i] = tsv_to_deck("Rune_Earth.tsv")
            D[i].set_leader_class("RUNE")
            # Aggro
        elif deck_type[i] == 2:
            D[i] = tsv_to_deck("Sword.tsv")
            D[i].set_leader_class("SWORD")
            # Mid
        elif deck_type[i] == 3:
            D[i] = tsv_to_deck("New-Shadow.tsv")
            D[i].set_leader_class("SHADOW")
            # Mid
        elif deck_type[i] == 4:
            D[i] = tsv_to_deck("Dragon_PDK.tsv")
            D[i].set_leader_class("DRAGON")
            # Mid
        elif deck_type[i] == 5:
            D[i] = tsv_to_deck("Haven.tsv")
            D[i].set_leader_class("HAVEN")
            # Control
        elif deck_type[i] == 6:
            D[i] = tsv_to_deck("Blood.tsv")
            D[i].set_leader_class("BLOOD")
            # Control
        elif deck_type[i] == 7:
            D[i] = tsv_to_deck("Dragon.tsv")
            D[i].set_leader_class("DRAGON")
            # Control
        elif deck_type[i] == 8:
            D[i] = tsv_to_deck("Forest.tsv")
            D[i].set_leader_class("FOREST")
            # Combo
        elif deck_type[i] == 9:
            D[i] = tsv_to_deck("Rune.tsv")
            D[i].set_leader_class("RUNE")
            # Combo
        """
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

    import itertools
    if Player1.mulligan_policy.data_use_flg:
        mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player1.mulligan_policy.mulligan_data,
                                                                            Player1.mulligan_policy.win_data)))))
    if Player2.mulligan_policy.data_use_flg:
        mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player2.mulligan_policy.mulligan_data,
                                                                            Player2.mulligan_policy.win_data)))))
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
    if Player1.policy.name.split("_")[0] == "Genetic":
        for fit_key in sorted(list(Player1.policy.better_parameters), key=lambda ele: -ele[0]):
            mylogger.info("{}:{}".format(fit_key[1], fit_key[0]))
    if "MAST" in Player1.policy.name:
        for key in list(Player1.policy.Q_dict.keys()):
            n = Player1.policy.Q_dict[key][1]
            mean_value = Player1.policy.Q_dict[key][0] / max(n, 1)
            mylogger.info("{}".format((Action_Code(key[0]).name, key[1])))
            mylogger.info("n:{:<5} mean_value:{:.3f}".format(n, mean_value))
    if len(value_history) > 9:
        fig = plt.figure()
        mylogger.info("value_history[0]:{}".format(value_history[0]))
        # p1,p2 = None,None
        ave_p1 = {}
        ave_p2 = {}
        max_len = 0
        max_turn = 40
        end_turn_counter = []
        for i in range(max_turn):
            ave_p1[i] = []
            ave_p2[i] = []
        for i, data in enumerate(value_history):
            if i % 2 == 0:
                # p1=plt.plot(data[0], data[1], color="red")
                for j in range(max_turn):
                    if j < len(data[1]):
                        ave_p1[j].append(data[1][j])
                    else:
                        ave_p1[j].append(data[1][-1])
            else:
                # p2=plt.plot(data[0], data[1], color="blue")
                for j in range(max_turn):
                    if j < len(data[1]):
                        ave_p2[j].append(data[1][j])
                    else:
                        ave_p2[j].append(data[1][-1])
            max_len = max(max_len, len(data[1]))
            end_turn_counter.append(len(data[1]) - 1)
        mean_p1 = []
        std_p1 = []
        mean_p2 = []
        std_p2 = []
        # for i in range(80):
        for i in range(max_len):
            mean_p1.append(sum(ave_p1[i]) / (iteration // 2))
            std_p1.append(np.std(ave_p1[i], ddof=1) / (iteration // 2))
            mean_p2.append(sum(ave_p2[i]) / (iteration // 2))
            std_p2.append(np.std(ave_p2[i], ddof=1) / (iteration // 2))
        max_data = np.array(list(range(max_len)))
        # max_data = np.array(list(range(80)))

        mean_p1 = np.array(mean_p1[:max_len])
        mean_p2 = np.array(mean_p2[:max_len])
        # mean_p1 = np.array(mean_p1)
        # mean_p2 = np.array(mean_p2)
        ax1 = fig.add_subplot(1, 1, 1)
        p3 = ax1.plot(max_data, mean_p1, color="c", linewidth=3)
        p4 = ax1.plot(max_data, mean_p2, color="m", linewidth=3)
        # p3 = plt.plot(max_data, mean_p1, color="c",linewidth=3)
        # p4 = plt.plot(max_data, mean_p2, color="m",linewidth=3)
        # markersize = 10
        ax1.errorbar(max_data, mean_p1, yerr=std_p1,
                     capsize=5, fmt='o', ecolor='black',
                     markeredgecolor="black",
                     color='w')
        ax1.errorbar(max_data, mean_p2, yerr=std_p2,
                     capsize=5, fmt='o', ecolor='black',
                     markeredgecolor="black",
                     color='w')
        ax2 = ax1.twinx()
        # mylogger.info("end_turn_couner:{}".format(end_turn_counter))
        edges = range(0, max_len, 1)
        ax2.hist(end_turn_counter, bins=edges, normed=True, alpha=0.5)
        ax2.set_title('value graph/end_turn histogram')
        ax1.set_xlabel('turn')
        ax1.set_ylabel('value')
        ax2.set_ylabel('freq')
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])
        ax1.set_xlim([0, max_len])
        # mylogger.info("std_p1:{},std_p2:{}".format(std_p1,std_p2))
        # plt.legend((p1[0], p2[0],p3[0],p4[0]), ("first", "second","ave_first","ave_second"), loc=0)
        ax1.legend((p3[0], p4[0]), ("ave_first", "ave_second"), loc=0)
        fig.savefig("graph/{}({})_{}({})_{}iteration.png".format(Player1.policy.name, deck_id_2_name[deck_type[0]],
                                                                 Player2.policy.name, deck_id_2_name[deck_type[1]],
                                                                 iteration))
        plt.show()


def get_contributions(Player_1, Player_2, iteration, virtual_flg=False, player1_deck_num=None, directory_name=None):
    assert player1_deck_num is not None
    assert directory_name is not None
    Player1 = copy.deepcopy(Player_1)
    Player2 = copy.deepcopy(Player_2)
    Player1 = Player_1
    Player2 = Player_2
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
    Player1 = copy.deepcopy(Player_1)
    Player2 = copy.deepcopy(Player_2)
    Player1 = Player_1
    Player2 = Player_2
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
    Player1 = copy.deepcopy(Player_1)
    Player2 = copy.deepcopy(Player_2)
    Player1 = Player_1
    Player2 = Player_2
    Player1.name = "Alice"
    Player2.name = "Bob"
    if result_name is None:
        result_name = "{}_vs_{}_{}iteration(deck_list={}).tsv".format(Player_1.policy.name, Player_2.policy.name, iteration,deck_lists)
    assert Player1 != Player2
    win = 0
    lose = 0
    lib_num = 0
    D = []

    deck_id_2_name = {}
    if not basic:
        # -1: "Forest_Basic",
        #                  -2: "Sword_Basic",
        #                  -3: "Rune_Basic",
        #                  -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic",
        #                  -8: "Portal_Basic",-9: "Spell-Rune",100: "Test",
        deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                          6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune",

                           11: "PtP-Forest", 12: "Mid-Shadow"}
        key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                          3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"],
                          5: ["Test-Haven.tsv", "HAVEN"],
                          6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                          9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                          11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"]}

        if deck_lists is None:
            deck_lists = list(deck_id_2_name.keys())
        else:
            assert all(key in deck_id_2_name for key in deck_lists)
        mylogger.info("deck_lists:{}".format(deck_lists))
        D = [Deck() for i in range(len(deck_lists))]
        deck_index = 0
        sorted_keys = sorted(list(deck_id_2_name.keys()))
        for i in sorted_keys:
            if i not in deck_lists:
                continue
            mylogger.info("{}(deck_id:{}):{}".format(deck_index,i,key_2_tsv_name[i]))
            D[deck_index] = tsv_to_deck(key_2_tsv_name[i][0])
            D[deck_index].set_leader_class(key_2_tsv_name[i][1])
            deck_index += 1

        """
        for i in list(deck_id_2_name.keys()):
            if i not in deck_lists:
                continue
            if i == 0:
                D[i] = tsv_to_deck("Sword_Aggro.tsv")
                #D[i] = tsv_to_deck("Sword_Aggro.tsv")
                #D[i].set_leader_class("SWORD")
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
    """
    """
    else:
        deck_id_2_name = {0: "Forest", 1: "Sword", 2: "Rune", 3: "Dragon", 4: "Shadow", 5: "Blood",
                          6: "Haven", 7: "Portal"}
        D = [Deck() for i in range(len(deck_id_2_name))]
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
    """
    # D[0].mean_cost=D[0].get_mean_cost()
    # D[1].mean_cost=D[1].get_mean_cost()
    assert all(len(D[i].deck) == 40 for i in range(len(D)))
    # Turn_Players=[Player1,Player2]
    Results = {}
    mylogger.info("same_flg:{}".format(same_flg))
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
    Player1 = copy.deepcopy(Player_1)
    Player2 = copy.deepcopy(Player_2)
    Player1 = Player_1
    Player2 = Player_2
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
                      -9: "Spell-Rune", -10: "Test-Haven",-11:"PtP-Forest",-12:"Mid-Shadow"}
    key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                      3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                      6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                      9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                      -11: ["PtP_Forest.tsv", "FOREST"], -12: ["Mid_Shadow.tsv", "SHADOW"]}

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
    players = copy.deepcopy(initial_players)
    D = [Deck() for i in range(2)]
    for i, d in enumerate(D):
        if deck_type[i] in key_2_tsv_name:
            D[i] = tsv_to_deck(key_2_tsv_name[deck_type[i]][0])
            D[i].set_leader_class(key_2_tsv_name[deck_type[i]][1])
    """
    for i, d in enumerate(D):
        if deck_type[i] == 0:
            D[i] = tsv_to_deck("Sword_Aggro.tsv")
            D[i].set_leader_class("SWORD")
            # Aggro
        elif deck_type[i] == 1:
            D[i] = tsv_to_deck("Rune_Earth.tsv")
            D[i].set_leader_class("RUNE")
            # Aggro
        elif deck_type[i] == 2:
            D[i] = tsv_to_deck("Sword.tsv")
            D[i].set_leader_class("SWORD")
            # Mid
        elif deck_type[i] == 3:
            # D[i]=tsv_to_deck("Shadow.tsv")
            D[i] = tsv_to_deck("New-Shadow.tsv")
            D[i].set_leader_class("SHADOW")
            # Mid
        elif deck_type[i] == 4:
            D[i] = tsv_to_deck("Dragon_PDK.tsv")
            D[i].set_leader_class("DRAGON")
            # Mid
        elif deck_type[i] == 5:
            D[i] = tsv_to_deck("Test-Haven.tsv")
            D[i].set_leader_class("HAVEN")
            # D[i] = tsv_to_deck("Haven.tsv")
            # D[i].set_leader_class("HAVEN")
            # Control
        elif deck_type[i] == 6:
            D[i] = tsv_to_deck("Blood.tsv")
            D[i].set_leader_class("BLOOD")
            # Control
        elif deck_type[i] == 7:
            D[i] = tsv_to_deck("Dragon.tsv")
            D[i].set_leader_class("DRAGON")
            # Control
        elif deck_type[i] == 8:
            D[i] = tsv_to_deck("Forest.tsv")
            D[i].set_leader_class("FOREST")
            # Combo
        elif deck_type[i] == 9:
            D[i] = tsv_to_deck("Rune.tsv")
            D[i].set_leader_class("RUNE")
            # Combo
        
        if deck_type[i] == -1:
            D[i] = tsv_to_deck("Forest_Basic.tsv")
            D[i].set_leader_class("FOREST")
        elif deck_type[i] == -2:
            D[i] = tsv_to_deck("Sword_Basic.tsv")
            D[i].set_leader_class("SWORD")
        elif deck_type[i] == -3:
            D[i] = tsv_to_deck("Rune_Basic.tsv")
            D[i].set_leader_class("RUNE")
    """

    Results = {}
    for policy1_id, player1 in enumerate(players):
        P1 = copy.deepcopy(player1)
        P1.name = "Alice"
        last_id = len(players)
        for policy2_id in range(0, last_id):
            player2 = players[policy2_id]
            P2 = copy.deepcopy(player2)
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
        if True:
            for i in range(len(policy_id_2_name)):
                row = [policy_id_2_name[i]]
                for j in range(len(policy_id_2_name)):
                    row.append(Results[(i, j)][0])
                writer.writerow(row)


def get_custom_contributions(Player_1, Player_2, iteration, virtual_flg=False, player1_deck_num=None,
                             directory_name=None):
    assert player1_deck_num is not None
    assert directory_name is not None
    Player1 = copy.deepcopy(Player_1)
    Player2 = copy.deepcopy(Player_2)
    Player1 = Player_1
    Player2 = Player_2
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

def make_mirror_match_table(Player_1, Player_2, iteration,deck_lists=None):
    if deck_lists is None:
        deck_lists = [0,1,4,5,6,7,9,10,11,12]
    mylogger.info("deck_list:{}".format(deck_lists))
    for deck_id in deck_lists:
        execute_demo(Player_1, Player_2, iteration, virtual_flg=True, deck_type=[deck_id,deck_id], graph=False)

parser = argparse.ArgumentParser(description='対戦実行コード')

parser.add_argument('--N', help='試行回数')
parser.add_argument('--playertype1', help='プレイヤー1のAIタイプ')
parser.add_argument('--playertype2', help='プレイヤー2のAIタイプ')
parser.add_argument('--decktype1', help='プレイヤー1のデッキタイプ')
parser.add_argument('--decktype2', help='プレイヤー2のデッキタイプ')
parser.add_argument('--filename', help='ファイル名')
parser.add_argument('--playerlist', help='対戦AIタイプリスト')
parser.add_argument('--decklist', help='対戦デッキリスト')
parser.add_argument('--time_bound', help='計算時間上限')
parser.add_argument('--basic', help='ベーシック')
parser.add_argument('--graph', help='グラフ')
parser.add_argument('--mode', help='実行モード、demoで対戦画面表示,policyでdecktype固定で各AIタイプの組み合わせで対戦')
args = parser.parse_args()
mylogger.info("args:{}".format(args))

Players = []
Players.append(Player(9, True))  # 1
Players.append(Player(9, True, policy=AggroPolicy()))  # 2
Players.append(Player(9, True, policy=GreedyPolicy()))  # 3
Players.append(Player(9, True, policy=FastGreedyPolicy()))  # 4
Players.append(Player(9, True, policy=GreedyPolicy(), mulligan=Simple_mulligan_policy()))  # 5
Players.append(Player(9, True, policy=FastGreedyPolicy(), mulligan=Simple_mulligan_policy()))  # 6
Players.append(Player(9, True, policy=GreedyPolicy(), mulligan=Min_cost_mulligan_policy()))  # 7
Players.append(Player(9, True, policy=FastGreedyPolicy(), mulligan=Min_cost_mulligan_policy()))  # 8
Players.append(Player(9, True, policy=MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 9
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 10
Players.append(Player(9, True, policy=Test_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 11
Players.append(Player(9, True, policy=Test_2_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 12
Players.append(Player(9, True, policy=Test_3_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 13
Players.append(Player(9, True, policy=Aggro_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 14
Players.append(Player(9, True, policy=EXP3_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 15
Players.append(Player(9, True, policy=New_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 16
Players.append(Player(9, True, policy=New_Aggro_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 17
Players.append(Player(9, True, policy=Aggro_EXP3_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 18
Players.append(Player(9, True, policy=Alpha_Beta_MCTSPolicy(th=10), mulligan=Min_cost_mulligan_policy()))  # 19
Players.append(Player(9, True, policy=Alpha_Beta_MCTSPolicy(th=20), mulligan=Min_cost_mulligan_policy()))  # 20
Players.append(Player(9, True, policy=Alpha_Beta_MCTSPolicy(th=25), mulligan=Min_cost_mulligan_policy()))  # 21
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(th=10), mulligan=Min_cost_mulligan_policy()))  # 22
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(th=20), mulligan=Min_cost_mulligan_policy()))  # 23
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(th=25), mulligan=Min_cost_mulligan_policy()))  # 24
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(th=33), mulligan=Min_cost_mulligan_policy()))  # 25
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(th=5), mulligan=Min_cost_mulligan_policy()))  # 26
Players.append(Player(9, True, policy=Shallow_MCTSPolicy(th=3), mulligan=Min_cost_mulligan_policy()))  # 27
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.1), mulligan=Min_cost_mulligan_policy()))  # 28
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.25), mulligan=Min_cost_mulligan_policy()))  # 29
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.5), mulligan=Min_cost_mulligan_policy()))  # 30
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=1.0), mulligan=Min_cost_mulligan_policy()))  # 31
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.1, playout=AggroPolicy()),
                      mulligan=Min_cost_mulligan_policy()))  # 32
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.25, playout=AggroPolicy()),
                      mulligan=Min_cost_mulligan_policy()))  # 33
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.5, playout=AggroPolicy()),
                      mulligan=Min_cost_mulligan_policy()))  # 34
Players.append(Player(9, True, policy=Aggro_Shallow_MCTSPolicy(th=3), mulligan=Min_cost_mulligan_policy()))  # 35
Players.append(Player(9, True, policy=Expanded_Aggro_MCTS_Policy(), mulligan=Min_cost_mulligan_policy()))  # 36
Players.append(Player(9, True, policy=Genetic_GreedyPolicy(N=10), mulligan=Min_cost_mulligan_policy()))  # 37
Players.append(Player(9, True, policy=Genetic_GreedyPolicy(N=20), mulligan=Min_cost_mulligan_policy()))  # 38
Players.append(Player(9, True, policy=Genetic_New_GreedyPolicy(N=10), mulligan=Min_cost_mulligan_policy()))  # 39
Players.append(Player(9, True, policy=Genetic_New_GreedyPolicy(N=20), mulligan=Min_cost_mulligan_policy()))  # 40
Players.append(Player(9, True, policy=Genetic_Aggro_MCTSPolicy(N=10), mulligan=Min_cost_mulligan_policy()))  # 41
Players.append(Player(9, True, policy=Information_Set_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 42
Players.append(Player(9, True, policy=Flexible_Iteration_MCTSPolicy(N=100), mulligan=Min_cost_mulligan_policy()))  # 43
Players.append(Player(9, True, policy=Flexible_Iteration_MCTSPolicy(N=250), mulligan=Min_cost_mulligan_policy()))  # 44
Players.append(Player(9, True, policy=Flexible_Iteration_MCTSPolicy(N=500), mulligan=Min_cost_mulligan_policy()))  # 45
Players.append(
    Player(9, True, policy=Flexible_Iteration_Aggro_MCTSPolicy(N=100), mulligan=Min_cost_mulligan_policy()))  # 46
Players.append(
    Player(9, True, policy=Flexible_Iteration_Aggro_MCTSPolicy(N=250), mulligan=Min_cost_mulligan_policy()))  # 47
Players.append(
    Player(9, True, policy=Flexible_Iteration_Aggro_MCTSPolicy(N=500), mulligan=Min_cost_mulligan_policy()))  # 48
Players.append(Player(9, True, policy=Flexible_Iteration_Information_Set_MCTSPolicy(N=100),
                      mulligan=Min_cost_mulligan_policy()))  # 49
Players.append(Player(9, True, policy=Flexible_Iteration_Information_Set_MCTSPolicy(N=250),
                      mulligan=Min_cost_mulligan_policy()))  # 50
Players.append(Player(9, True, policy=Flexible_Iteration_Information_Set_MCTSPolicy(N=500),
                      mulligan=Min_cost_mulligan_policy()))  # 51
time_bound = 1.0
if args.time_bound is not None:
    time_bound = float(args.time_bound)
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=time_bound, playout=AggroPolicy()),
                      mulligan=Min_cost_mulligan_policy()))  # 52
Players.append(Player(9, True, policy=Time_bounded_Information_Set_MCTSPolicy(limit=time_bound),
                      mulligan=Min_cost_mulligan_policy()))  # 53
Players.append(Player(9, True, policy=Test_4_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 54
Players.append(Player(9, True, policy=Opponent_Modeling_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 55
Players.append(Player(9, True, policy=Improved_Aggro_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 56
Players.append(Player(9, True, policy=Opponent_Modeling_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 57
Players.append(
    Player(9, True, policy=Opponent_Modeling_ISMCTSPolicy(iteration=250), mulligan=Min_cost_mulligan_policy()))  # 58
Players.append(
    Player(9, True, policy=Opponent_Modeling_ISMCTSPolicy(iteration=500), mulligan=Min_cost_mulligan_policy()))  # 59
Players.append(
    Player(9, True, policy=Opponent_Modeling_ISMCTSPolicy(iteration=50), mulligan=Min_cost_mulligan_policy()))  # 60
Players.append(
    Player(9, True, policy=Flexible_Iteration_Aggro_MCTSPolicy(N=50), mulligan=Min_cost_mulligan_policy()))  # 61
Players.append(
    Player(9, True, policy=Alter_Opponent_Modeling_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 62
Players.append(Player(9, True, policy=Neo_MCTSPolicy(probability=0.1), mulligan=Min_cost_mulligan_policy()))  # 63
Players.append(Player(9, True, policy=Neo_MCTSPolicy(probability=0.2), mulligan=Min_cost_mulligan_policy()))  # 64
Players.append(Player(9, True, policy=Neo_MCTSPolicy(probability=0.25), mulligan=Min_cost_mulligan_policy()))  # 65
Players.append(Player(9, True, policy=Neo_MCTSPolicy(probability=0.50), mulligan=Min_cost_mulligan_policy()))  # 66
Players.append(Player(9, True, policy=Neo_OM_ISMCTSPolicy(probability=0.1), mulligan=Min_cost_mulligan_policy()))  # 67
Players.append(Player(9, True, policy=Neo_OM_ISMCTSPolicy(probability=0.2), mulligan=Min_cost_mulligan_policy()))  # 68
Players.append(Player(9, True, policy=Neo_OM_ISMCTSPolicy(probability=0.25), mulligan=Min_cost_mulligan_policy()))  # 69
Players.append(Player(9, True, policy=Neo_OM_ISMCTSPolicy(probability=0.50), mulligan=Min_cost_mulligan_policy()))  # 70
Players.append(Player(9, True, policy=Damped_Sampling_MCTS(), mulligan=Min_cost_mulligan_policy()))  # 71
Players.append(Player(9, True, policy=Damped_Sampling_ISMCTS(), mulligan=Min_cost_mulligan_policy()))  # 72
Players.append(Player(9, True, policy=Sampling_ISMCTS(), mulligan=Min_cost_mulligan_policy()))  # 73
Players.append(Player(9, True, policy=Simple_value_function_A_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 74
Players.append(
    Player(9, True, policy=Simple_value_function_OM_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 75
Players.append(Player(9, True, policy=Simple_value_function_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 76
Players.append(Player(9, True, policy=Second_value_function_A_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 77
Players.append(
    Player(9, True, policy=Second_value_function_OM_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 78
Players.append(Player(9, True, policy=Flexible_Simulation_A_MCTSPolicy(sim_num=1), mulligan=Min_cost_mulligan_policy()))  # 79
Players.append(Player(9, True, policy=Flexible_Simulation_A_MCTSPolicy(sim_num=5), mulligan=Min_cost_mulligan_policy()))  # 80
Players.append(Player(9, True, policy=Flexible_Simulation_MO_ISMCTSPolicy(sim_num=1), mulligan=Min_cost_mulligan_policy()))  # 81
Players.append(Player(9, True, policy=Flexible_Simulation_MO_ISMCTSPolicy(sim_num=5), mulligan=Min_cost_mulligan_policy()))  # 82
Players.append(Player(9, True, policy=Cheating_MO_MCTSPolicy(iteration=100), mulligan=Min_cost_mulligan_policy()))  #83
Players.append(Player(9, True, policy=Cheating_MO_ISMCTSPolicy(iteration=100), mulligan=Min_cost_mulligan_policy()))  #84
Players.append(Player(9, True, policy=New_Aggro_MCTSPolicy(iteration=250), mulligan=Min_cost_mulligan_policy()))  # 85
Players.append(Player(9, True, policy=New_Aggro_MCTSPolicy(iteration=500), mulligan=Min_cost_mulligan_policy()))  # 86
Players.append(Player(9, True, policy=Advanced_value_function_A_MCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 87
Players.append(Player(9, True, policy=Advanced_value_function_OM_ISMCTSPolicy(), mulligan=Min_cost_mulligan_policy()))  # 88
Players.append(Player(9, True, policy=Opponent_Modeling_MCTSPolicy(iteration=250), mulligan=Min_cost_mulligan_policy()))  # 89
Players.append(Player(9, True, policy=Cheating_MO_MCTSPolicy(iteration=250), mulligan=Min_cost_mulligan_policy()))  #90
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
    p1 = int(args.decktype1)
    p2 = int(args.decktype2)
    virtual_flg = args.mode == "background_demo"
    graph = args.graph is not None
    #cProfile.run('execute_demo(d1, d2, n, deck_type=[p1, p2],virtual_flg = virtual_flg)',sort="tottime")
    execute_demo(d1, d2, n, deck_type=[p1, p2], virtual_flg=virtual_flg,graph=graph)
# elif sys.argv[-1]=="-shadow":
elif args.mode == 'shadow':
    test_3(d1, d2, n)
# elif sys.argv[-1]=="-policy":
elif args.mode == 'policy':
    # file_name=sys.argv[-2]
    n = int(args.N)
    assert args.decktype1 is not None and args.decktype2 is not None, "deck1:{},deck2:{}".format(args.decktype1, args.decktype2)
    a = int(args.decktype1)
    b = int(args.decktype2)
    make_policy_table(n, initial_players=input_players, deck_type=[a, b], same_flg=a == b,
                      result_name=file_name)
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
    import os

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
    #p1 = int(args.decktype1)
    #p2 = int(args.decktype2)
    #graph = args.graph is not None
    #execute_demo(d1, d2, n, deck_type=[p1, p2], virtual_flg=True,graph=False)
    make_mirror_match_table(d1,d2,n,deck_lists=deck_list)
else:
    if args.N is not None:
        iteration = int(args.N)
        a = int(args.playertype1) - 1
        b = int(args.playertype2) - 1
        d1 = copy.deepcopy(Players[a])
        d2 = copy.deepcopy(Players[b])
        basic_flg = False
        if args.basic is not None:
            basic_flg = True
        if a == b:
            make_deck_table(d1, d2, iteration, same_flg=True, result_name=file_name, basic=basic_flg,deck_lists=deck_list)
        else:
            make_deck_table(d1, d2, iteration, result_name=file_name, basic=basic_flg,deck_lists=deck_list)
mylogger.info(t1)
t2 = datetime.datetime.now()
mylogger.info(t2)
mylogger.info(t2 - t1)
