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

mylogger = get_module_logger(__name__)
from my_enum import *
import argparse
import csv


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


def game_play(Player1, Player2, D1, D2, win, lose, lib_num, virtual_flg=False,deck_name_list=None):
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
    if virtual_flg == False:
        mylogger.info("Game end")
        mylogger.info("Player1 life:{} Player2 life:{}".format(f.players[0].life, f.players[1].life))
        f.show_field()
    player1_win_turn = False
    player2_win_turn = False
    if w==1:
        player1_win_turn=turn
    else:
        player2_win_turn=turn


    f.players[0].life = 20
    f.players[0].hand = []
    f.players[0].deck = None
    f.players[0].lib_out_flg = False
    f.players[1].life = 20
    f.players[1].hand = []
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
    if virtual_flg == False:
        f.graveyard.show_graveyard()
        f.play_cards.show_play_list()
    return win, lose, lib_num, turn, first,(player1_win_turn,player2_win_turn)

def demo_game_play(Player1, Player2, D1, D2, win, lose, lib_num, virtual_flg=False,deck_name_list=None):
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
    if virtual_flg == False:
        mylogger.info("Game end")
        mylogger.info("Player1 life:{} Player2 life:{}".format(f.players[0].life, f.players[1].life))
        f.show_field()
    player1_win_turn = False
    player2_win_turn = False
    if w==1:
        player1_win_turn=turn
    else:
        player2_win_turn=turn


    f.players[0].life = 20
    f.players[0].hand = []
    f.players[0].deck = None
    f.players[0].lib_out_flg = False
    f.players[1].life = 20
    f.players[1].hand = []
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
    #if virtual_flg == False:
    #    f.graveyard.show_graveyard()
    #    f.play_cards.show_play_list()

    if deck_name_list is not None:
        f.play_cards.play_cards_set()
        win_flg = [w,l]

        for i in range(2):
            for cost_key in list(f.play_cards.name_list[i].keys()):
                for category_key in list(f.play_cards.name_list[i][cost_key].keys()):
                    for name_key in list(f.play_cards.name_list[i][cost_key][category_key].keys()):
                        if name_key in deck_name_list[f.players[i].name]:
                            deck_name_list[f.players[i].name][name_key]["used_num"] += 1
                            deck_name_list[f.players[i].name][name_key]["win_num"] += win_flg[i]
        #mylogger.info(deck_name_list)


    return win, lose, lib_num, turn, first,(player1_win_turn,player2_win_turn)

# import numba
# @numba.jit
def execute_demo(Player_1, Player_2, iteration, virtual_flg=False, deck_type=None):
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
    if deck_type == None:
        deck_type = [5, 5]
    else:
        mylogger.info("deck_type:{}".format(deck_type))
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune",-1:"TEST"}
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    class_pool = [0, 0]
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
        elif deck_type[i] == 10:
            D[i] = tsv_to_deck("New-Shadow.tsv")
            # Combo
        elif deck_type[i] == -1:
            #テスト
            #D[i] = tsv_to_deck("Forest_Basic.tsv")
            #D[i] = tsv_to_deck("Sword_Basic.tsv")
            #D[i] = tsv_to_deck("Rune_Basic.tsv")
            #D[i] = tsv_to_deck("Dragon_Basic.tsv")
            #D[i] = tsv_to_deck("Shadow_Basic.tsv")
            #D[i] = tsv_to_deck("Blood_Basic.tsv")
            #D[i] = tsv_to_deck("Haven_Basic.tsv")
            D[i] = tsv_to_deck("Portal_Basic.tsv")
            D[i].set_leader_class("PORTAL")

    Player1.class_num = class_pool[0]
    Player2.class_num = class_pool[1]
    mylogger.info("Alice's deck mean cost:{:<4}".format(D[0].get_mean_cost()))
    mylogger.info("Bob's deck mean cost:{:<4}".format(D[1].get_mean_cost()))
    D[0].mean_cost = D[0].get_mean_cost()
    D[1].mean_cost = D[1].get_mean_cost()
    assert len(D[0].deck) == 40 and len(D[1].deck) == 40,"deck_len:{},{}"\
        .format(len(D[0].deck),len(D[1].deck))
    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]
    win_lose = [win, lose]
    first_num = 0
    win_turns = [0,0]
    deck_name_list = {"Alice":{},"Bob":{}}
    deck_name_list["Alice"] = D[0].get_name_set()
    deck_name_list["Bob"] = D[1].get_name_set()
    for i in range(iteration):
        if not virtual_flg:
            mylogger.info("Game {}".format(i + 1))
        # mylogger.info("name:{}".format(Turn_Players[i%2].name))
        Turn_Players[i % 2].is_first = True
        Turn_Players[i % 2].player_num = 0
        Turn_Players[(i + 1) % 2].is_first = False
        Turn_Players[(i + 1) % 2].player_num = 1
        assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
            Turn_Players[0].player_num)
        (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first,(player1_win_turn,player2_win_turn)) \
            = demo_game_play(Turn_Players[i % 2],
           Turn_Players[(i + 1) % 2],
           D[i % 2], D[(i + 1) % 2],
           win_lose[i % 2],
           win_lose[(i + 1) % 2], lib_num,
           virtual_flg=virtual_flg,deck_name_list=deck_name_list)
        first_num += first
        sum_of_turn += end_turn
        if player1_win_turn is not False:
            win_turns[i%2]+=player1_win_turn
        elif player2_win_turn is not False:
            win_turns[(i+1)% 2] += player2_win_turn

        #mylogger.info("{}\n{}".format(deck_name_list[i % 2], deck_name_list[(i + 1) % 2]))
        if (i + 1) % span == 0:
            mylogger.info(
                "Halfway {}:win={}, lose={}, libout_num={}, win_rate:{:.3f}".format(i + 1, win_lose[0], win_lose[1],
                                                                                    lib_num, win_lose[0] / (i + 1)))
    mylogger.info("Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                 win_lose[0] / iteration))
    mylogger.info("deck size:{} mean_end_turn {:<3}".format(len(D[0].deck), sum_of_turn / iteration))
    mylogger.info("first_win_rate:{:<3}".format(first_num / iteration))
    if win_lose[0]==0:
        win_lose[0]=1
    if win_lose[1]==0:
        win_lose[1]=1
    mylogger.info("mean_win_turn:{:.3f},{:.3f}".format(win_turns[0]/win_lose[0],win_turns[1]/win_lose[1]))

    import itertools
    if Player1.mulligan_policy.data_use_flg == True:
        mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player1.mulligan_policy.mulligan_data,
                                                                            Player1.mulligan_policy.win_data)))))
    if Player2.mulligan_policy.data_use_flg == True:
        mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player2.mulligan_policy.mulligan_data,
                                                                            Player2.mulligan_policy.win_data)))))
    mylogger.info("deck_type:{}".format(deck_type))
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    contribution_list={"Alice":[],"Bob":[]}
    for i,player_key in enumerate(list(deck_name_list.keys())):
        mylogger.info("Player{}".format(i+1))
        for key in list(deck_name_list[player_key].keys()):
            target_cell = deck_name_list[player_key][key]
            if target_cell["used_num"] > 0:
                mylogger.info("{}'s contribution(used_num:{}):{:.3f}".format(key,target_cell["used_num"],target_cell["win_num"]/target_cell["used_num"]))
                contribution_list[player_key].append((key,target_cell["win_num"]/target_cell["used_num"]))

    contribution_list["Alice"].sort(key= lambda element:-element[1])
    contribution_list["Bob"].sort(key= lambda element:-element[1])
    for i,player_key in enumerate(list(contribution_list.keys())):
        mylogger.info("Player{}".format(i + 1))
        for j,cell in enumerate(contribution_list[player_key]):
            mylogger.info("No.{} {}:{:.3f}".format(j+1,cell[0],cell[1]))
        print("")

def get_contributions(Player_1, Player_2, iteration, virtual_flg=False, player1_deck_num=None,directory_name=None):
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
    D = [Deck()]*10
    assert player1_deck_num < len(D),"player1_deck_num:{}".format(player1_deck_num)
    deck_id_2_name = {0: "Aggro-Sword", 1: "Earth-RUne", 2: "Midrange-Sword", 3: "Midrange-Shadow", 4: "PDK-Dragon", 5: "Elana-Haven",
                      6: "Control-Blood", 7: "Ramp-Dragon", 8: "Forest", 9: "Spell-Rune"}

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
        current_decks = [D[player1_deck_num],D[deck_id]]
        win_lose = [0, 0]
        win_turns = [0, 0]
        first_num = 0
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
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first,(player1_win_turn,player2_win_turn)) \
                = demo_game_play(Turn_Players[i % 2],
               Turn_Players[(i + 1) % 2],
               current_decks[i % 2], current_decks[(i + 1) % 2],
               win_lose[i % 2],
               win_lose[(i + 1) % 2], lib_num,
               virtual_flg=True,deck_name_list=deck_name_list)
            first_num += first
            sum_of_turn += end_turn
            if player1_win_turn is not False:
                win_turns[i%2]+=player1_win_turn
            elif player2_win_turn is not False:
                win_turns[(i+1)% 2] += player2_win_turn
        result_txt = "Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                     win_lose[0] / iteration)
        mylogger.info(result_txt)
        if win_lose[0] == 0:
            win_lose[0] = 1
        if win_lose[1] == 0:
            win_lose[1] = 1

        #mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[player1_deck_num], Player_2.policy.name,
        #                                       deck_id_2_name[deck_id]))
        contribution_list={"Alice":[],"Bob":[]}
        for i,player_key in enumerate(list(deck_name_list.keys())):
            #mylogger.info("Player{}".format(i+1))
            for key in list(deck_name_list[player_key].keys()):
                target_cell = deck_name_list[player_key][key]
                if target_cell["used_num"] > 0:
                    #mylogger.info("{}'s contribution(used_num:{}):{:.3f}".format(key,target_cell["used_num"],target_cell["win_num"]/target_cell["used_num"]))
                    contribution_list[player_key].append((key,target_cell["win_num"]/target_cell["used_num"],target_cell["used_num"]))

        contribution_list["Alice"].sort(key= lambda element:-element[1])
        contribution_list["Bob"].sort(key= lambda element:-element[1])
        file_name=""
        title = "{}({})vs {}({})({} matchup)".format(Player_1.policy.name, deck_id_2_name[player1_deck_num],
                                                       Player_2.policy.name,
                                                       deck_id_2_name[deck_id], iteration)
        mylogger.info(title)
        title += "\n"
        if player1_deck_num!=deck_id:
            file_name = "contribution_{}_vs_{}.txt".format(deck_id_2_name[player1_deck_num],deck_id_2_name[deck_id])
        else:
            file_name = "contribution_{}_mirror.txt".format(deck_id_2_name[deck_id])
        with open(directory_name+"/"+file_name,mode="w") as f:

            f.write(title)
            f.write(result_txt+"\n")
            for i,player_key in enumerate(list(contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j,cell in enumerate(contribution_list[player_key]):
                    txt = "No.{} {}(used_num:{}):{:.3f}\n".format(j + 1, cell[0],cell[2],cell[1])
                    f.write(txt)
                    #mylogger.info("No.{} {}:{:.3f}".format(j+1,cell[0],cell[1]))
                f.write("\n")
        mylogger.info("{}/{} complete".format((deck_id+1),len(D)))

def get_basic_contributions(Player_1, Player_2, iteration, virtual_flg=False, player1_deck_num=None,directory_name=None):
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
    D = [Deck()]*8
    assert player1_deck_num < len(D),"player1_deck_num:{}".format(player1_deck_num)
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
        current_decks = [D[player1_deck_num],D[deck_id]]
        win_lose = [0, 0]
        win_turns = [0, 0]
        first_num = 0
        deck_name_list = {"Alice": current_decks[0].get_name_set(), "Bob": current_decks[1].get_name_set()}
        for i in range(iteration):
            Turn_Players[i % 2].is_first = True
            Turn_Players[i % 2].player_num = 0
            Turn_Players[(i + 1) % 2].is_first = False
            Turn_Players[(i + 1) % 2].player_num = 1
            assert Turn_Players[0].player_num != Turn_Players[1].player_num, "same error {}".format(
                Turn_Players[0].player_num)
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first,(player1_win_turn,player2_win_turn)) \
                = demo_game_play(Turn_Players[i % 2],
               Turn_Players[(i + 1) % 2],
               current_decks[i % 2], current_decks[(i + 1) % 2],
               win_lose[i % 2],
               win_lose[(i + 1) % 2], lib_num,
               virtual_flg=True,deck_name_list=deck_name_list)
            first_num += first
            sum_of_turn += end_turn
            if player1_win_turn is not False:
                win_turns[i%2]+=player1_win_turn
            elif player2_win_turn is not False:
                win_turns[(i+1)% 2] += player2_win_turn
        result_txt = "Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                     win_lose[0] / iteration)
        mylogger.info(result_txt)
        result_txt += "\n"
        if win_lose[0] == 0:
            win_lose[0] = 1
        if win_lose[1] == 0:
            win_lose[1] = 1

        #mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[player1_deck_num], Player_2.policy.name,
        #                                       deck_id_2_name[deck_id]))
        contribution_list={"Alice":[],"Bob":[]}
        for i,player_key in enumerate(list(deck_name_list.keys())):
            #mylogger.info("Player{}".format(i+1))
            for key in list(deck_name_list[player_key].keys()):
                target_cell = deck_name_list[player_key][key]
                if target_cell["used_num"] > 0:
                    #mylogger.info("{}'s contribution(used_num:{}):{:.3f}".format(key,target_cell["used_num"],target_cell["win_num"]/target_cell["used_num"]))
                    contribution_list[player_key].append((key,target_cell["win_num"]/target_cell["used_num"],target_cell["used_num"]))

        contribution_list["Alice"].sort(key= lambda element:-element[1])
        contribution_list["Bob"].sort(key= lambda element:-element[1])
        file_name=""
        title = "{}({})vs {}({})({} matchup)".format(Player_1.policy.name, deck_id_2_name[player1_deck_num],
                                                       Player_2.policy.name,
                                                       deck_id_2_name[deck_id], iteration)
        mylogger.info(title)
        title += "\n"
        if player1_deck_num!=deck_id:
            file_name = "contribution_{}_vs_{}.txt".format(deck_id_2_name[player1_deck_num],deck_id_2_name[deck_id])
        else:
            file_name = "contribution_{}_mirror.txt".format(deck_id_2_name[deck_id])
        with open(directory_name+"/"+file_name,mode="w") as f:

            f.write(title)
            f.write(result_txt)
            for i,player_key in enumerate(list(contribution_list.keys())):
                f.write("Player{}\n".format(i + 1))
                for j,cell in enumerate(contribution_list[player_key]):
                    txt = "No.{} {}(used_num:{}):{:.3f}\n".format(j + 1, cell[0],cell[2],cell[1])
                    f.write(txt)
                    #mylogger.info("No.{} {}:{:.3f}".format(j+1,cell[0],cell[1]))
                f.write("\n")
        mylogger.info("{}/{} complete".format((deck_id+1),len(D)))

def test_2(Player_1, Player_2, iteration, same_flg=False, result_name="Result.tsv"):
    mylogger.info("{} vs {}".format(Player_1.policy.name, Player_2.policy.name))
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

    for i, d in enumerate(D):
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

    # D[0].mean_cost=D[0].get_mean_cost()
    # D[1].mean_cost=D[1].get_mean_cost()
    assert all(len(D[i].deck) == 40 for i in range(8))
    # Turn_Players=[Player1,Player2]
    Results = {}
    mylogger.info("same_flg:{}".format(same_flg))
    for j in range(len(D)):
        l = 0
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
                (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first,_) = game_play(Turn_Players[i % 2],
                                                                                               Turn_Players[
                                                                                                   (i + 1) % 2], D[j],
                                                                                               D[k], \
                                                                                               win_lose[i % 2],
                                                                                               win_lose[(i + 1) % 2],
                                                                                               lib_num,
                                                                                               virtual_flg=True)
                first_num += first
            Results[(j, k)] = [win_lose[0] / iteration, first_num / iteration]
        mylogger.info("complete:{}/{}".format(j + 1, len(D)))
    # for key in list(Results.keys()):
    #    mylogger.info("({}):rate:{} first:{}".format(key,Results[key][0],Results[key][1]))
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune"}
    with open("Battle_Result/" + result_name, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        row = ["RULE\\MCTS"]
        mylogger.info("row:{}".format(row))
        mylogger.info("row:{}".format([deck_id_2_name[i] for i in range(9)]))
        row = row + [deck_id_2_name[i] for i in range(9)]
        mylogger.info("row:{}".format(row))
        writer.writerow(row)
        if same_flg:
            for i in range(9):
                row = [deck_id_2_name[i]]
                for j in range(0, i + 1):
                    row.append(Results[(i, j)][0])
                mylogger.info(row)
                writer.writerow(row)
        else:
            for i in range(9):
                row = [deck_id_2_name[i]]
                for j in range(9):
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
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first,_) = game_play(Turn_Players[i % 2],
                                                                                           Turn_Players[(i + 1) % 2],
                                                                                           D[j], D[k], \
                                                                                           win_lose[i % 2],
                                                                                           win_lose[(i + 1) % 2],
                                                                                           lib_num, virtual_flg=False)
            first_num += first


def make_policy_table(n, initial_players=None, deck_type=None, same_flg=False, result_name="Policy_table_result.tsv"):
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune"}
    mylogger.info("{} vs {}".format(deck_id_2_name[deck_type[0]], deck_id_2_name[deck_type[1]]))
    policy_id_2_name={}
    for i,target_player in enumerate(initial_players):
        policy_id_2_name[i]=target_player.policy.name
    mylogger.info("players:{}".format(policy_id_2_name))
    iteration = n
    win = 0
    lose = 0
    lib_num = 0
    assert initial_players != None, "Non-players!"
    assert deck_type != None, "Non-Deck_type!"
    players = copy.deepcopy(initial_players)
    D = [Deck() for i in range(2)]
    for i, d in enumerate(D):
        if deck_type[i] == 0:
            D[i] = tsv_to_deck("Sword_Aggro.tsv")
            # Aggro
        elif deck_type[i] == 1:
            D[i] = tsv_to_deck("Rune_Earth.tsv")
            # Aggro
        elif deck_type[i] == 2:
            D[i] = tsv_to_deck("Sword.tsv")
            # Mid
        elif deck_type[i] == 3:
            # D[i]=tsv_to_deck("Shadow.tsv")
            D[i] = tsv_to_deck("New-Shadow.tsv")
            # Mid
        elif deck_type[i] == 4:
            D[i] = tsv_to_deck("Dragon_PDK.tsv")
            # Mid
        elif deck_type[i] == 5:
            D[i] = tsv_to_deck("Haven.tsv")
            # Control
        elif deck_type[i] == 6:
            D[i] = tsv_to_deck("Blood.tsv")
            # Control
        elif deck_type[i] == 7:
            D[i] = tsv_to_deck("Dragon.tsv")
            # Control
        elif deck_type[i] == 8:
            D[i] = tsv_to_deck("Forest.tsv")
            # Combo
        elif deck_type[i] == 9:
            D[i] = tsv_to_deck("Rune.tsv")
            # Combo
    Results = {}
    for policy1_id, player1 in enumerate(players):
        P1 = copy.deepcopy(player1)
        last_id = len(players)
        for policy2_id in range(0,last_id):
            player2=players[policy2_id]
            P2 = copy.deepcopy(player2)
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
                (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first,_) = game_play(Turn_Players[i % 2],
                                                                                               Turn_Players[
                                                                                                   (i + 1) % 2],
                                                                                               D[i % 2], D[(i + 1) % 2],
                                                                                               win_lose[i % 2],
                                                                                               win_lose[(i + 1) % 2],
                                                                                               lib_num,
                                                                                               virtual_flg=True)
                first_num += first
            Results[(policy1_id, policy2_id)] = [win_lose[0] / iteration, first_num / iteration]
        mylogger.info("complete:{}/{}".format(policy1_id + 1, len(players)))
    deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune"}
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
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.1,playout=AggroPolicy()), mulligan=Min_cost_mulligan_policy()))  # 32
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.25,playout=AggroPolicy()), mulligan=Min_cost_mulligan_policy()))  # 33
Players.append(Player(9, True, policy=Time_bounded_MCTSPolicy(limit=0.5,playout=AggroPolicy()), mulligan=Min_cost_mulligan_policy()))  # 34
Players.append(Player(9, True, policy=Aggro_Shallow_MCTSPolicy(th=3), mulligan=Min_cost_mulligan_policy()))  # 35
Players.append(Player(9, True, policy=Expanded_Aggro_MCTS_Policy(), mulligan=Min_cost_mulligan_policy()))  # 36

parser = argparse.ArgumentParser(description='対戦実行コード')

parser.add_argument('--N', help='試行回数')
parser.add_argument('--playertype1', help='プレイヤー1のAIタイプ')
parser.add_argument('--playertype2', help='プレイヤー2のAIタイプ')
parser.add_argument('--decktype1', help='プレイヤー1のデッキタイプ')
parser.add_argument('--decktype2', help='プレイヤー2のデッキタイプ')
parser.add_argument('--filename', help='ファイル名')
parser.add_argument('--playerlist', help='対戦AIタイプリスト')
parser.add_argument('--mode', help='実行モード、demoで対戦画面表示,policyでdecktype固定で各AIタイプの組み合わせで対戦')
args = parser.parse_args()
mylogger.info("args:{}".format(args))
# assert False
n = 100
a = 0
b = 0
v = False
deck_flg = False
p1, p2 = 0, 0
human_player = HumanPlayer(9, first=True)
file_name = "Result.tsv"

import cProfile
import re
import datetime

iteration = n
d1 = None
d2 = None

input_players = [Players[0], Players[1], Players[4],Players[8], Players[9], Players[11], Players[13], Players[14]]
if args.playerlist is not None:
    input_players=[]
    tmp = list(args.playerlist.split(","))
    player_id_list = [int(ele) for ele in tmp]
    player_id_list = sorted(list(set(player_id_list)))
    for player_id in player_id_list:
        assert player_id >= 0 and player_id <= len(Players)
        input_players.append(Players[player_id-1])

t1 = datetime.datetime.now()
# if sys.argv[-1]=="-demo":
if args.mode == 'demo':
    # cProfile.run('execute_demo(d1,d2,n,deck_type=[p1,p2])')
    n = int(args.N)
    a = int(args.playertype1) - 1
    b = int(args.playertype2) - 1
    if args.playertype1=='0':
        d1 = copy.deepcopy(human_player)
    else:
        d1 = copy.deepcopy(Players[a])
    if args.playertype2=='0':
        d2 = copy.deepcopy(human_player)
    else:
        d2 = copy.deepcopy(Players[b])
    p1 = int(args.decktype1)
    p2 = int(args.decktype2)
    execute_demo(d1, d2, n, deck_type=[p1, p2])
# elif sys.argv[-1]=="-shadow":
elif args.mode == 'shadow':
    test_3(d1, d2, n)
# elif sys.argv[-1]=="-policy":
elif args.mode == 'policy':
    # file_name=sys.argv[-2]
    n = int(args.N)
    if args.filename != None:
        file_name = args.filename
    assert args.decktype1 != None and args.decktype2 != None, "deck1:{},deck2:{}".format(args.decktype1, args.decktype2)
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
    if args.filename != None:
        file_name = args.filename
    get_contributions(d1, d2, iteration, virtual_flg=True, player1_deck_num = player1_deck_id, directory_name=file_name)
elif args.mode == 'basic':
    iteration = int(args.N)
    a = int(args.playertype1) - 1
    b = int(args.playertype2) - 1
    player1_deck_id = int(args.decktype1)
    d1 = copy.deepcopy(Players[a])
    d2 = copy.deepcopy(Players[b])
    if args.filename != None:
        file_name = args.filename
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
    path = "{}_vs_{}_in_basic_{}times".format(short_name_1,short_name_2,iteration)
    os.makedirs(path)
    for i in range(8):
        player1_deck_id = i
        next_path = "/{}_vs_{}_in_basic_{}times_contributions_{}".format(short_name_1,short_name_2,
                                                                   iteration,deck_id_2_name[i])
        os.makedirs(path+next_path)
        file_name = path+next_path
        get_basic_contributions(d1, d2, iteration, virtual_flg=True, player1_deck_num=player1_deck_id,
                                directory_name=file_name)


else:
    iteration = int(args.N)
    a = int(args.playertype1) - 1
    b = int(args.playertype2) - 1
    d1 = copy.deepcopy(Players[a])
    d2 = copy.deepcopy(Players[b])
    if args.filename != None:
        file_name = args.filename
    if a == b:
        test_2(d1, d2, iteration, same_flg=True, result_name=file_name)
    else:
        test_2(d1, d2, iteration, result_name=file_name)
mylogger.info(t1)
t2 = datetime.datetime.now()
mylogger.info(t2)
mylogger.info(t2 - t1)

