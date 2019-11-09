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


def game_play(Player1, Player2, D1, D2, win, lose, lib_num, virtual_flg=False):
    assert Player1.player_num != Player2.player_num, "same error"
    f = Field(5)
    f.players[0] = Player1
    f.players[0].field = f
    f.players[1] = Player2
    f.players[1].field = f
    f.players[0].deck = Deck()
    for card in D1.deck:
        f.players[0].deck.deck.append(card.get_copy())
    f.players[1].deck = Deck()
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
    return win, lose, lib_num, turn, first


# import numba
# @numba.jit
def test_1(Player_1, Player_2, iteration, virtual_flg=False, deck_type=None):
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
                      6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune"}
    mylogger.info("{}({})vs {}({})".format(Player_1.policy.name, deck_id_2_name[deck_type[0]], Player_2.policy.name,
                                           deck_id_2_name[deck_type[1]]))
    class_pool = [0, 0]
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
        elif deck_type[i] == 10:
            D[i] = tsv_to_deck("New-Shadow.tsv")
            # Combo
        elif deck_type[i] == 11:
            class_pool[i] = 2
            d.set_deck_type(2)
            # テスト用デッキ
            d.append(Creature(creature_name_to_id["Goblin"]), num=3)
            d.append(Creature(creature_name_to_id["Fighter"]), num=3)
            d.append(Creature(creature_name_to_id["Unicorn Dancer Unica"]), num=3)
            d.append(Spell(spell_name_to_id["Seraphic Blade"]), num=3)
            d.append(Creature(creature_name_to_id["Ax Fighter"]), num=3)
            d.append(Creature(creature_name_to_id["Golyat"]), num=3)
            d.append(Creature(creature_name_to_id["Gilgamesh"]), num=3)

    Player1.class_num = class_pool[0]
    Player2.class_num = class_pool[1]
    mylogger.info("Alice's deck mean cost:{:<4}".format(D[0].get_mean_cost()))
    mylogger.info("Bob's deck mean cost:{:<4}".format(D[1].get_mean_cost()))
    D[0].mean_cost = D[0].get_mean_cost()
    D[1].mean_cost = D[1].get_mean_cost()
    assert len(D[0].deck) == 40 and len(D[1].deck) == 40
    """
        mylogger.info("Player1_Deck_Cost_Rate")
        D[0].get_cost_histgram()
        mylogger.info("Player2_Deck_Cost_Rate")
        D[1].get_cost_histgram()
        """
    sum_of_turn = 0
    span = max(iteration // 10, 1)
    Turn_Players = [Player1, Player2]
    win_lose = [win, lose]
    first_num = 0
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
        (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first) = game_play(Turn_Players[i % 2],
                                                                                       Turn_Players[(i + 1) % 2],
                                                                                       D[i % 2], D[(i + 1) % 2],
                                                                                       win_lose[i % 2],
                                                                                       win_lose[(i + 1) % 2], lib_num,
                                                                                       virtual_flg=virtual_flg)
        first_num += first
        sum_of_turn += end_turn
        if (i + 1) % span == 0:
            mylogger.info(
                "Halfway {}:win={}, lose={}, libout_num={}, win_rate:{:.3f}".format(i + 1, win_lose[0], win_lose[1],
                                                                                    lib_num, win_lose[0] / (i + 1)))
    mylogger.info("Result:win={}, lose={}, libout_num={}, win_rate:{:<3}".format(win_lose[0], win_lose[1], lib_num,
                                                                                 win_lose[0] / iteration))
    mylogger.info("deck size:{} mean_end_turn {:<3}".format(len(D[0].deck), sum_of_turn / iteration))
    mylogger.info("first_win_rate:{:<3}".format(first_num / iteration))

    import itertools
    if Player1.mulligan_policy.data_use_flg == True:
        mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player1.mulligan_policy.mulligan_data,
                                                                            Player1.mulligan_policy.win_data)))))
    if Player2.mulligan_policy.data_use_flg == True:
        mylogger.info("mulligan_data:{}".format(set(list(itertools.compress(Player2.mulligan_policy.mulligan_data,
                                                                            Player2.mulligan_policy.win_data)))))


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
                (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first) = game_play(Turn_Players[i % 2],
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
            (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first) = game_play(Turn_Players[i % 2],
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
        last_id=len(players)
        if same_flg:
            last_id=policy1_id+1
        #for policy2_id, player2 in enumerate(players):
        #for policy2_id in range(init_id,len(players)):for policy2_id in range(init_id,len(players)):
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
                (win_lose[i % 2], win_lose[(i + 1) % 2], lib_num, end_turn, first) = game_play(Turn_Players[i % 2],
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
    #policy_id_2_name = {0: "Random", 1: "Aggro", 2: "Greedy", 3: "MCTS", 4: "Test2-MCTS", 5: "A-MCTS", 6: "EXP3_MCTS",
    #                    7: "New-MCTS", 8: "New-A-MCTS"}
    mylogger.info("keys:{}".format(list(Results.keys())))
    with open("Battle_Result/" + result_name, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        row = ["{} vs {}".format(deck_id_2_name[deck_type[0]], deck_id_2_name[deck_type[1]])]
        mylogger.info("row:{}".format(row))
        mylogger.info("row:{}".format([policy_id_2_name[i] for i in range(len(policy_id_2_name))]))
        row = row + [policy_id_2_name[i] for i in range(len(policy_id_2_name))]
        mylogger.info("row:{}".format(row))
        writer.writerow(row)
        if same_flg:
            for i in range(len(policy_id_2_name)):
                row = [policy_id_2_name[i]]
                for j in range(0, i + 1):
                    row.append(Results[(i,j)][0])
                for j in range(i+1,len(policy_id_2_name)):
                    assert (j,i) in Results,"Null Result!,({},{})".format(j,i)
                    row.append(1-Results[(j,i)][0])
                mylogger.info(row)
                writer.writerow(row)
        else:
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

# Player(5,True,policy=AggroPolicy(),mulligan=Min_cost_mulligan_policy)
# Player(5,False,policy=GreedyPolicy(),mulligan=Min_cost_mulligan_policy)
parser = argparse.ArgumentParser(description='対戦実行コード')

parser.add_argument('--N', help='試行回数')
parser.add_argument('--playertype1', help='プレイヤー1のAIタイプ')
parser.add_argument('--playertype2', help='プレイヤー2のAIタイプ')
parser.add_argument('--decktype1', help='プレイヤー1のデッキタイプ')
parser.add_argument('--decktype2', help='プレイヤー2のデッキタイプ')
parser.add_argument('--filename', help='ファイル名')
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
"""
if len(sys.argv)>=2:
    n=int(sys.argv[1])
if len(sys.argv)>=4:
    a=int(sys.argv[2])-1
    b=int(sys.argv[3])-1
if len(sys.argv)>=5:
    file_name=sys.argv[4]
    v=sys.argv[-1]=="-v"
if len(sys.argv)>=6 and sys.argv[-1]!="-policy":
    mylogger.info("Deck")
    p1=int(sys.argv[4])
    p2=int(sys.argv[5])
    deck_flg=True
"""
# raise Exception("Debug {} {}".format(sys.argv,len(sys.argv)))

import cProfile
import re
import datetime

iteration = n
d1 = None
d2 = None
"""
if a==-1:
    d1=human_player
else:
    d1=copy.deepcopy(Players[a])
if b==-1:
    d2=human_player
else:
    d2=copy.deepcopy(Players[b])
"""

input_players = [Players[0], Players[1], Players[4],Players[8], Players[11], Players[13], Players[14], Players[15],
                 Players[16]]
t1 = datetime.datetime.now()
# if sys.argv[-1]=="-demo":
if args.mode == 'demo':
    # cProfile.run('test_1(d1,d2,n,deck_type=[p1,p2])')
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
    test_1(d1, d2, n, deck_type=[p1, p2])
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
"""
import time
t1=time.gmtime()
d1=None
d2=None
if a==-1:
    d1=human_player
else:
    d1=copy.deepcopy(Players[a])
if b==-1:
    d2=human_player
else:
    d2=copy.deepcopy(Players[b])
if v == True:
    if deck_flg==True:
        #cProfile.run('test_1(d1,d2,n,virtual_flg=True,deck_type=[p1,p2])')
        test_1(d1,d2,n,virtual_flg=True,deck_type=[p1,p2])
    else:
        test_1(d1,d2,n,virtual_flg=True)
else:
    if deck_flg==True:
        test_1(d1,d2,n,deck_type=[p1,p2])
    else:
        test_1(d1,d2,n)
mylogger.info(t1)
t2=time.gmtime()
mylogger.info(t2)
"""
