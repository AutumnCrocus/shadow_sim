import logging
from my_moduler import get_module_logger
import util_ability
from my_enum import *
import numpy as np
#import matplotlib.pyplot as plt

mylogger = get_module_logger(__name__)


class Game:

    def mulligan(self, Player1, Player2, virtual=False):
        assert Player1.player_num != Player2.player_num, "same error"
        Player1.mulligan(Player1.deck, virtual=virtual)
        if not virtual:
            print("")
        Player2.mulligan(Player2.deck, virtual=virtual)
        assert len(Player1.hand) == 3 and len(Player2.hand) == 3,"{},{}".format(len(Player1.hand),len(Player2.hand))

    def start(self, f, virtual_flg=False,history_flg = False):
        turn = 1
        win, lose, lib_num = 0, 0, 0
        f.secret = bool(virtual_flg)
        self.mulligan(f.players[0], f.players[1], virtual=virtual_flg)
        while (True):
            end_flg = False

            # (win,lose,lib_num,turn,end_flg)=self.play_turn(f,0,win,lose,lib_num,turn,virtual_flg)
            (win, lose, lib_num, turn, end_flg) = f.play_turn(0, win, lose, lib_num, turn, virtual_flg)
            if end_flg:
                break
            (win, lose, lib_num, turn, end_flg) = f.play_turn(1, win, lose, lib_num, turn, virtual_flg)
            # (win,lose,lib_num,turn,end_flg)=self.play_turn(f,1,win,lose,lib_num,turn,virtual_flg)
            if end_flg:
                break
        if history_flg:

            #mylogger.info("state_value_histroy")
            #for i,value in enumerate(f.state_value_history):
            #    mylogger.info("Turn {}:{:.3f}".format(i+1,value))

            turns = np.array(range(len(f.state_value_history)))
            state_values = f.state_value_history
            #if f.players[0].name == "Bob":
            #    state_values = [1 - value for value in state_values]
            state_values = np.array(state_values)
            return win, lose, lib_num, turn, [turns,state_values]

        #if f.players[0].mulligan_policy.data_use_flg:
        #    f.players[0].mulligan_policy.append_win_data(bool(win))
        #if f.players[1].mulligan_policy.data_use_flg:
        #    f.players[1].mulligan_policy.append_win_data(bool(lose))

        return win, lose, lib_num, turn

    def start_for_train_data(self, f, virtual_flg=False,target_player_num=0):
        turn = 1
        win, lose, lib_num = 0, 0, 0
        f.secret = bool(virtual_flg)
        self.mulligan(f.players[0], f.players[1], virtual=virtual_flg)
        train_datas = []
        reward = 0.0
        accumulate_turn = 0
        while True:
            (win, lose, end_flg, train_data) = f.play_turn_for_train(0)
            if target_player_num == 0:
                train_datas.extend(train_data)
            if end_flg:
                break
            (win, lose, end_flg, train_data) = f.play_turn_for_train(1)
            if target_player_num == 1:
                train_datas.extend(train_data)
            if end_flg:
                break
            accumulate_turn += 1
            assert accumulate_turn < 100,"infinite loop"
        #reward = int(target_player_num == 0)*(2*win-1) + 1 - win
        reward = win if target_player_num == 0 else lose
        return train_datas, reward

    def start_for_dual(self, f, virtual_flg=False,target_player_num=0):
        turn = 1
        win, lose, lib_num = 0, 0, 0
        f.secret = bool(virtual_flg)
        f.players[0].draw(f.players[0].deck, 3)
        f.players[1].draw(f.players[1].deck, 3)
        self.mulligan(f.players[0], f.players[1], virtual=virtual_flg)
        train_datas = [[],[]]
        #reward = 0.0
        accumulate_turn = 0
        while True:
            (win, lose, end_flg, train_data) = f.play_turn_for_dual(0)
            train_datas[0].extend(train_data)
            #if target_player_num == 0:
            #    train_datas.extend(train_data)
            if end_flg:
                break
            (win, lose, end_flg, train_data) = f.play_turn_for_dual(1)
            train_datas[1].extend(train_data)
            #if target_player_num == 1:
            #    train_datas.extend(train_data)
            if end_flg:
                break
            accumulate_turn += 1
            assert accumulate_turn < 100,"infinite loop\n{}".format(f.get_observable_data(player_num=target_player_num))
            #print("accumulate_turn:{}".format(accumulate_turn))
        #reward = win if target_player_num == 0 else lose
        #reward = 2*reward - 1.0
        reward = [2*win-1,2*lose-1]
        return train_datas, reward


from collections import namedtuple
Detail_State_data = namedtuple('Value', ('hand_card_categories_and_ids', 'hand_card_costs', 'follower_card_ids',
                                         'amulet_card_ids', 'follower_stats', 'follower_abilities', 'follower_is_evolved',
                                         'life_data'))
import Embedd_Network_model
def get_data(field, player_num=0):
    return Embedd_Network_model.get_data(field,player_num=player_num)
"""
    hand_card_categories_and_ids = []
    hand_card_costs = []

    for hand_card in f.players[0].hand:
        hand_card_categories_and_ids.append(Card_Category[hand_card.card_category].value * (hand_card.card_id + 500))
        hand_card_costs.append(hand_card.cost)

    for j in range(len(f.players[0].hand), 9):
        hand_card_categories_and_ids.append(0)
        hand_card_costs.append(0)
    follower_card_ids = []
    amulet_card_ids = []
    follower_stats = []
    follower_abilities = []
    follower_is_evolved = f.get_able_to_evo(0)
    for i in range(2):
        for card in f.card_location[i]:
            if card.card_category == "Creature":
                follower_card_ids.append(card.card_id + 500)
                follower_stats.extend([card.power, card.get_current_toughness()])
                follower_abilities.append(card.ability[:])
                amulet_card_ids.append(0)
            else:
                follower_card_ids.append(0)
                follower_stats.extend([0, 0])
                follower_abilities.append([])
                amulet_card_ids.append(card.card_id + 500)

        for k in range(len(f.card_location[i]), 5):
            follower_card_ids.append(0)
            follower_stats.extend([0, 0])
            follower_abilities.append([])
            amulet_card_ids.append(0)
    life_data = [f.players[0].life, f.players[1].life, f.current_turn[0]]
    datas = Detail_State_data(hand_card_categories_and_ids, hand_card_costs, follower_card_ids, amulet_card_ids,
                              follower_stats, follower_abilities, follower_is_evolved, life_data)

    return datas

    input_field_data = []
    for hand_card in f.players[player_num].hand:
        input_field_data.append(Card_Category[hand_card.card_category].value)
        input_field_data.append(hand_card.cost)
        input_field_data.append(hand_card.card_id)

    input_field_data.extend([0,0,0]*(9-len(f.players[player_num].hand)))
    for side_num in range(2):
        i = (side_num + player_num) % 2
        for card in f.card_location[i]:
            if card.card_category == "Creature":
                input_field_data.append(card.card_id)
                input_field_data.extend([card.power, card.get_current_toughness()])
                #embed_ability = [int(ability_id in card.ability) for ability_id in range(1, 16)]
                input_field_data.append(card.ability[:])
            # input_field_data.extend([card.card_id, card.power, card.get_current_toughness(),
            #                         int(KeywordAbility.WARD.value in card.ability)])
            else:
                input_field_data.append(0)
                input_field_data.extend([0, 0])
                input_field_data.append([])

        for k in range(len(f.card_location[i]), 5):
            input_field_data.append(0)
            input_field_data.extend([0, 0])
            input_field_data.append([])
    input_field_data.extend([f.players[player_num].life, f.players[1-player_num].life, f.current_turn[player_num]])
    #print(input_field_data)
    return input_field_data
"""
