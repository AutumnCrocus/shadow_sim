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

        if f.players[0].mulligan_policy.data_use_flg:
            f.players[0].mulligan_policy.append_win_data(bool(win))
        if f.players[1].mulligan_policy.data_use_flg:
            f.players[1].mulligan_policy.append_win_data(bool(lose))

        return win, lose, lib_num, turn


