import logging
from my_moduler import get_module_logger
import util_ability
from my_enum import *
mylogger = get_module_logger(__name__)


class Game:

    def mulligan(self,Player1,Player2,virtual=False):
        assert Player1.player_num!=Player2.player_num,"same error"
        Player1.mulligan(Player1.deck,virtual=virtual)
        if virtual==False:
            print("")
        Player2.mulligan(Player2.deck,virtual=virtual)



    def start(self,f,virtual_flg=False):
        turn=1
        win,lose,lib_num=0,0,0
        self.mulligan(f.players[0],f.players[1],virtual=virtual_flg)
        while(True):
            end_flg=False

            #(win,lose,lib_num,turn,end_flg)=self.play_turn(f,0,win,lose,lib_num,turn,virtual_flg)
            (win,lose,lib_num,turn,end_flg)=f.play_turn(0,win,lose,lib_num,turn,virtual_flg)
            if end_flg==True:
                break
            (win,lose,lib_num,turn,end_flg)=f.play_turn(1,win,lose,lib_num,turn,virtual_flg)
            #(win,lose,lib_num,turn,end_flg)=self.play_turn(f,1,win,lose,lib_num,turn,virtual_flg)
            if end_flg==True:
                break
        if f.players[0].mulligan_policy.data_use_flg==True: 
            f.players[0].mulligan_policy.append_win_data(bool(win))
        if f.players[1].mulligan_policy.data_use_flg==True: 
            f.players[1].mulligan_policy.append_win_data(bool(lose))
        return win,lose,lib_num,turn

    """
    def play_turn(self,f,turn_player_num,win,lose,lib_num,turn,virtual_flg):
            while(True):
                can_play=True
                can_attack=True
                f.untap(turn_player_num)
                f.increment_cost(turn_player_num)
                if virtual_flg==False:
                    mylogger.info("Turn {}".format(turn))
                    mylogger.info("Player{} turn start cost:{}".format(turn_player_num+1,f.cost[turn_player_num]))
                f.start_of_turn(turn_player_num,virtual=virtual_flg)
                if f.check_game_end()==True:
                    if f.players[turn_player_num].life<=0 or f.players[turn_player_num].deck.deck==[]:
                        if turn_player_num==0:
                            lose+=1
                        else:
                            win+=1
                    elif f.players[1-turn_player_num].life<=0 or f.players[1-turn_player_num].deck.deck==[]:
                        if turn_player_num==0:
                            win+=1
                        else:
                            lose+=1
                    break

                if turn_player_num==1 and f.current_turn[turn_player_num]==1:
                    util_ability.draw_cards(f.players[turn_player_num],virtual_flg,num=1)
                util_ability.draw_cards(f.players[turn_player_num],virtual_flg,num=1)
                if f.check_game_end():
                    if turn_player_num==0:
                        lose+=1
                    else:
                        win+=1
                    lib_num+=1
                    return win,lose,lib_num,turn,True
                while True:
                    end_flg=f.players[turn_player_num].decide(\
                        f.players[turn_player_num],f.players[1-turn_player_num],f,virtual=virtual_flg)
                    if end_flg==True :
                            break
                if virtual_flg==False:
                    f.players[turn_player_num].show_hand()
                    mylogger.info("Player1 life:{} Player2 life:{} remain_cost:{}".format(f.players[0].life,f.players[1].life,\
                        f.remain_cost[turn_player_num]))
                    f.show_field()

                            
                if f.check_game_end():
                    if f.players[turn_player_num].life<=0 or f.players[turn_player_num].lib_out_flg==True:
                        if turn_player_num==0:
                            lose+=1
                        else:
                            win+=1
                    elif f.players[1-turn_player_num].life<=0 or f.players[1-turn_player_num].lib_out_flg==True:
                        if turn_player_num==0:
                            win+=1
                        else:
                            lose+=1
                    break

                f.end_of_turn(turn_player_num,virtual=virtual_flg)

                if f.check_game_end():
                    if f.players[turn_player_num].life<=0 or f.players[turn_player_num].lib_out_flg==True:
                        if turn_player_num==0:
                            lose+=1
                        else:
                            win+=1
                    elif f.players[1-turn_player_num].life<=0 or f.players[1-turn_player_num].lib_out_flg==True:
                        if turn_player_num==0:
                            win+=1
                        else:
                            lose+=1
                    break
                if virtual_flg==False:
                    mylogger.info("Player{} turn end".format(turn_player_num+1))
                turn+=1
                if f.ex_turn_count[turn_player_num]>0:
                    if virtual_flg==False:
                        mylogger.info("remain_turn:{}".format(f.ex_turn_count[turn_player_num]))
                    f.ex_turn_count[turn_player_num]-=1
                else:
                    break
            
            return win,lose,lib_num,turn,f.check_game_end()
    """