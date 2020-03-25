from my_enum import *
import itertools
import copy
from my_moduler import get_module_logger
mylogger = get_module_logger(__name__)
def adjust_action_code(field,sim_field,player_num,action_code = None, msg=None):
    player = field.players[player_num]
    sim_player = sim_field.players[player.player_num]
    action_num, card_id, target_id = action_code
    prev_card_id = int(card_id)
    prev_target_id = copy.copy(target_id)
    opponent_num = 1 - player_num
    if msg == Action_Code.PLAY_CARD.value:
        playing_card = sim_player.hand[card_id]
        error_flg = True
        for i, real_card in enumerate(player.hand):
            if real_card.eq(playing_card):
                card_id = i
                error_flg = False
                break

    elif msg == Action_Code.ATTACK_TO_FOLLOWER.value:
        attacking_card = sim_field.card_location[player.player_num][card_id]
        attacked_card = sim_field.card_location[opponent_num][target_id]
        error_flg = True
        for i, real_card in enumerate(field.card_location[player.player_num]):
            if real_card.eq(attacking_card):
                card_id = i
                error_flg = False
                break
        error_flg = True
        for i, real_card in enumerate(field.card_location[opponent_num]):
            if real_card.eq(attacked_card):
                card_id = i
                error_flg = False
                break
    elif msg == Action_Code.ATTACK_TO_PLAYER.value:
        attacking_card = sim_field.card_location[player.player_num][card_id]
        error_flg = True
        for i, real_card in enumerate(field.card_location[player.player_num]):
            if real_card.eq(attacking_card):
                card_id = i
                error_flg = False
                break
    elif msg == Action_Code.EVOLVE.value:
        evolving_card = sim_field.card_location[player.player_num][card_id]
        error_flg = True
        for i, real_card in enumerate(field.card_location[player.player_num]):
            if real_card.eq(evolving_card):
                card_id = i
                error_flg = False
                break
    else:
        assert False, "msg:{}".format(msg)

    if error_flg:
        if msg == Action_Code.PLAY_CARD.value:
            mylogger.info("sim_card_id:{}".format(card_id))
            mylogger.info("real")
            player.show_hand()
            mylogger.info("sim")
            sim_player.show_hand()
        elif msg == Action_Code.ATTACK_TO_FOLLOWER.value or msg == Action_Code.ATTACK_TO_PLAYER.value \
                or msg == Action_Code.EVOLVE.value:
            mylogger.info("sim_attacker_id:{}, sim_attacked_id:{}".format(card_id,target_id))
            mylogger.info("real")
            field.show_field()
            mylogger.info("sim")
            sim_field.show_field()
        assert False,"{}".format(Action_Code(msg).name)

    if target_id is not None and \
            not (action_num == Action_Code.ATTACK_TO_PLAYER.value or action_num == Action_Code.ATTACK_TO_FOLLOWER.value):
        target_type = None
        if action_num == Action_Code.PLAY_CARD.value:
            target_type = sim_player.hand[prev_card_id].current_target
        if action_num == Action_Code.EVOLVE.value:
            target_type = sim_field.card_location[player_num][prev_card_id].current_target
        if target_type is not None:
            if target_type == Target_Type.ENEMY_FOLLOWER.value or target_type == Target_Type.ENEMY_CARD.value:
                targeted_card = sim_field.card_location[opponent_num][prev_target_id]
                error_flg = True
                for i, real_card in enumerate(field.card_location[opponent_num]):
                    if real_card.eq(targeted_card):
                        target_id = i
                        return action_num, card_id, target_id

                mylogger.info("sim_target_id:{}".format(target_id))
                mylogger.info("real")
                field.show_field()
                mylogger.info("sim")
                sim_field.show_field()
                assert False,"error:{}".format(Target_Type(target_type).name)

            elif target_type == Target_Type.ALLIED_FOLLOWER.value or target_type == Target_Type.ALLIED_CARD.value or\
                    target_type == Target_Type.ALLIED_AMULET.value:
                targeted_card = sim_field.card_location[player_num][prev_target_id]
                for i, real_card in enumerate(field.card_location[opponent_num]):
                    if real_card.eq(targeted_card):
                        target_id = i
                        return action_num, card_id, target_id

                mylogger.info("sim_target_id:{}".format(target_id))
                mylogger.info("real")
                field.show_field()
                mylogger.info("sim")
                sim_field.show_field()
                assert False,"error:{}".format(Target_Type(target_type).name)

            elif target_type == Target_Type.ENEMY.value:
                if prev_target_id == -1:
                    return action_num, card_id, target_id
                targeted_card = sim_field.card_location[player_num][prev_target_id]
                for i, real_card in enumerate(field.card_location[opponent_num]):
                    if real_card.eq(targeted_card):
                        target_id = i
                        return action_num, card_id, target_id

                mylogger.info("sim_target_id:{}".format(target_id))
                mylogger.info("real")
                field.show_field()
                mylogger.info("sim")
                sim_field.show_field()
                assert False,"error:{}".format(Target_Type(target_type).name)

            elif target_type == Target_Type.FOLLOWER.value or target_type == Target_Type.CARD.value or\
                target_type == Target_Type.ALLIED_CARD_AND_ENEMY_FOLLOWER.value:
                targeted_card = sim_field.card_location[prev_target_id[0]][prev_target_id[1]]
                for i in range(2):
                    if prev_target_id[0] != i:
                        continue
                    for j, real_card in enumerate(field.card_location[i]):
                        if real_card.eq(targeted_card):
                            target_id = (i,j)
                            return action_num, card_id, target_id

                mylogger.info("sim_target_id:{}".format(target_id))
                mylogger.info("real")
                field.show_field()
                mylogger.info("sim")
                sim_field.show_field()
                assert False,"error:{}".format(Target_Type(target_type).name)

            elif target_type == Target_Type.CARD_IN_HAND.value:
                if action_num == Action_Code.PLAY_CARD.value:
                    itself_index = sim_player.hand.index(sim_player.hand[prev_card_id])
                    real_index = player.hand.index(player.hand[card_id])
                    if prev_target_id > itself_index:
                        prev_target_id += 1
                    targeted_card = sim_field.card_location[player_num][prev_target_id]
                    for i, hand_card in enumerate(player.hand):
                        if hand_card.eq(targeted_card):
                            if i > real_index:
                                target_id = i - 1
                            else:
                                target_id = i
                            return action_num, card_id, target_id

                mylogger.info("sim_target_id:{}".format(target_id))
                mylogger.info("real")
                player.show_hand()
                mylogger.info("sim")
                sim_player.show_hand()
                assert False, "error:{}".format(Target_Type(target_type).name)

    return action_num, card_id, target_id