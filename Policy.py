from abc import ABCMeta, abstractmethod
import operator
import random
import copy
import math
import numpy as np
import datetime
import heapq
from scipy.special import comb
from collections import deque
# import networkx as nx
import Field_setting
# import matplotlib.pyplot as plt
import random
from my_moduler import get_module_logger
# import tensorflow as tmp_field_list
import os.path
import time

from card_setting import *
import torch
from Network_model import Net,state_change_to_full
import Game_setting
from collections import namedtuple
State_tuple = namedtuple('Value', ('state'))
PATH = 'model/value_net.pth'

mylogger = get_module_logger(__name__)
max_life_value = math.exp(-1) - math.exp(-20)
# from numba import jit
from my_enum import *
import warnings
import Embedd_Network_model
ACTION_SIZE = 45

# warnings.simplefilter('ignore')

class Policy:
    __metaclass__ = ABCMeta

    @abstractmethod
    def decide(self, player, opponent, field):
        pass

    @abstractmethod
    def __str__(self):
        pass


class RandomPolicy(Policy):
    def __defaults__(self):
        return None

    def __str__(self):
        return 'RandomPolicy'

    def __init__(self):
        self.policy_type = 0
        self.name = "RandomPolicy"

    def decide(self, player, opponent, field):
        (ward_list, _, can_be_attacked, regal_targets) = field.get_situation(player, opponent)
        (can_play, can_attack, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) \
            = field.get_flag_and_choices(player, opponent, regal_targets)
        target_id = 0
        # length = len(able_to_play + able_to_creature_attack) + 1
        length = len(able_to_play + able_to_creature_attack)

        depth = 1 - len(able_to_evo)
        if not can_evo:
            depth = 0
        tmp = random.randint(depth, length)

        if tmp < 1 and can_evo:
            card_id = random.choice(able_to_evo)
            card = field.card_location[player.player_num][card_id]
            target_id = None
            if card.evo_target is not None:
                choices = field.get_regal_targets(card, player_num=player.player_num)
                if choices != []:
                    target_id = random.choice(choices)
            return Action_Code.EVOLVE.value, card_id, target_id

        # if tmp == 1 or (not can_play and not can_attack):
        if (length + len(able_to_evo)) == 0:
            return Action_Code.TURN_END.value, 0, 0

        if 0 <= tmp <= len(able_to_play) - 1 and can_play:

            card_id = random.choice(able_to_play)
            if player.hand[card_id].have_target == 0:
                return Action_Code.PLAY_CARD.value, card_id, None
            else:
                target_id = None
                if regal_targets[card_id] != []:
                    target_id = random.choice(regal_targets[card_id])
                return Action_Code.PLAY_CARD.value, card_id, target_id

        elif tmp > len(able_to_play) - 1 and len(able_to_creature_attack) > 0:
            card_id = random.choice(able_to_creature_attack)
            if ward_list != []:
                target_id = random.choice(ward_list)
                return Action_Code.ATTACK_TO_FOLLOWER.value, card_id, target_id

            else:
                if len(can_be_attacked) > 0:
                    targets = can_be_attacked
                    if card_id in able_to_attack:
                        targets.append(-1)
                    target_id = random.choice(targets)
                    if target_id == -1:
                        return Action_Code.ATTACK_TO_PLAYER.value, card_id, None
                    return Action_Code.ATTACK_TO_FOLLOWER.value, card_id, target_id

                elif card_id in able_to_attack:
                    return Action_Code.ATTACK_TO_PLAYER.value, card_id, None

        return Action_Code.TURN_END.value, 0, 0


class AggroPolicy(Policy):
    def __str__(self):
        return 'AgrroPolicy'

    def __init__(self):
        self.policy_type = 1
        self.name = "AggroPolicy"

    def decide(self, player, opponent, field):
        (ward_list, _, can_be_attacked, regal_targets) = field.get_situation(player, opponent)
        (can_play, can_attack, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) \
            = field.get_flag_and_choices(player, opponent, regal_targets)

        if len(able_to_play) == 0:
            can_play = False
        if len(able_to_creature_attack) == 0:
            can_attack = False

        if can_play:
            able_to_play_cost = [player.hand[i].cost for i in able_to_play]
            card_id = able_to_play[able_to_play_cost.index(max(able_to_play_cost))]
            target_id = None
            if regal_targets[card_id] != []:
                target_id = random.choice(regal_targets[card_id])
            return Action_Code.PLAY_CARD.value, card_id, target_id

        if can_evo:
            can_evolve_power = [field.card_location[player.player_num][i].power for i in able_to_evo]
            max_index = can_evolve_power.index(max(can_evolve_power))
            card_id = able_to_evo[max_index]
            target_id = None
            card = field.card_location[player.player_num][card_id]
            if card.evo_target is not None:
                choices = field.get_regal_targets(card, player_num=player.player_num)
                if choices != []:
                    target_id = random.choice(choices)
            return Action_Code.EVOLVE.value, card_id, target_id  # 進化

        if can_attack:
            opponent_creatures_stats = []
            able_to_creature_attack_power = [field.card_location[player.player_num][i].power for i in
                                             able_to_creature_attack]
            card_id = able_to_creature_attack[able_to_creature_attack_power.index(max(able_to_creature_attack_power))]
            attack_creature = field.card_location[player.player_num][card_id]
            target_id = None
            creature_attack_flg = True
            if ward_list != []:
                ward_creatures_stats = [(field.card_location[opponent.player_num][ward_list[i]].power,
                                         field.card_location[opponent.player_num][ward_list[i]].get_current_toughness())
                                        for i in range(len(ward_list))]
                opponent_creatures_stats = ward_creatures_stats

                for i, ele in enumerate(ward_creatures_stats):
                    if ele[1] <= sum(able_to_creature_attack_power):
                        target_id = ward_list[i]
                        break

            else:
                able_to_attack_power = [field.card_location[player.player_num][i].power for i in able_to_attack]
                if (len(field.get_creature_location()[opponent.player_num]) > 0 or opponent.life - sum(
                        able_to_attack_power) > 0):

                    opponent_creatures_stats = [(field.card_location[opponent.player_num][i].power,
                                                 field.card_location[opponent.player_num][i].get_current_toughness())
                                                for i in can_be_attacked]
                    leader_attack_flg = True
                    for i in range(len(can_be_attacked)):
                        if opponent_creatures_stats[i][1] <= attack_creature.power:
                            if opponent_creatures_stats[i][0] < attack_creature.get_current_toughness():
                                target_id = can_be_attacked[i]
                                leader_attack_flg = False
                                break

                    if leader_attack_flg and attack_creature.can_attack_to_player():
                        creature_attack_flg = False

                elif attack_creature.can_attack_to_player():
                    creature_attack_flg = False

            if not creature_attack_flg:
                if attack_creature.player_attack_regulation is None or attack_creature.player_attack_regulation(
                        player):
                    return Action_Code.ATTACK_TO_PLAYER.value, card_id, None

            if len(opponent_creatures_stats) > 0 and target_id is not None:
                return Action_Code.ATTACK_TO_FOLLOWER.value, card_id, target_id

        return Action_Code.TURN_END.value, 0, 0

class GreedyPolicy(Policy):

    def __init__(self):
        self.policy_type = 2
        self.name = "GreedyPolicy"

    def state_value(self, field, player_num):

        partial_observable_data = field.get_observable_data(player_num=player_num)
        if field.check_game_end():
            if partial_observable_data["player"]["life"] <= 0 or field.players[player_num].lib_out_flg:
                return 0
            elif partial_observable_data["opponent"]["life"] <= 0 or field.players[1 - player_num].lib_out_flg:
                return 1
        card_location = field.card_location
        life_diff = partial_observable_data["player"]["life"] - partial_observable_data["opponent"]["life"]
        life_diff /= 40
        hand_diff = partial_observable_data["player"]["hand_len"] - partial_observable_data["opponent"]["hand_len"]
        hand_diff /= 18
        board_diff = len(card_location[player_num]) - len(card_location[1 - player_num])
        board_diff /= 10
        value = board_diff * 0.45 + life_diff * 0.45 + hand_diff * 0.1
        return 1 / (1 + np.exp(-(value * 10)))

    def decide(self, player, opponent, field):
        (ward_list, can_be_targeted, can_be_attacked, regal_targets) = field.get_situation(player, opponent)
        (can_play, can_attack, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) \
            = field.get_flag_and_choices(player, opponent, regal_targets)
        first = player.player_num
        dicision = [0, 0, 0]
        action_set = []
        max_state_value = -10000
        length = len(able_to_play + able_to_creature_attack) + 1
        end_field_id_list = []

        tmp_field_list = []
        for i in range(length):
            new_field = Field_setting.Field(5)
            new_field.set_data(field)

            tmp_field_list.append(new_field)
        state_value_list = [0 for i in range(length)]  # 各行動後の状態価値のリスト
        state_value_list[0] = self.state_value(tmp_field_list[0], first)

        target_creatures_toughness = []
        if len(can_be_targeted) > 0:
            target_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness() \
                                          for i in can_be_targeted]

        if can_evo and len(able_to_evo) > 0 and able_to_play == []:

            leader_flg = False
            can_evolve_power = [field.card_location[player.player_num][i].power for i in able_to_evo]
            evo_id = able_to_evo[can_evolve_power.index(max(can_evolve_power))]

            direct_index = able_to_evo[0]
            for i, ele in enumerate(able_to_evo):
                if field.card_location[player.player_num][ele].can_attack_to_player():
                    leader_flg = True
                    direct_index = ele

                    break
            if len(field.get_creature_location()[opponent.player_num]) == 0 and leader_flg:
                evo_id = direct_index
            assert evo_id in able_to_evo
            target_id = None
            creature = field.card_location[first][evo_id]
            if creature.evo_target is not None:
                choices = field.get_regal_targets(creature, player_num=player.player_num)
                if choices != []:
                    target_id = random.choice(choices)
            evo_field = Field_setting.Field(5)
            evo_field.set_data(field)
            assert evo_field.get_able_to_evo(evo_field.players[first]) == field.get_able_to_evo(player)
            evo_field.players[first].execute_action(evo_field, evo_field.players[1 - first],
                                                    action_code=(Action_Code.EVOLVE.value, evo_id, target_id),
                                                    virtual=True)

            tmp_state_value = self.state_value(evo_field, first)
            action_set.append([(Action_Code.EVOLVE.value, evo_id, target_id), tmp_state_value])
            if max_state_value < tmp_state_value:
                max_state_value = tmp_state_value
                dicision = [Action_Code.EVOLVE.value, evo_id, target_id]


        if can_play:
            for i in range(1, len(able_to_play) + 1):
                assert i not in end_field_id_list, "{} {}".format(i, end_field_id_list)
                target_id = None
                card_id = able_to_play[i - 1]
                card = player.hand[card_id]
                if card.card_category != "Spell" and len(field.card_location[player.player_num]) == field.max_field_num:
                    continue
                if card.have_target != 0:
                    choices = field.get_regal_targets(card, target_type=1, player_num=player.player_num)
                    if choices != []:
                        target_id = random.choice(choices)
                    elif card.card_category == "Spell":
                        assert False,"target_category:{} name:{}".format(card.have_target, card.name)
                tmp_field_list[i].players[first].execute_action(tmp_field_list[i], tmp_field_list[i].players[1 - first], \
                                                                action_code=(
                                                                    Action_Code.PLAY_CARD.value, card_id, target_id),
                                                                virtual=True)

                state_value_list[i] = self.state_value(tmp_field_list[i], first)
                action_set.append([(Action_Code.PLAY_CARD.value, card_id, target_id),state_value_list[i]])
                end_field_id_list.append(i)
                if max_state_value < state_value_list[i] and tmp_field_list[i].players[first].life > 0:
                    max_state_value = state_value_list[i]
                    dicision = [Action_Code.PLAY_CARD.value, card_id, target_id]


        opponent_creatures_toughness = []
        if len(can_be_attacked) > 0:
            opponent_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness() \
                                            for i in can_be_attacked]

        if can_attack:
            if len(ward_list) > 0:
                ward_creatures_toughness = [field.card_location[opponent.player_num][i].get_current_toughness() \
                                            for i in ward_list]
                able_to_creature_attack_power = [field.card_location[player.player_num][i].power for i in
                                                 able_to_creature_attack]

                for i in range(len(able_to_play) + 1, length):
                    assert i not in end_field_id_list, "{} {}".format(i, end_field_id_list)
                    target_id = None
                    attacker_id = able_to_creature_attack[i - len(able_to_play) + 1 - 2]
                    attacker_power = tmp_field_list[i].card_location[first][attacker_id].power
                    if min(ward_creatures_toughness) <= sum(able_to_creature_attack_power):
                        target_id = ward_list[ward_creatures_toughness.index(min(ward_creatures_toughness))]
                        tmp_field_list[i].players[first].execute_action(tmp_field_list[i],
                                                                        tmp_field_list[i].players[1 - first],
                                                                        action_code=(
                                                                            Action_Code.ATTACK_TO_FOLLOWER.value,
                                                                            attacker_id, target_id), virtual=True)

                        state_value_list[i] += self.state_value(tmp_field_list[i], first)
                        action_set.append([(Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id),state_value_list[i]])
                        if max_state_value < state_value_list[i] and target_id is not None:
                            max_state_value = state_value_list[i]
                            dicision = [Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id]

            else:
                for i in range(len(able_to_play) + 1, length):
                    assert i not in end_field_id_list, "{} {}".format(i, end_field_id_list)
                    direct_flg = False
                    target_id = None
                    attacker_id = able_to_creature_attack[i - (len(able_to_play) + 1)]
                    attacking_creature = tmp_field_list[i].card_location[player.player_num][attacker_id]
                    assert attacking_creature.can_attack_to_follower()

                    attacker_power = attacking_creature.power

                    if (len(opponent_creatures_toughness) == 0 or min(opponent_creatures_toughness) > attacker_power \
                            ) and attacking_creature.can_attack_to_player():
                        if attacker_id in able_to_attack:
                            return Action_Code.ATTACK_TO_PLAYER.value, attacker_id, 0


                    elif (opponent_creatures_toughness) != [] and min(opponent_creatures_toughness) <= attacker_power:
                        target_id = can_be_attacked[
                            opponent_creatures_toughness.index(min(opponent_creatures_toughness))]
                        defencing_creature = tmp_field_list[i].card_location[opponent.player_num][target_id]
                        assert defencing_creature.can_be_attacked()
                        tmp_field_list[i].players[first].execute_action(tmp_field_list[i],
                                                                        tmp_field_list[i].players[1 - first], \
                                                                        action_code=(
                                                                            Action_Code.ATTACK_TO_FOLLOWER.value,
                                                                            attacker_id, target_id), virtual=True)
                        state_value_list[i] += self.state_value(tmp_field_list[i], first)
                        action_set.append([(Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id),state_value_list[i]])
                        end_field_id_list.append(i)

                    elif attacking_creature.can_attack_to_player():
                        if attacker_id in able_to_attack:
                            direct_flg = True
                            return Action_Code.ATTACK_TO_PLAYER.value, attacker_id, 0

                    if tmp_field_list[i].players[1 - first].life <= 0:
                        state_value_list[i] = 10000

                    if max_state_value < state_value_list[i]:
                        if direct_flg  and attacking_creature.can_attack_to_player():
                            if attacker_id in able_to_attack:
                                max_state_value = state_value_list[i]
                                dicision = [Action_Code.ATTACK_TO_PLAYER.value, attacker_id, 0]
                                action_set.append((Action_Code.ATTACK_TO_PLAYER.value, attacker_id, 0))
                        if not direct_flg and opponent_creatures_toughness != [] and target_id is not None:
                            max_state_value = state_value_list[i]
                            dicision = [Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id]
                            action_set.append((Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id))

        if not field.secret:
            for code in action_set:
                mylogger.info("action:{},value:{:.3f}".format(code[0],code[1]))
        return dicision[0], dicision[1], dicision[2]


class FastGreedyPolicy(GreedyPolicy):
    def __init__(self):
        self.f = lambda x: int(x) + (1, 0)[x - int(x) < 0.5]
        self.name = "FastGreedyPolicy"

    def __str__(self):
        return 'FastGreedyPolicy(now freezed)'



class Advanced_GreedyPolicy(GreedyPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Advanced_GreedyPolicy"

    def state_value(self, field, player_num):
        return advanced_state_value(field,player_num)

class Default_GreedyPolicy(GreedyPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Default_GreedyPolicy"

    def state_value(self, field, player_num):
        return default_state_value(field,player_num)


class Node:

    def __init__(self, field=None, player_num=0, finite_state_flg=False, depth=0):
        self.field = field
        self.finite_state_flg = finite_state_flg
        self.value = 0.0
        self.visit_num = 0
        self.is_root = False
        self.uct = 0.0
        self.depth = depth
        self.child_nodes = []
        self.max_child_visit_num = [None, None]
        self.parent_node = None
        self.regal_targets = {}
        self.action_value_dict = {}
        self.current_probability = None
        self.children_moves = [(0, 0, 0)]
        self.player_num = player_num
        self.field_value = 0
        children_moves = []
        end_flg = field.check_game_end()
        if not end_flg:
            player = field.players[player_num]
            #player.sort_hand()
            field.update_hand_cost(player_num=player_num)
            (ward_list, _, can_be_attacked, regal_targets) = \
                field.get_situation(player, field.players[1 - player_num])

            (_, _, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) = \
                field.get_flag_and_choices(player, field.players[1 - player_num], regal_targets)
            self.regal_targets = regal_targets
            children_moves.append((0, 0, 0))
            hand_id = 0
            # mylogger.info("{},{},{},{}".format(able_to_play,able_to_creature_attack,able_to_attack,able_to_evo))
            remain_able_to_play = able_to_play[:]
            while hand_id < len(player.hand):
                if hand_id in remain_able_to_play:
                    # mylogger.info("remain:{}".format(remain_able_to_play))
                    other_hand_ids = remain_able_to_play[:]
                    other_hand_ids.remove(hand_id)
                    # mylogger.info("other_hand_ids:{}".format(other_hand_ids))
                    if other_hand_ids is not None:
                        for other_id in other_hand_ids:
                            if player.hand[hand_id].eq(player.hand[other_id]):
                                remain_able_to_play.remove(other_id)
                hand_id += 1
            able_to_play = remain_able_to_play

            remain_able_to_creature_attack = able_to_creature_attack[:]
            remain_able_to_attack = able_to_attack[:]
            remain_able_to_evo = able_to_evo[:]
            location_id = 0
            side = field.card_location[player_num]
            while location_id < len(side):
                if location_id in remain_able_to_creature_attack:
                    other_follower_ids = remain_able_to_creature_attack[:]
                    other_follower_ids.remove(location_id)
                    if other_follower_ids is not None:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_creature_attack.remove(other_id)
                if location_id in remain_able_to_attack:
                    other_follower_ids = remain_able_to_attack[:]
                    other_follower_ids.remove(location_id)
                    if other_follower_ids is not None:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_attack.remove(other_id)
                if location_id in remain_able_to_evo:
                    other_follower_ids = remain_able_to_evo[:]
                    other_follower_ids.remove(location_id)
                    if other_follower_ids is not None:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_evo.remove(other_id)
                location_id += 1
            able_to_creature_attack = remain_able_to_creature_attack
            able_to_attack = remain_able_to_attack
            able_to_evo = remain_able_to_evo
            remain_can_be_attacked = can_be_attacked[:]
            remain_ward_list = ward_list[:]
            side = field.card_location[1 - player_num]
            location_id = 0
            while location_id < len(side):
                if location_id in remain_can_be_attacked:
                    other_follower_ids = remain_can_be_attacked[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_can_be_attacked.remove(other_id)
                            # elif side[location_id].name == side[other_id].name:
                            #    mylogger.info("{}".format(side[location_id]))
                            #    mylogger.info("{}".format(side[other_id]))
                if location_id in remain_ward_list:
                    other_follower_ids = remain_ward_list[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_ward_list.remove(other_id)
                                assert len(remain_ward_list) > 0
                location_id += 1
            can_be_attacked = remain_can_be_attacked
            ward_list = remain_ward_list
            for play_id in able_to_play:
                if player.hand[play_id].card_category != "Spell" and len(
                        field.card_location[player_num]) >= field.max_field_num:
                    continue

                if len(regal_targets[play_id]) > 0:
                    for i in range(len(regal_targets[play_id])):
                        children_moves.append((Action_Code.PLAY_CARD.value, play_id, regal_targets[play_id][i]))
                else:
                    if player.hand[play_id].card_category == "Spell":
                        if player.hand[play_id].have_target == 0:
                            children_moves.append((Action_Code.PLAY_CARD.value, play_id, None))
                    else:
                        children_moves.append((Action_Code.PLAY_CARD.value, play_id, None))

            if len(ward_list) == 0:
                for attacker_id in able_to_creature_attack:
                    for target_id in can_be_attacked:
                        children_moves.append((Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id))
                for attacker_id in able_to_attack:
                    children_moves.append((Action_Code.ATTACK_TO_PLAYER.value, attacker_id, None))

            else:
                for attacker_id in able_to_creature_attack:
                    for target_id in ward_list:
                        children_moves.append((Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id))

            if can_evo:
                for evo_id in able_to_evo:
                    evo_creature = field.card_location[player_num][evo_id]
                    if evo_creature.evo_target is None:
                        children_moves.append((Action_Code.EVOLVE.value, evo_id, None))
                    else:
                        targets = field.get_regal_targets(evo_creature, target_type=0, player_num=player_num)
                        for target_id in targets:
                            children_moves.append((Action_Code.EVOLVE.value, evo_id, target_id))

        self.children_moves = children_moves

    def print_tree(self, single=False):
        if self.is_root:
            print("ROOT(id:{})".format(id(self)))
        # if self.parent_node!=None:
        #    print("   "*self.depth+("parent_id:{})".format(id(self.parent_node))))
        depth = self.depth
        if single:
            depth = 0
        print("   " * depth + "depth:{} finite:{} mean_value:{} field_value:{} visit_num:{} player_num:{}".format(
            depth,
            self.finite_state_flg, int(self.value / max(1, self.visit_num)), self.field_value, self.visit_num,
            self.player_num))

        if self.child_nodes != []:
            print("   " * depth + "child_node_num:{}".format(len(self.child_nodes)))
            print("   " * depth + "child_node_set:{")
            for child in self.child_nodes:
                action_name = Action_Code(child[0][0]).name
                txt = ""
                if child[0][0] == Action_Code.PLAY_CARD.value:
                    play_card_name = self.field.players[self.player_num].hand[child[0][1]].name
                    txt = "{},{}({}),{}".format(action_name, play_card_name, child[0][1], child[0][2])
                elif child[0][0] == Action_Code.ATTACK_TO_FOLLOWER.value:
                    attacker_name = self.field.card_location[self.player_num][child[0][1]].name
                    defencer_name = self.field.card_location[1 - self.player_num][child[0][2]].name
                    txt = "{},{}({}),{}({})".format(action_name, attacker_name, child[0][1], defencer_name, child[0][2])
                elif child[0][0] == Action_Code.ATTACK_TO_PLAYER.value:
                    attacker_name = self.field.card_location[self.player_num][child[0][1]].name
                    txt = "{},{}({})".format(action_name, attacker_name, child[0][1])
                elif child[0][0] == Action_Code.EVOLVE.value:
                    evo_name = self.field.card_location[self.player_num][child[0][1]].name
                    txt = "{},{}({}),{}".format(action_name, evo_name, child[0][1], child[0][2])
                else:
                    txt = "{}".format(action_name)

                print("   " * depth + "action:{}".format(txt))
                if action_name=="TURN_END":
                    print("   " * depth + "opponent_moves_num:{}".format(len(child[1].children_moves)))
                if not single:
                    child[1].print_tree()
                else:
                    print("   " * depth + "ave_value:{},field_value:{},visit_num:{}"
                          .format(child[1].value / max(1, child[1].visit_num),child[1].field_value, child[1].visit_num))
                print("   " * depth + "}")

    def get_exist_action(self):
        exist_action = []
        for i in range(len(self.child_nodes)):
            exist_action.append(self.child_nodes[i][0])

        return exist_action

    def get_able_action_list(self):
        return sorted(list(set(self.children_moves) - set(self.get_exist_action())))

    def print_estimated_action_value(self):
        if self.visit_num == 0:
            print("this node is never simulated!")
            return
        print("visit_num:{} depth:{}".format(self.visit_num, self.depth))
        for key in list(self.action_value_dict.keys()):
            target_child_node = list(filter(lambda cell: cell[0] == key, self.child_nodes))[0]
            print("{}(visit_num:{}):{}".format(key, target_child_node[1].visit_num,
                                               self.action_value_dict[key] / (self.visit_num + 1)))

        # for cell in self.child_nodes:
        #    print("{}:{} times".format(cell[0],cell[1].visit_num))
        # print("")


EPSILON = 10e-6
ITERATION = 100
WIN_BONUS = 100000


class MCTSPolicy(Policy):
    def __init__(self):
        self.action_seq = []
        self.initial_seq = []
        self.num_simulations = 0
        self.uct_c = 1. / np.sqrt(2)
        self.play_out_policy = RandomPolicy()
        self.end_count = 0
        self.decide_node_seq = []
        self.starting_node = None
        self.current_node = None
        self.next_node = None
        self.node_index = 0
        self.policy_type = 3
        self.name = "MCTSPolicy"
        self.type = "root"
        self.iteration = 100
        self.default_iteration = 10

    # import numba
    # @numba.jit
    def state_value(self, field, player_num):
        if field.players[1 - player_num].life <= 0:
            return WIN_BONUS
        power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            power_sum += field.card_location[1 - player_num][card_id].power
        if power_sum >= field.players[player_num].life:
            return -WIN_BONUS

        return (field.players[1 - player_num].max_life - field.players[1 - player_num].life) * 50 + \
               (len(field.card_location[player_num]) - len(
                   field.card_location[1 - player_num])) * 50 + len(field.players[player_num].hand)

    def decide(self, player, opponent, field):
        if self.current_node is None:
            if not field.secret:
                mylogger.info("generate new tree")
            action = self.uct_search(player, opponent, field)
            if not field.secret:
                mylogger.info("root tree")
                self.current_node.print_tree()
            tmp_move = action
            #if len(self.current_node.child_nodes) > 0:
            if self.current_node.child_nodes == []:
                self.current_node = None
                return 0,0,0
            #next_node, tmp_move = self.best(self.current_node, player_num=player.player_num)
            next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
            self.current_node = next_node
            #if not field.secret and len(self.current_node.child_nodes) != len(self.current_node.children_moves):
            #    mylogger.info("children_moves:{}".format(self.starting_node.children_moves))
            #    self.starting_node.print_tree(single=True)
            self.type = "root"
            if tmp_move == (0, 0, 0):
                #if not field.secret:
                #    if self.current_node.parent_node is not None:
                #        if self.current_node.parent_node.children_moves != [(0,0,0)]:
                #            mylogger.info("turn end is prior to other moves")
                #            mylogger.info("children_moves:{}".format(self.current_node.parent_node.children_moves))
                #            self.current_node.parent_node.print_tree(single=True)
                self.current_node = None
            return tmp_move  # action
        else:
            self.type = "blanch"

            next_node = None
            tmp_move = None
            if not self.fully_expand(self.current_node, player_num=player.player_num):
                #if not field.secret:
                #    mylogger.info("reuse existed node as root(able_move_num:{},child_num:{})"
                #                  .format(len(self.current_node.children_moves),len(self.current_node.child_nodes)))
                action = self.uct_search(player, opponent, field)
                tmp_move = action
                if len(self.current_node.child_nodes) > 0:
                    next_node, tmp_move = self.best(self.current_node, player_num=player.player_num)
                    self.current_node = next_node
                if tmp_move == (0, 0, 0):
                    self.current_node = None
                return tmp_move
            if len(self.current_node.child_nodes) == 0:
                #if not field.secret:
                #    mylogger.info("no child error")
                return Action_Code.ERROR.value, 0, 0
            #next_node, tmp_move = self.best(self.current_node, player_num=player.player_num)
            next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
            self.current_node = next_node
            if tmp_move == (0, 0, 0):
                if not field.secret:
                    if self.current_node.parent_node is not None:
                        if self.current_node.parent_node.children_moves != [(0,0,0)]:
                            mylogger.info("turn end is prior to other moves")
                            mylogger.info("children_moves:{}".format(self.current_node.parent_node.children_moves))
                            self.current_node.parent_node.print_tree(single=True)
                self.current_node = None
            return tmp_move  # action

    def uct_search(self, player, opponent, field):
        field.get_regal_target_dict(player, opponent)
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field, player_num=player.player_num)
        starting_node.is_root = True
        self.starting_node = starting_node
        self.current_node = starting_node
        self.decide_node_seq = []
        self.decide_node_seq.append(starting_node)
        if starting_node.get_able_action_list() == [(0, 0, 0)]:
            return 0, 0, 0
        for i in range(self.iteration):
            if time.time() - field.time > 89:
                break
            node = self.tree_policy(starting_node, player_num=player_num)
            if node.field.players[1 - player_num].life <= 0 and node.field.players[player_num].life > 0:
                self.back_up(node, WIN_BONUS, player_num=player_num)
                break
            value = self.default_policy(node, player_num=player_num)
            self.back_up(node, value, player_num=player_num)

            if starting_node.max_child_visit_num[0] is not None and starting_node.max_child_visit_num[1] is not None:
                if starting_node.max_child_visit_num[1].visit_num - \
                        starting_node.max_child_visit_num[0].visit_num > self.iteration - i:
                    break

            elif starting_node.max_child_visit_num[1] is not None:
                if starting_node.max_child_visit_num[1].visit_num > int(self.iteration / 2):
                    break
        if len(self.starting_node.child_nodes) > 0:
            _, move = self.best(self.starting_node, player_num=player_num)
        else:
            move = Action_Code.TURN_END.value, 0, 0

        assert self.starting_node is not None and self.current_node is not None, "{},{}".format(self.starting_node,
                                                                                                self.current_node)
        return move

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and check is False:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    node, _ = self.best(node, player_num=player_num)
                else:
                    check = self.fully_expand(node, player_num=player_num)
                    if not check:
                        return self.expand(node, player_num=player_num)
                    else:
                        node, _ = self.best(node, player_num=player_num)
                length_of_children = len(node.child_nodes)
            else:
                return self.expand(node, player_num=player_num)

            count += 1
            if count > 100:
                field = node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            if not node.finite_state_flg:
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)

                current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                player = current_field.players[player_num]
                opponent = current_field.players[1 - player_num]
                # current_field.get_regal_target_dict(player, opponent)

                action_count = 0
                while True:
                    (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

                    end_flg = player.execute_action(current_field, opponent,
                                                    action_code=(action_num, card_id, target_id), virtual=True)

                    if current_field.check_game_end() or end_flg:
                        break

                    # current_field.get_regal_target_dict(player, opponent)
                    action_count += 1
                    if action_count > 100:
                        player.show_hand()
                        current_field.show_field()
                        assert False
                if current_field.check_game_end():
                    sum_of_value += WIN_BONUS
                    return sum_of_value
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end():
                    sum_of_value += WIN_BONUS
                    return sum_of_value
                else:
                    sum_of_value += self.state_value(current_field, player_num)

        return sum_of_value / self.default_iteration

    def fully_expand(self, node, player_num=0):
        if len(node.get_able_action_list()) == 0:
            return True
        return len(node.child_nodes) == len(node.children_moves) or node.finite_state_flg  # turn_endの場合を追加

    def expand(self, node, player_num=0):

        field = node.field

        new_choices = node.get_able_action_list()
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.players[0].deck.shuffle()
        next_field.players[1].deck.shuffle()
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])
        # regal_targetsの更新のため
        next_node = None
        if move[0] == 0:
            next_node = Node(field=next_field, player_num=player_num, finite_state_flg=True, depth=node.depth + 1)
        else:
            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = next_field.check_game_end()
            next_node = Node(field=next_field, player_num=player_num,
                             finite_state_flg=bool(flg), depth=node.depth + 1)
        next_node.parent_node = node
        node.child_nodes.append((move, next_node))
        return next_node

    def best(self, node, player_num=0):
        children = node.child_nodes
        uct_values = {}
        for i in range(len(children)):
            uct_values[i] = self.uct(children[i][1], node, player_num=player_num, end_flg=children[i][0][0] == 0)

        uct_values_list = list(uct_values.values())
        max_uct_value = max(uct_values_list)
        max_list_index = uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]

        return max_value_node, action

    def execute_best(self, node, player_num=0):
        children = node.child_nodes
        values = {}
        for i in range(len(children)):
            values[i] = children[i][1].value/children[i][1].visit_num

        values_list = list(values.values())
        max_uct_value = max(values_list)
        max_list_index = values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]

        return max_value_node, action

    def uct(self, child_node, node, player_num=0, end_flg=False):
        over_all_n = node.visit_num
        n = child_node.visit_num
        w = child_node.value
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        if end_flg:
            exploitation_value = max(exploitation_value / 10, exploitation_value - 0.5)
        # if node.node_id_2_edge_action[id(child_node)][0] == Action_Code.TURN_END.value:
        #    exploitation_value = max(exploitation_value / 10, exploitation_value - 0.5)
        # assert over_all_n>0,"over_all_n:{}".format(over_all_n)
        if over_all_n == 0:
            over_all_n = 1
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / (n + epsilon))

        value = exploitation_value + exploration_value

        child_node.uct = value

        return value

    def back_up(self, last_visited, reward, player_num=0):
        current = last_visited
        while True:
            current.visit_num += 1
            current.value += reward

            if current.is_root:
                break

            elif current.parent_node.is_root:
                best_childen = current.parent_node.max_child_visit_num
                best_node = best_childen[1]
                second_node = best_childen[0]
                if best_node == None:  # ベストが空
                    best_node = current
                elif second_node == None:  # ベストが1つのみ
                    if current != best_node:
                        if best_node.visit_num < current.visit_num:
                            best_childen = [best_node, current]
                        else:
                            best_childen = [current, best_node]
                else:
                    if second_node.visit_num > best_node.visit_num:
                        best_children = [best_node, second_node]

                    different_flg = current not in best_children[0]
                    if different_flg:
                        if current.visit_num > best_children[1].visit_num:
                            best_children = [best_children[1], current]

                        elif current.visit_num > best_children[0].visit_num:
                            best_children = [current, best_children[1]]

            current = current.parent_node
            if current is None:
                break

    def __str__(self):
        return 'MCTSPolicy'


class Shallow_MCTSPolicy(MCTSPolicy):

    def __init__(self, th=10):
        super().__init__()
        self.name = "Shallow(th={})_MCTSPolicy".format(th)
        self.th = th

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and check is False:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    node, _ = self.best(node, player_num=player_num)
                else:
                    check = self.fully_expand(node, player_num=player_num)
                    if not check:
                        return self.expand(node, player_num=player_num)
                    else:
                        node, _ = self.best(node, player_num=player_num)
                length_of_children = len(node.child_nodes)
            else:
                if node.visit_num < self.th:
                    break
                return self.expand(node, player_num=player_num)

            count += 1
            if count > 100:
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node


class Alpha_Beta_MCTSPolicy(MCTSPolicy):

    def __init__(self, th=10):
        super().__init__()
        self.name = "Alpha_Beta_MCTSPolicy(th={})".format(th)
        self.th = th

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and check == False:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    node, _ = self.best(node, player_num=player_num)
                else:
                    check = self.fully_expand(node, player_num=player_num)
                    if not check:
                        return self.expand(node, player_num=player_num)
                    else:
                        node, _ = self.best(node, player_num=player_num)
                length_of_children = len(node.child_nodes)
            else:
                if node.visit_num < self.th:
                    break
                return self.expand(node, player_num=player_num)

            count += 1
            if count > 100:
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node

    def best(self, node, player_num=0):
        children = node.child_nodes
        uct_values = {}
        alpha = 0
        beta = 1000
        for i in range(len(children)):
            target = children[i][1]
            if target.visit_num > 0 and target.value / target.visit_num < alpha:
                uct_values[i] = -100
            else:
                uct_values[i] = self.uct(target, node, player_num=player_num)
                if uct_values[i] > alpha:
                    alpha = uct_values[i]

        uct_values_list = list(uct_values.values())
        max_uct_value = max(uct_values_list)
        max_list_index = uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]

        return max_value_node, action


class Test_MCTSPolicy(MCTSPolicy):
    # デッキ公開制のときのみ
    def __init__(self, sim_num=1):
        super().__init__()
        self.name = "Test(Random,sim_num={})_MCTSPolicy".format(sim_num)
        self.opponent_policy = RandomPolicy()
        self.sim_num = sim_num
    def decide(self, player, opponent, field):
        if self.current_node is None:

            action = self.uct_search(player, opponent, field)
            self.end_count += 1
            tmp_move = action
            if len(self.current_node.child_nodes) > 0:
                next_node, tmp_move = self.best(self.current_node, player_num=player.player_num)
                self.current_node = next_node
            elif not field.secret:
                mylogger.info("children_moves:{}".format(self.current_node.children_moves))
            self.node_index = 0
            self.type = "root"
            if tmp_move == (0, 0, 0):
                if not field.secret and self.current_node.parent_node is not None:
                    mylogger.info("children_moves:{}".format(self.current_node.parent_node.children_moves))
                    self.current_node.parent_node.print_estimated_action_value()
                self.current_node = None
            return tmp_move  # action
        else:
            self.type = "blanch"
            self.node_index += 1

            next_node = None
            tmp_move = None
            if len(self.current_node.child_nodes) == 0:
                if not field.secret:
                    mylogger.info("no child error")
                return Action_Code.ERROR.value, 0, 0
            next_node, tmp_move = self.best(self.current_node, player_num=player.player_num)
            self.current_node = next_node
            if tmp_move == (0, 0, 0):
                if not field.secret and self.current_node.parent_node is not None:
                    mylogger.info("children_moves:{}".format(self.current_node.parent_node.children_moves))
                    self.current_node.parent_node.print_estimated_action_value()
                self.current_node = None
            return tmp_move  # action
    def default_policy(self, node, player_num=0):
        current_field = Field_setting.Field(5)
        if node.field.check_game_end():
            current_field.set_data(node.field)
            current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(current_field, player_num)

        sum_of_value = 0
        end_flg = False

        for i in range(self.sim_num):

            current_field.set_data(node.field)
            current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
            current_field.players[1].deck.shuffle()
            player = current_field.players[player_num]
            opponent = current_field.players[1 - player_num]
            if not node.finite_state_flg:

                while True:
                    current_field.get_regal_target_dict(player, opponent)
                    (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent,
                                                                                   current_field)
                    end_flg = player.execute_action(current_field, opponent,
                                                    action_code=(action_num, card_id, target_id), virtual=True)

                    if current_field.check_game_end() == True or end_flg == True:
                        break
                if current_field.check_game_end() and player.life > 0:
                    sum_of_value += WIN_BONUS
                    continue
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end() and player.life > 0:
                    sum_of_value += WIN_BONUS
                    continue
            # 相手ターンのシミュレーション
            current_field.untap(1 - player_num)
            current_field.increment_cost(1 - player_num)
            current_field.start_of_turn(1 - player_num, virtual=True)

            opponent_hand = opponent.hand
            opponent_deck = opponent.deck
            hand_len = len(opponent.hand)
            while len(opponent.hand) > 0:
                opponent_deck.append(opponent_hand.pop())
            opponent_deck.shuffle()
            opponent.draw(opponent_deck, hand_len + 1)

            current_field.get_regal_target_dict(player, opponent)
            opponent.policy = self.opponent_policy
            while True:
                end_flg = opponent.decide(opponent, player, current_field, virtual=True)
                if end_flg:
                    break
            if current_field.check_game_end() and opponent.life > 0:
                sum_of_value = - WIN_BONUS
            else:
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end() and opponent.life > 0:
                    sum_of_value = - WIN_BONUS
                else:
                    sum_of_value += self.state_value(current_field, player_num)

        return sum_of_value / self.sim_num


class Test_2_MCTSPolicy(Test_MCTSPolicy):

    def __init__(self, sim_num=1):
        super().__init__(sim_num=sim_num)
        self.name = "Test(Aggro,sim_num={})_MCTSPolicy".format(sim_num)
        self.opponent_policy = AggroPolicy()


class Test_3_MCTSPolicy(Test_MCTSPolicy):

    def __init__(self, sim_num=1):
        super().__init__(sim_num=sim_num)
        self.name = "Test(Greedy,sim_num={})_MCTSPolicy".format(sim_num)
        self.opponent_policy = GreedyPolicy()


class Test_4_MCTSPolicy(Test_MCTSPolicy):

    def __init__(self, sim_num=1):
        super().__init__(sim_num=sim_num)
        self.name = "Test(A-MCTS,sim_num={})_MCTSPolicy".format(sim_num)
        self.opponent_policy = Expanded_Aggro_MCTS_Policy()

    def state_value(self, field, player_num):
        return self.opponent_policy.state_value(field, player_num)


class Five_Times_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "five_Times_MCTSPolicy"
        self.iteration = 500


class Ten_Times_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Ten_Times_MCTSPolicy"
        self.iteration = 1000


class Aggro_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Aggro_MCTSPolicy"
        # RandomPolicyとAggroPolicyのPlayOutでの比較
        self.play_out_policy = AggroPolicy()


class New_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "New_MCTSPolicy"

    def state_value(self, field, player_num):
        if field.check_game_end():
            if field.players[player_num].life <= 0 or field.players[player_num].lib_out_flg:
                return 0
            else:
                return 1
        power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            power_sum += field.card_location[1 - player_num][card_id].power
        if power_sum >= field.players[player_num].life and not field.check_ward()[player_num]:
            return 0
        cum_damage = (field.players[1 - player_num].max_life - field.players[1 - player_num].life) * 50
        board_ad = (len(field.card_location[player_num]) - len(
            field.card_location[1 - player_num])) * 50
        hand_len = len(field.players[player_num].hand)
        value = cum_damage + board_ad + hand_len +250
        max_value = 1509  # 1000+250+9+250

        return value / max_value

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            if not node.finite_state_flg:
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)

                current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                player = current_field.players[player_num]
                opponent = current_field.players[1 - player_num]

                action_count = 0
                while True:
                    (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

                    end_flg = player.execute_action(current_field, opponent,
                                                    action_code=(action_num, card_id, target_id), virtual=True)

                    if current_field.check_game_end() or end_flg:
                        break

                    action_count += 1
                    if action_count > 100:
                        player.show_hand()
                        current_field.show_field()
                        assert False
                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field, player_num)
                    return sum_of_value / (i + 1)
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field, player_num)
                    return sum_of_value / (i + 1)
                else:
                    sum_of_value += self.state_value(current_field, player_num)
        #mylogger.info("play_out_result:{}".format(sum_of_value / self.default_iteration))
        return sum_of_value / self.default_iteration


class New_Aggro_MCTSPolicy(New_MCTSPolicy):
    def __init__(self,iteration=100):
        super().__init__()
        self.name = "New_A_MCTS(n={})Policy".format(iteration)
        self.play_out_policy = AggroPolicy()
        self.iteration = iteration


class Default_Aggro_MCTSPolicy(New_MCTSPolicy):
    def __init__(self,iteration=100):
        super().__init__()
        self.name = "Default_A_MCTS(n={})Policy".format(iteration)
        self.play_out_policy = AggroPolicy()
        self.iteration = iteration

    def state_value(self, field, player_num):
        return default_state_value(field,player_num)


class EXP3_MCTSPolicy(Policy):
    def __init__(self):
        self.tree = None
        self.first_action_flg = False
        self.action_seq = []
        self.initial_seq = []
        self.num_simulations = 0
        self.uct_c = 1. / np.sqrt(2)
        self.play_out_policy = RandomPolicy()
        self.end_count = 0
        self.decide_node_seq = []
        self.starting_node = None
        self.current_node = None
        self.node_index = 0
        self.policy_type = 3
        self.next_node = None
        self.prev_node = None
        self.hyper_parameter = [1, 1, 1, 1, 1, 1]  # [1/6]*6
        self.eta = 1
        self.probability_check_func = lambda x: x >= 0 and x <= 1
        self.name = "EXP3_MCTSPolicy"
        self.iteration = 100
        self.default_iteration = 10

    def state_value(self, field, player_num):
        if field.check_game_end():
            if field.players[1 - player_num].life <= 0 or len(field.players[1 - player_num].deck.deck) == 0:
                return 1.0
            return 0.0
        power_sum = 0
        if len(field.get_ward_list(player_num)) == 0:
            for card_id in field.get_creature_location()[player_num]:
                if field.card_location[player_num][card_id].can_attack_to_player():
                    power_sum += field.card_location[player_num][card_id].power
            if power_sum >= field.players[player_num].life:
                return 1.0
        power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            power_sum += field.card_location[1 - player_num][card_id].power
        if power_sum >= field.players[player_num].life:
            return 0.0

        life_ad = (field.players[1 - player_num].max_life - field.players[1 - player_num].life) * 100
        board_ad = (len(field.get_creature_location()[player_num]) - len(
            field.get_creature_location()[1 - player_num])) * 10
        hand_ad = len(field.players[player_num].hand)
        return ((life_ad + board_ad + hand_ad) + 50) / ((2000 + 50 + 9) + 50)

    def decide(self, player, opponent, field):
        if self.current_node is None:

            self.exp3_search(player, opponent, field)

            if len(self.current_node.child_nodes) == 0:
                self.current_node = None
                return 0, 0, 0
            else:
                next_node, action, _ = self.roulette(self.current_node)
                self.current_node = next_node
                if action == (0, 0, 0): self.current_node = None
                return action

        else:
            if self.current_node.finite_state_flg or len(self.current_node.children_moves) == 1:
                self.current_node = None
                return 0, 0, 0
            elif len(self.current_node.child_nodes) == 0:
                return Action_Code.ERROR.value, 0, 0
            else:
                next_node, action, _ = self.roulette(self.current_node)
                self.current_node = next_node
                if action == (0, 0, 0):
                    self.current_node = None
                return action

    def exp3_search(self, player, opponent, field):
        field.get_regal_target_dict(player, opponent)
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field, player_num=player.player_num)
        mark_node = None
        starting_node.is_root = True
        self.starting_node = starting_node
        self.current_node = starting_node
        end_flg = False
        self.decide_node_seq = []
        self.decide_node_seq.append(starting_node)
        if starting_node.get_able_action_list() == [(0, 0, 0)]:
            # mylogger.info("check:{}".format(self.current_node==self.starting_node))
            return
        for i in range(self.iteration):
            if time.time() - field.time > 89:
                break
            node, probability = self.tree_policy(starting_node, player_num=player_num)
            value = self.default_policy(node, probability, player_num=player_num)
            self.back_up(node, value, player_num=player_num)
            if starting_node.max_child_visit_num[0] != None and starting_node.max_child_visit_num[1] != None:
                if starting_node.max_child_visit_num[1].visit_num - starting_node.max_child_visit_num[
                    0].visit_num > ITERATION - i:
                    break

            elif starting_node.max_child_visit_num[1] != None:
                if starting_node.max_child_visit_num[1].visit_num > int(self.iteration / 2):
                    break

        assert self.starting_node != None and self.current_node != None, "{},{}".format(self.starting_node,
                                                                                        self.current_node)
        # mylogger.info("check:{}".format(self.current_node==self.starting_node))
        assert len(self.current_node.field.card_location[0]) == len(field.card_location[0])
        assert len(self.current_node.field.card_location[1]) == len(field.card_location[1])
        assert len(self.current_node.field.players[player.player_num].hand) == len(
            field.players[player.player_num].hand)
        # mylogger.info("action_value:{}".format(self.current_node.action_value_dict))
        return  # move

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and check == False:
            return self.expand(node, player_num=player_num)
        count = 0
        probability = 0.01
        while not node.finite_state_flg:
            if length_of_children > 0:

                if random.uniform(0, 1) < .5:
                    node, action, probability = self.roulette(node, player_num=player_num)
                else:
                    check = self.fully_expand(node, player_num=player_num)
                    if check == False:
                        return self.expand(node, player_num=player_num)
                    else:
                        node, action, probability = self.roulette(node, player_num=player_num)

                length_of_children = len(node.child_nodes)
            else:
                return self.expand(node, player_num=player_num)

            count += 1
            if count > 100:
                field = node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node, probability

    def default_policy(self, node, probability, player_num=0):
        if node.finite_state_flg:
            action = None
            for cell in node.parent_node.child_nodes:
                if cell[-1] == node:
                    action = cell[0]
            if node.field.check_game_end():
                if action not in node.parent_node.action_value_dict:
                    node.parent_node.action_value_dict[action] = 0
                # mylogger.info("action:{} depth:{} value:{} probability:{}".format(action,node.depth,self.state_value(node.field,player_num),probability))
                node.parent_node.action_value_dict[action] += self.state_value(node.field, player_num) * (
                        100 / node.depth) / probability
                # mylogger.info("action_value:{}".format(node.parent_node.action_value_dict[action]))
            else:
                node.parent_node.action_value_dict[action] = -1000 * (node.parent_node.visit_num + 1) * (
                        1 / node.depth)  # 0
            # mylogger.info("{}:{},visit_num:{} depth:{}".format(action,self.state_value(node.field,player_num),node.visit_num,node.depth))
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            if not node.finite_state_flg:
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)

                current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                player = current_field.players[player_num]
                opponent = current_field.players[1 - player_num]
                # current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])

                action_count = 0
                while True:
                    current_field.get_regal_target_dict(player, opponent)
                    (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, \
                                                                                   current_field)

                    end_flg = player.execute_action(current_field, opponent, \
                                                    action_code=(action_num, card_id, target_id), virtual=True)

                    if current_field.check_game_end() == True or end_flg == True:
                        break

                    # current_field.get_regal_target_dict(current_field.players[player_num],current_field.players[1-player_num])
                    action_count += 1
                    if action_count > 100:
                        player.show_hand()
                        current_field.show_field()
                        assert False

                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field,player_num)
                    action = None
                    for cell in node.parent_node.child_nodes:
                        if cell[-1] == node:
                            action = cell[0]
                    node.parent_node.action_value_dict[action] = self.state_value(current_field, player_num) * (
                            node.visit_num + 1)
                    return (sum_of_value / (i + 1))
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end():
                    action = None
                    for cell in node.parent_node.child_nodes:
                        if cell[-1] == node:
                            action = cell[0]
                    node.parent_node.action_value_dict[action] = self.state_value(current_field, player_num) * (
                            node.visit_num + 1)
                    sum_of_value += self.state_value(current_field,player_num)
                    return (sum_of_value / (i + 1))
                else:
                    assert self.state_value(current_field, player_num) >= 0, "{},{}".format(
                        self.state_value(current_field, player_num), current_field.check_game_end())
                    sum_of_value += self.state_value(current_field, player_num)
            else:
                assert False, "finite:True"
        result = sum_of_value / self.default_iteration
        if node.parent_node != []:
            action = None
            for cell in node.parent_node.child_nodes:
                if cell[-1] == node:
                    action = cell[0]
            assert action is not None, "child_nodes:{}".format(node.parent_node.child_nodes)
            if action not in node.parent_node.action_value_dict:
                node.parent_node.action_value_dict[action] = 0.0
                # mylogger.info("append {} to action_value_dict:{}".format(action,node.parent_node.action_value_dict))
            assert result / probability >= 0.0, "result:{} probability:{} sum_of_value:{}".format(result, probability,
                                                                                                  sum_of_value)
            node.parent_node.action_value_dict[action] += result / probability
            # mylogger.info("{}:{},visit_num:{} depth:{}".format(action,result,node.visit_num,node.depth))
            # mylogger.info("now_value:{}".format(node.parent_node.action_value_dict[action]))
        return result

    def fully_expand(self, node, player_num=0):
        return len(node.child_nodes) == len(node.children_moves) or node.finite_state_flg == True  # turn_endの場合を追加

    def expand(self, node, player_num=0):

        field = node.field

        new_choices = node.get_able_action_list()
        if new_choices == []:
            field.show_field()

            assert False,\
            "children_moves:{} exist_action:{}".format(node.children_moves,
                                                       node.get_exist_action())
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.players[0].deck.shuffle()
        next_field.players[1].deck.shuffle()
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])  # regal_targetsの更新のため
        next_node = None

        if move[0] == 0:
            next_field.end_of_turn(player_num, virtual=True)
            # if (0,0,0) not in node.action_value_dict:
            # node.action_value_dict[move]=self.state_value(next_field,player_num)
            next_node = Node(field=next_field, player_num=player_num, finite_state_flg=True, depth=node.depth + 1)
        else:
            if move[0] == 1 and move[2] is not None and node.regal_targets[move[1]] != []:
                assert False,"Null target! in_node:{}".format(node.regal_targets[move[1]])

            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = (next_field.check_game_end() == True)
            if flg:
                # node.action_value_dict[move]=self.state_value(next_field,player_num)
                assert any(self.state_value(next_field, player_num) == i for i in [0, 1]), "{}".format(
                    (self.state_value(next_field, player_num)))
            next_node = Node(field=next_field, player_num=player_num, finite_state_flg=flg == True,
                             depth=node.depth + 1)
        next_node.parent_node = node
        node.child_nodes.append((move, next_node))
        next_node.current_probability = 1 / len(new_choices)
        # if move==(0,0,0):
        #   node.action_value_dict[move]=self.state_value(next_field,player_num)

        # node.action_value_dict[move]=0.0
        # mylogger.info("append {} to action_value_dict:{}".format(move,node.action_value_dict))
        return next_node, 1  # /len(new_choices)

    def best(self, node, player_num=0):
        children = node.child_nodes
        uct_values = {}
        for i in range(len(children)):
            uct_values[i] = self.uct(children[i][1], node, player_num=player_num)
        uct_values_list = list(uct_values.values())
        max_uct_value = max(uct_values_list)
        max_list_index = uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]

        return max_value_node, action

    def uct(self, child_node, node, player_num=0):
        over_all_n = node.visit_num
        n = child_node.visit_num
        w = child_node.value
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / (n + epsilon))

        value = exploitation_value + exploration_value

        child_node.uct = value

        return value

    def exp3(self, node, player_num=0):
        over_all_n = node.visit_num
        # A = len(node.child_nodes)
        dict_key_list = list(node.action_value_dict.keys())
        A = len(dict_key_list)
        assert A > 0, "child_nodes:{} child_moves:{} able_to_actions:{}".format(node.child_nodes, node.children_moves,
                                                                                node.get_exist_action())
        e = np.e
        # value = (A*np.log(A))/((e-1)*over_all_n)
        value = (A * np.log(A)) / ((e - 1) * over_all_n)
        value = np.sqrt(value)
        gamma = min(1, value)
        gamma = min(0.001, gamma)
        # gamma = 0.01
        # mylogger.info("gamma:{}".format(gamma))
        # gamma = 0.5
        # eta = gamma / over_all_n
        # eta = 10000*gamma/np.sqrt(over_all_n)
        # eta = 20000 * gamma / A
        eta = 20000
        distribution = [0.0] * len(dict_key_list)
        value_list = [node.action_value_dict[key] / over_all_n for key in dict_key_list]
        first_term = gamma / A
        max_value = max(value_list)

        value_list = [(value_list[i] - max_value) * eta for i in range(len(value_list))]
        value_list = np.exp(value_list) / np.sum(np.exp(value_list))
        for i in range(len(distribution)):
            second_term = (1 - gamma) * (value_list[i])
            distribution[i] = first_term + second_term

        return distribution

    def roulette(self, node, player_num=0, show_flg=False):
        distribution = self.exp3(node, player_num=player_num)
        if show_flg:
            mylogger.info("distribution:{}".format(distribution))
        assert len(distribution) > 0
        population = [key for key in list(node.action_value_dict.keys())]
        assert len(distribution) == len(population), "{},{}".format(len(distribution), len(population))
        assert np.nan not in distribution, "{}".format(distribution)
        decision = random.choices(population, weights=distribution, k=1)
        for cell in node.child_nodes:
            if cell[0] == decision[0]:
                decision_node = cell[-1]
                action = cell[0]
                index = population.index(cell[0])
                # mylogger.info("probability:{}".format(distribution[index]))
                decision_node.current_probability = distribution[index]
                return decision_node, action, distribution[index]
        assert decision[0] is not None, "population:{},distribution:{}".format(population, distribution)
        assert False, "{},{} {}".format(decision[0], node.action_value_dict, node.child_nodes)

    def back_up(self, last_visited, reward, player_num=0):
        current = last_visited
        probabilities = []
        while True:
            current.visit_num += 1
            # current.value += reward

            if current.is_root:
                break
            probabilities.append(current.current_probability)
            if current != last_visited:
                target_action = \
                    [cell[0] for i, cell in enumerate(current.parent_node.child_nodes) if cell[1] == current][0]
                p = 0
                # mylogger.info("probabilities:{}".format(probabilities))
                if len(probabilities) == 1:
                    p = probabilities[0]
                else:
                    p = np.prod(np.array(probabilities))
                current.parent_node.action_value_dict[target_action] += reward / p

            elif current.parent_node.is_root:
                best_childen = current.parent_node.max_child_visit_num
                best_node = best_childen[1]
                second_node = best_childen[0]
                if best_node is None:  # ベストが空
                    best_node = current
                elif second_node is None:  # ベストが1つのみ
                    if current != best_node:
                        if best_node.visit_num < current.visit_num:
                            best_childen = [best_node, current]
                        else:
                            best_childen = [current, best_node]
                else:
                    if second_node.visit_num > best_node.visit_num:
                        best_children = [best_node, second_node]

                    different_flg = current not in best_children[0]
                    if different_flg:
                        if current.visit_num > best_children[1].visit_num:
                            best_children = [best_children[1], current]

                        elif current.visit_num > best_children[0].visit_num:
                            best_children = [current, best_children[1]]

            current = current.parent_node
            if current is None:
                break

    def __str__(self):
        return 'EXP3_MCTSPolicy'


class Aggro_EXP3_MCTSPolicy(EXP3_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.play_out_policy = AggroPolicy()
        self.name = "Aggro_EXP3_MCTSPolicy"


class Quadruple_Aggro_EXP3_MCTSPolicy(EXP3_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.play_out_policy = AggroPolicy()
        self.name = "Quadruple_Aggro_EXP3_MCTSPolicy"
        self.iteration = 400


class Time_bounded_MCTSPolicy(MCTSPolicy):
    def __init__(self, limit=90, playout=RandomPolicy()):
        super().__init__()
        self.name = "Time_bounded(limit={},playout={})_MCTSPolicy" \
            .format(limit, playout.name.split("Policy")[0])
        self.limit = limit
        self.play_out_policy = playout

    def uct_search(self, player, opponent, field):
        field.get_regal_target_dict(player, opponent)
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field, player_num=player.player_num)
        mark_node = None
        starting_node.is_root = True
        self.starting_node = starting_node
        self.current_node = starting_node
        end_flg = False
        self.decide_node_seq = []
        self.decide_node_seq.append(starting_node)
        if starting_node.get_able_action_list() == [(0, 0, 0)]:
            return 0, 0, 0
        t1 = time.time()
        while time.time() - t1 < self.limit:

            node = self.tree_policy(starting_node, player_num=player_num)
            if node.field.players[1 - player_num].life <= 0 and node.field.players[player_num].life > 0:
                end_flg = True
                mark_node = node
                break
            value = self.default_policy(node, player_num=player_num)
            self.back_up(node, value, player_num=player_num)

        if end_flg:
            while True:
                if mark_node.parent_node is None:
                    break
                for child in mark_node.parent_node.child_nodes:
                    if child[1] == mark_node:
                        self.action_seq.append(child[0])
                        break
                mark_node = mark_node.parent_node
            self.action_seq = self.action_seq[::-1]

        self.initial_seq = self.action_seq[:]
        _, move = self.best(self.current_node, player_num=player_num)

        assert self.starting_node is not None and self.current_node is not None, "{},{}".format(self.starting_node,
                                                                                                self.current_node)
        return move


class Aggro_Shallow_MCTSPolicy(Shallow_MCTSPolicy):
    def __init__(self, th=10):
        super().__init__(th=th)
        self.play_out_policy = AggroPolicy()
        self.name = self.name = "Aggro_Shallow(th={})_MCTSPolicy".format(th)


class Expanded_Aggro_MCTS_Policy(Aggro_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Expanded_Aggro_MCTS_Policy"
        self.parameters = {
            0: 2, 1: 20, 2: 1, 3: 10, 4: 5, 5: 50
            # Sword_basic_vs_Sword_basic
        }

    def state_value(self, field, player_num):
        partial_observable_data = field.get_observable_data(player_num=player_num)
        if partial_observable_data["opponent"]["life"] <= 0:
            return WIN_BONUS
        able_attack_power_sum = 0
        if field.get_ward_list(player_num) == []:
            for card in field.card_location[player_num]:
                if card.card_category == "Creatrue" and card.can_attack_to_player():
                    able_attack_power_sum += card.power
            if able_attack_power_sum >= partial_observable_data["opponent"]["life"]:
                return WIN_BONUS
        opponent_power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            opponent_power_sum += field.card_location[1 - player_num][card_id].power
        if opponent_power_sum >= partial_observable_data["player"]["life"]:
            return -WIN_BONUS

        life_ad = partial_observable_data["player"]["life"] * self.parameters[0] \
                  - partial_observable_data["opponent"]["life"] * self.parameters[1]

        hand_ad = partial_observable_data["player"]["hand_len"] * self.parameters[2] \
                  - partial_observable_data["opponent"]["hand_len"] * self.parameters[3]

        creature_location = field.get_creature_location()
        board_ad = len(creature_location[player_num]) * self.parameters[4] \
                   - len(creature_location[1 - player_num]) * self.parameters[5]

        return life_ad + hand_ad + board_ad


class Genetic_GreedyPolicy(GreedyPolicy):
    def __init__(self, N=10):
        self.name = "Genetic_Greedy(N={})Policy".format(N)
        self.policy_type = 4
        self.epic_num = 0
        self.population = None
        self.population_num = N
        self.parameters = {}
        self.parameters_range = {0: 100, 1: 100, 2: 20, 3: 100, 4: 100, 5: 100}
        for i in range(6):
            self.parameters[i] = random.randint(0, self.parameters_range[i])
        self.fitness = {}
        self.better_parameters = deque()

    def set_fitness(self, win_late):
        input_key = (self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3], self.parameters[4],
                     self.parameters[5])
        self.fitness[input_key] = win_late
        if len(self.better_parameters) < self.population_num:
            self.better_parameters.append((win_late, input_key))
        elif min(self.better_parameters)[0] < win_late:
            update_flg = True
            for cell in self.better_parameters:
                if cell[1] == input_key:
                    update_flg = False
            if update_flg:
                self.better_parameters.append((win_late, input_key))
                self.better_parameters.remove(min(self.better_parameters))
        if len(self.better_parameters) < self.population_num:
            for i in range(6):
                self.parameters[i] = random.randint(0, self.parameters_range[i])
        else:
            if random.random() < 0.5:
                best = max(self.better_parameters)[1]
                for j in range(6):
                    self.parameters[j] = best[j]
                    if random.random() < 0.05:
                        self.parameters[j] = random.randint(0, self.parameters_range[j])
                same_flg = True
                for k in range(6):
                    if self.parameters[k] != best[k]:
                        same_flg = False
                        break
                if same_flg:
                    change_num = random.randint(1, 6)
                    change_ids = random.sample(list(range(6)), k=change_num)
                    for change_id in change_ids:
                        self.parameters[change_id] = random.randint(0, self.parameters_range[change_id])
            else:
                crossover = random.sample(self.better_parameters, k=2)
                choice_parameter_id = random.sample(list(range(6)), k=3)
                for j in range(6):
                    if j in choice_parameter_id:
                        self.parameters[j] = crossover[0][1][j]
                    else:
                        self.parameters[j] = crossover[1][1][j]
        self.epic_num += 1
        if self.epic_num % self.population_num == 0:
            mylogger.info("populations:{}".format(self.better_parameters))

    def state_value(self, field, player_num):
        partial_observable_data = field.get_observable_data(player_num=player_num)
        if partial_observable_data["opponent"]["life"] <= 0:
            return WIN_BONUS
        estimate_term = self.parameters[0]
        able_attack_power_sum = 0
        if field.get_ward_list(player_num) == []:
            for card in field.card_location[player_num]:
                if card.card_category == "Creatrue" and card.can_attack_to_player():
                    able_attack_power_sum += card.power
            if able_attack_power_sum >= partial_observable_data["opponent"]["life"]:
                return WIN_BONUS
            elif partial_observable_data["opponent"]["leader_class"] == "BLOOD":
                estimate_term = self.parameters[1]
        opponent_power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            opponent_power_sum += field.card_location[1 - player_num][card_id].power
        if opponent_power_sum >= partial_observable_data["player"]["life"]:
            return -WIN_BONUS

        life_ad = estimate_term * \
                  (partial_observable_data["opponent"]["max_life"] - partial_observable_data["opponent"]["life"])
        low_life_term = 1 + 10 * int(partial_observable_data["player"]["life"] <= self.parameters[2])
        board_ad = (len(field.get_creature_location()[player_num]) - len(
            field.get_creature_location()[1 - player_num])) * low_life_term
        return life_ad * self.parameters[3] + \
               board_ad * self.parameters[4] + partial_observable_data["player"]["hand_len"] * self.parameters[5]


class Genetic_New_GreedyPolicy(Genetic_GreedyPolicy):
    def __init__(self, N=10):
        self.name = "Genetic_New_Greedy(N={})Policy".format(N)
        self.policy_type = 4
        self.epic_num = 0
        self.population = None
        self.population_num = N
        self.parameters = {}
        self.parameters_range = {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
        for i in range(6):
            self.parameters[i] = random.randint(0, self.parameters_range[i])
        self.fitness = {}
        self.better_parameters = deque()

    def state_value(self, field, player_num):
        partial_observable_data = field.get_observable_data(player_num=player_num)
        if partial_observable_data["opponent"]["life"] <= 0:
            return WIN_BONUS
        able_attack_power_sum = 0
        if field.get_ward_list(player_num) == []:
            for card in field.card_location[player_num]:
                if card.card_category == "Creatrue" and card.can_attack_to_player():
                    able_attack_power_sum += card.power
            if able_attack_power_sum >= partial_observable_data["opponent"]["life"]:
                return WIN_BONUS
        opponent_power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            opponent_power_sum += field.card_location[1 - player_num][card_id].power
        if opponent_power_sum >= partial_observable_data["player"]["life"]:
            return -WIN_BONUS

        life_ad = partial_observable_data["player"]["life"] * self.parameters[0] \
                  - partial_observable_data["opponent"]["life"] * self.parameters[1]

        hand_ad = partial_observable_data["player"]["hand_len"] * self.parameters[2] \
                  - partial_observable_data["opponent"]["hand_len"] * self.parameters[3]

        creature_location = field.get_creature_location()
        board_ad = len(creature_location[player_num]) * self.parameters[4] \
                   - len(creature_location[1 - player_num]) * self.parameters[5]

        return life_ad + hand_ad + board_ad


class Genetic_Aggro_MCTSPolicy(Aggro_MCTSPolicy):
    def __init__(self, N=10):
        super().__init__()
        self.name = "Genetic_Aggro_MCTS_Policy(N={})".format(N)
        self.epic_num = 0
        self.population = None
        self.population_num = N
        self.parameters = {}
        self.parameters_range = {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
        for i in range(6):
            self.parameters[i] = random.randint(0, self.parameters_range[i])
        self.fitness = {}
        self.better_parameters = deque()

    def set_fitness(self, win_late):
        # input_key = (self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3],
        # self.parameters[4],self.parameters[5])
        input_key = tuple(self.parameters.values())
        self.fitness[input_key] = win_late
        if len(self.better_parameters) < self.population_num:
            self.better_parameters.append((win_late, input_key))
        elif min(self.better_parameters)[0] < win_late:
            update_flg = True
            for cell in self.better_parameters:
                if cell[1] == input_key:
                    update_flg = False
            if update_flg:
                self.better_parameters.append((win_late, input_key))
                self.better_parameters.remove(min(self.better_parameters))
        if len(self.better_parameters) < self.population_num:
            for i in range(6):
                self.parameters[i] = random.randint(0, self.parameters_range[i])
        else:
            if random.random() < 0.5:
                best = max(self.better_parameters)[1]
                for j in range(6):
                    self.parameters[j] = best[j]
                    if random.random() < 0.05:
                        self.parameters[j] = random.randint(0, self.parameters_range[j])
                same_flg = True
                for k in range(6):
                    if self.parameters[k] != best[k]:
                        same_flg = False
                        break
                if same_flg:
                    change_num = random.randint(1, 6)
                    change_ids = random.sample(list(range(6)), k=change_num)
                    for change_id in change_ids:
                        self.parameters[change_id] = random.randint(0, self.parameters_range[change_id])
            else:
                crossover = random.sample(self.better_parameters, k=2)
                choice_parameter_id = random.sample(list(range(6)), k=3)
                for j in range(6):
                    if j in choice_parameter_id:
                        self.parameters[j] = crossover[0][1][j]
                    else:
                        self.parameters[j] = crossover[1][1][j]
        self.epic_num += 1
        if self.epic_num % self.population_num == 0:
            mylogger.info("better_parameters")
            for cell in sorted(list(self.better_parameters), key=lambda ele: -ele[0]):
                mylogger.info("{:%}:{}".format(cell[0], cell[1]))

    def state_value(self, field, player_num):
        partial_observable_data = field.get_observable_data(player_num=player_num)
        if partial_observable_data["opponent"]["life"] <= 0:
            return WIN_BONUS
        able_attack_power_sum = 0
        if field.get_ward_list(player_num) == []:
            for card in field.card_location[player_num]:
                if card.card_category == "Creatrue" and card.can_attack_to_player():
                    able_attack_power_sum += card.power
            if able_attack_power_sum >= partial_observable_data["opponent"]["life"]:
                return WIN_BONUS
        opponent_power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            opponent_power_sum += field.card_location[1 - player_num][card_id].power
        if opponent_power_sum >= partial_observable_data["player"]["life"]:
            return -WIN_BONUS

        life_ad = partial_observable_data["player"]["life"] * self.parameters[0] \
                  - partial_observable_data["opponent"]["life"] * self.parameters[1]

        hand_ad = partial_observable_data["player"]["hand_len"] * self.parameters[2] \
                  - partial_observable_data["opponent"]["hand_len"] * self.parameters[3]

        creature_location = field.get_creature_location()
        board_ad = len(creature_location[player_num]) * self.parameters[4] \
                   - len(creature_location[1 - player_num]) * self.parameters[5]

        return life_ad + hand_ad + board_ad


class New_Node:

    def __init__(self, field=None, player_num=0, finite_state_flg=False, depth=0,root=False):
        self.field = field
        self.finite_state_flg = finite_state_flg
        self.value = 0.0
        self.state_value = None
        self.pai = [1/ACTION_SIZE]*ACTION_SIZE
        self.visit_num = 0
        self.is_root = root
        self.uct = 0.0
        self.depth = depth
        self.child_nodes = []
        self.child_actions = []
        self.max_child_visit_num = [None, None]
        self.parent_node = None
        self.regal_targets = {}
        self.action_value_dict = {}
        self.current_probability = None
        self.children_moves = [(0, 0, 0)]
        self.node_id_2_edge_action = {}
        self.edge_action_2_node_id = {}
        self.action_counter = {}
        self.action_visit_num_dict = {}
        self.player_num = player_num

        children_moves = []
        end_flg = field.check_game_end()

        if not end_flg:
            player = field.players[player_num]
            #player.sort_hand()
            field.update_hand_cost(player_num=player_num)
            (ward_list, _, can_be_attacked, regal_targets) = \
                field.get_situation(player, field.players[1 - player_num])

            (_, _, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) = \
                field.get_flag_and_choices(player, field.players[1 - player_num], regal_targets)
            self.regal_targets = regal_targets
            if root and not field.secret:
                mylogger.info("root player hand")
                player.show_hand()
                mylogger.info("able_to_play:{}".format(able_to_play))
            children_moves.append((0, 0, 0))
            remain_able_to_play = able_to_play[:]
            hand_id = 0
            while hand_id < len(player.hand):
                if hand_id in remain_able_to_play:
                    # mylogger.info("remain:{}".format(remain_able_to_play))
                    other_hand_ids = remain_able_to_play[:]
                    other_hand_ids.remove(hand_id)
                    # mylogger.info("other_hand_ids:{}".format(other_hand_ids))
                    if other_hand_ids is not None:
                        for other_id in other_hand_ids:
                            if player.hand[hand_id].eq(player.hand[other_id]):
                                remain_able_to_play.remove(other_id)
                hand_id += 1
            able_to_play = remain_able_to_play

            remain_able_to_creature_attack = able_to_creature_attack[:]
            remain_able_to_attack = able_to_attack[:]
            remain_able_to_evo = able_to_evo[:]
            location_id = 0
            side = field.card_location[player_num]

            while location_id < len(side):
                if location_id in remain_able_to_creature_attack:
                    other_follower_ids = remain_able_to_creature_attack[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_creature_attack.remove(other_id)
                if location_id in remain_able_to_attack:
                    other_follower_ids = remain_able_to_attack[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_attack.remove(other_id)
                if location_id in remain_able_to_evo:
                    other_follower_ids = remain_able_to_evo[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_evo.remove(other_id)
                location_id += 1
            able_to_creature_attack = remain_able_to_creature_attack
            able_to_attack = remain_able_to_attack
            able_to_evo = remain_able_to_evo
            remain_can_be_attacked = can_be_attacked[:]
            remain_ward_list = ward_list[:]
            side = field.card_location[1 - player_num]
            location_id = 0
            while location_id < len(side):
                if location_id in remain_can_be_attacked:
                    other_follower_ids = remain_can_be_attacked[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_can_be_attacked.remove(other_id)
                            # elif side[location_id].name == side[other_id].name:
                            #    mylogger.info("{}".format(side[location_id]))
                            #    mylogger.info("{}".format(side[other_id]))
                if location_id in remain_ward_list:
                    other_follower_ids = remain_ward_list[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_ward_list.remove(other_id)
                location_id += 1
            can_be_attacked = remain_can_be_attacked
            ward_list = remain_ward_list
            if root and not field.secret:
                mylogger.info("remain_able_to_play:{}".format(able_to_play))
            for play_id in able_to_play:
                #if field.players[player_num].hand[play_id].cost > field.remain_cost[player_num]:
                #    continue
                if field.players[player_num].hand[play_id].card_category != "Spell" and len(
                        field.card_location[player_num]) >= field.max_field_num:
                    continue
                if len(regal_targets[play_id]) > 0:
                    for i in range(len(regal_targets[play_id])):
                        children_moves.append((Action_Code.PLAY_CARD.value, play_id, regal_targets[play_id][i]))
                else:
                    if field.players[player_num].hand[play_id].card_category == "Spell":
                        if field.players[player_num].hand[play_id].have_target == 0:
                            children_moves.append((Action_Code.PLAY_CARD.value, play_id, None))
                    else:
                        children_moves.append((Action_Code.PLAY_CARD.value, play_id, None))

            if len(ward_list) == 0:
                for attacker_id in able_to_creature_attack:
                    for target_id in can_be_attacked:
                        children_moves.append((Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id))
                for attacker_id in able_to_attack:
                    children_moves.append((Action_Code.ATTACK_TO_PLAYER.value, attacker_id, None))
            else:
                for attacker_id in able_to_creature_attack:
                    for target_id in ward_list:
                        children_moves.append((Action_Code.ATTACK_TO_FOLLOWER.value, attacker_id, target_id))

            if can_evo:
                for evo_id in able_to_evo:
                    evo_creature = field.card_location[player_num][evo_id]
                    if evo_creature.evo_target is None:
                        children_moves.append((Action_Code.EVOLVE.value, evo_id, None))
                    else:
                        targets = field.get_regal_targets(evo_creature, target_type=0, player_num=player_num)
                        for target_id in targets:
                            children_moves.append((Action_Code.EVOLVE.value, evo_id, target_id))

            self.children_moves = children_moves

    def print_tree(self, single=False):
        if self.is_root:
            print("ROOT(id:{})".format(id(self)))
            mylogger.info("\nable_moves:{}\nchild_moves:{}".format(self.children_moves, self.child_actions))
        # if self.parent_node!=None:
        #    print("   "*self.depth+("parent_id:{})".format(id(self.parent_node))))

        print("   " * self.depth + "depth:{} finite:{} mean_value:{:.3f} visit_num:{} player_num:{}".format(self.depth,
                                                                                                            self.finite_state_flg,
                                                                                                            self.value / max(
                                                                                                                1,
                                                                                                                self.visit_num),
                                                                                                            self.visit_num,
                                                                                                            self.player_num))

        if self.child_nodes != []:
            layer_height = int(not single)*self.depth
            print("   " * layer_height + "child_node_num:{}".format(len(self.child_nodes)))
            print("   " * layer_height + "child_node_set:{")
            for action_key in list(self.edge_action_2_node_id.keys()):
                for child in self.child_nodes:
                    # if id(child) not in self.edge_action_2_node_id[action_key]:
                    if child not in self.edge_action_2_node_id[action_key]:
                        continue
                    action_code = self.node_id_2_edge_action[id(child)]
                    action_name = Action_Code(action_code[0]).name
                    txt = ""
                    if action_code[0] == Action_Code.PLAY_CARD.value:
                        play_card_name = self.field.players[self.player_num].hand[action_code[1]].name
                        txt = "{},{}({}),{}".format(action_name,
                                                    play_card_name, action_code[1], action_code[2])
                    elif action_code[0] == Action_Code.ATTACK_TO_FOLLOWER.value:
                        attacker_name = self.field.card_location[self.player_num][action_code[1]].name
                        defencer_name = self.field.card_location[1 - self.player_num][action_code[2]].name
                        txt = "{},{}({}),{}({})".format(
                            action_name, attacker_name, action_code[1], defencer_name, action_code[2])
                    elif action_code[0] == Action_Code.ATTACK_TO_PLAYER.value:
                        attacker_name = self.field.card_location[self.player_num][action_code[1]].name
                        txt = "{},{}({})".format(action_name, attacker_name, action_code[1])
                    elif action_code[0] == Action_Code.EVOLVE.value:
                        evo_name = self.field.card_location[self.player_num][action_code[1]].name
                        txt = "{},{}({}),{}".format(action_name, evo_name, action_code[1], action_code[2])
                    else:
                        txt = "{}".format(action_name)

                    print("   " * layer_height + "action:{}".format(txt))
                    if action_code[0] == Action_Code.TURN_END.value:
                        print("   " * layer_height + "able_moves_num:{} pp/max_pp:{}/{}" \
                              .format(len(self.child_actions), \
                                      self.field.remain_cost[self.player_num], self.field.cost[self.player_num]))
                        # self.field.players[self.player_num].show_hand()
                    print(
                        "   " * layer_height + "ave_value:{} visit_num:{}".format(child.value / max(1, child.visit_num),
                                                                                  child.visit_num))
                    #assert abs(child.value / max(1, child.visit_num)) <= 1.0,"{},{}".format(child.value,child.visit_num)
                    if not single:
                        child.print_tree()
            print("   " * layer_height + "}")


class Information_Set_MCTSPolicy():
    def __init__(self, iteration=100):
        self.num_simulations = 0
        self.uct_c = 1. / np.sqrt(2)
        self.play_out_policy = AggroPolicy()  # RandomPolicy()
        self.sub_play_out_policy = RandomPolicy()
        self.probability = 0.00
        self.starting_node = None
        self.current_node = None
        self.prev_node = None
        self.node_index = 0
        self.policy_type = 4
        self.name = "ISMCTS(n={})Policy".format(iteration)
        self.type = "root"
        self.iteration = iteration
        self.default_iteration = 10

    def state_value(self, field, player_num):
        return default_state_value(field,player_num)

    def decide(self, player, opponent, field):
        if self.current_node is None:
            if not field.secret:
                mylogger.info("generate new tree")
            action = self.uct_search(player, opponent, field)
            if not field.secret:
                self.current_node.print_tree(single=True)
            tmp_move = action
            if len(self.current_node.child_nodes) > 0:
                next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
                self.prev_node = self.current_node
                self.current_node = next_node
            self.node_index = 0
            self.type = "root"
            if tmp_move == (0, 0, 0):
                self.current_node = None
            return tmp_move  # action
        else:

            hit_flg = False
            self.type = "blanch"
            for child in self.prev_node.child_nodes:
                if child.field.eq(field):
                    hit_flg = True
                    self.current_node = child
                    break

            if not hit_flg:
                if not field.secret:
                    mylogger.info("not exist in simulation child_num:{}"
                                  .format(len(self.prev_node.child_nodes)))
                return Action_Code.ERROR.value, "not exist in simulation child_num:{}"\
                                  .format(len(self.prev_node.child_nodes)), 0


            self.node_index += 1
            if len(self.current_node.child_nodes) == 0:
                if not field.secret:
                    mylogger.info("no child in this node")
                return Action_Code.ERROR.value, "no child in this node", 0
            if not field.secret:
                mylogger.info("use existed tree")
            next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
            self.prev_node = self.current_node
            if tmp_move[0] == Action_Code.TURN_END.value:
                self.current_node = None
            return tmp_move

    def uct_search(self, player, opponent, field,use_existed_node=False):
        field.get_regal_target_dict(player, opponent)
        player_num = player.player_num
        starting_node = None
        if use_existed_node:
            starting_node = self.current_node
            starting_node.parent_node = None
            starting_node.is_root = True
            #mylogger.info(starting_node.children_moves)
        else:
            starting_field = Field_setting.Field(5)
            starting_field.set_data(field)
            starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                                 starting_field.players[opponent.player_num])
            starting_node = New_Node(field=starting_field, player_num=player.player_num,root=True)
            starting_node.is_root = True
        self.starting_node = starting_node
        self.current_node = starting_node
        if starting_node.children_moves == [(0, 0, 0)]:
            return 0, 0, 0
        for i in range(self.iteration):
            if time.time() - field.time > 89:
                break
            node = self.tree_policy(starting_node, player_num=player_num)
            value = self.default_policy(node, player_num=player_num)
            self.back_up(node, value, player_num=player_num)
            if starting_node.max_child_visit_num[0] is not None \
                    and starting_node.max_child_visit_num[1] is not None:
                if starting_node.max_child_visit_num[1].visit_num - \
                        starting_node.max_child_visit_num[0].visit_num > self.iteration - i:
                    break

            elif starting_node.max_child_visit_num[1] is not None:
                if starting_node.max_child_visit_num[1].visit_num > int(self.iteration / 2):
                    break
        _, move = self.execute_best(self.current_node, player_num=player_num)

        return move

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and not check:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    parent_node = node
                    node, move = self.best(node, player_num=player_num)
                    exist_flg = False
                    for child in parent_node.child_nodes:
                        if move == parent_node.node_id_2_edge_action[id(child)]:
                            if child.field.eq(node.field):
                                exist_flg = True
                                break
                    if not exist_flg:
                        # mylogger.info("new_node")
                        return self.expand(node, player_num=player_num)

                else:
                    check = self.fully_expand(node, player_num=player_num)
                    if not check:
                        return self.expand(node, player_num=player_num)
                    else:
                        parent_node = node
                        node, move = self.best(node, player_num=player_num)
                        exist_flg = False
                        for child in parent_node.child_nodes:
                            if move == parent_node.node_id_2_edge_action[id(child)]:
                                if child.field.eq(node.field):
                                    exist_flg = True
                                    break
                        if not exist_flg:
                            # mylogger.info("new_node")
                            return self.expand(node, player_num=player_num)
                length_of_children = len(node.child_nodes)
            else:
                return self.expand(node, player_num=player_num)

            count += 1
            if count > 100:
                field = node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            if not node.finite_state_flg:
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)

                current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                player = current_field.players[player_num]
                opponent = current_field.players[1 - player_num]
                current_field.get_regal_target_dict(player, opponent)

                action_count = 0
                while True:
                    if random.random() < self.probability:
                        (action_num, card_id, target_id) = self.sub_play_out_policy.decide(player, opponent,
                                                                                           current_field)
                    else:
                        (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

                    end_flg = player.execute_action(current_field, opponent,
                                                    action_code=(action_num, card_id, target_id), virtual=True)

                    if current_field.check_game_end() or end_flg:
                        break

                    current_field.get_regal_target_dict(player, opponent)
                    action_count += 1
                    if action_count > 100:
                        player.show_hand()
                        current_field.show_field()
                        assert False
                if current_field.check_game_end():
                    sum_of_value += 1  # WIN_BONUS
                    return sum_of_value
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end():
                    sum_of_value += 1  # WIN_BONUS
                    return sum_of_value
                else:
                    sum_of_value += self.state_value(current_field, player_num)

        return sum_of_value / self.default_iteration

    def fully_expand(self, node, player_num=0):
        remain_actions = list(set(node.children_moves) - set(node.child_actions))
        if remain_actions != []:
            remain_num = len(remain_actions)
            player = node.field.players[player_num]
            for action in remain_actions:
                if action[0] == Action_Code.TURN_END.value:
                    check = action in node.edge_action_2_node_id
                    if check:
                        remain_num -= 1

                if remain_num <= 0:
                    return True
                if action[0] == Action_Code.EVOLVE.value:
                    evo_actions = list(filter(lambda cell: cell[0] == -1, node.child_actions))
                    side = node.field.card_location[player_num]
                    opponent_side = node.field.card_location[1 - player_num]
                    for evo_action in evo_actions:
                        if side[action[1]].eq(side[evo_action[1]]):
                            if action[2] is None:
                                # return True
                                node.children_moves.remove(action)
                                remain_num -= 1
                                break
                            else:
                                if side[action[1]].evo_target == Target_Type.ENEMY_FOLLOWER.value:
                                    if opponent_side[action[2]].eq(opponent_side[evo_action[2]]):
                                        node.children_moves.remove(action)
                                        remain_num -= 1
                                        break
                                elif side[action[1]].name == "Maisy, Red Riding Hood":
                                    target_num = side[action[1]].evo_target
                                    assert False, "target_type:{}".format(target_num, Target_Type(target_num))

                if remain_num <= 0:
                    return True
                if action[0] == Action_Code.PLAY_CARD.value:
                    play_actions = list(filter(lambda cell: cell[0] == 1, node.child_actions))
                    for play_action in play_actions:
                        if player.hand[action[1]].eq(player.hand[play_action[1]]):
                            # return True
                            node.children_moves.remove(action)
                            remain_num -= 1
                            break
                if remain_num <= 0:
                    return True
                if action[0] == Action_Code.ATTACK_TO_FOLLOWER.value:
                    attack_actions = list(filter(lambda cell: cell[0] == 2, node.child_actions))
                    field = node.field
                    side = field.card_location[player_num]
                    opponent_side = field.card_location[1 - player_num]
                    length = len(opponent_side)
                    assert action[2] < length, "out-of-range(action):{},{}".format(action[2], length)
                    for attack_action in attack_actions:
                        assert attack_action[2] < length, "out-of-range(attack_action):{},{}".format(action[2], length)
                        if side[action[1]].power == 0 and KeywordAbility.BANE.value not in side[action[1]].ability:
                            # return True
                            node.children_moves.remove(action)
                            remain_num -= 1
                            break
                        if side[action[1]].eq(side[attack_action[1]]):
                            if opponent_side[action[2]].eq(opponent_side[attack_action[2]]):
                                # return True
                                node.children_moves.remove(action)
                                remain_num -= 1
                                break
                if remain_num <= 0:
                    return True
            return False
        return True

        # return node.finite_state_flg  # turn_endの場合を追加

    def expand(self, node, player_num=0):
        child_node_fields = []
        for cell in node.child_nodes:
            child_node_fields.append(cell.field)
        old_choices = node.children_moves[:]
        new_choices = node.children_moves[:]
        if (0,0,0) in node.edge_action_2_node_id:
            old_choices.remove((0,0,0))
            new_choices.remove((0, 0, 0))
        assert len(new_choices) > 0, "non-choice-error"
        next_node, move, exist_flg = self.execute_single_action(node, new_choices, child_node_fields,
                                                                player_num=player_num)
        if exist_flg:
            while exist_flg:
                new_choices = list(set(node.children_moves) - set(node.child_actions))
                # assert len(new_choices) > 0, "non-choice-error({},{})".format(node.children_moves,node.child_actions)
                if len(new_choices) == 0:
                    return self.best(node, player_num)[0]
                next_node, move, exist_flg = self.execute_single_action(node, new_choices, child_node_fields,
                                                                        player_num=player_num)

        node.node_id_2_edge_action[id(next_node)] = move
        next_node.parent_node = node
        node.child_nodes.append(next_node)
        if move in node.edge_action_2_node_id:
            node.edge_action_2_node_id[move].append(next_node)
            # node.edge_action_2_node_id[move].append(id(next_node))
        else:
            node.edge_action_2_node_id[move] = [next_node]
            # node.edge_action_2_node_id[move] = [id(next_node)]
        if move not in node.child_actions:
            node.child_actions.append(move)
            assert move in node.children_moves, "ill-move!"
        return next_node

    def execute_single_action(self, node, new_choices, child_node_fields, player_num=0):
        field = node.field
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.players[0].deck.shuffle()
        next_field.players[1].deck.shuffle()
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])
        next_node = None
        exist_flg = False
        if move[0] == Action_Code.TURN_END.value:
            next_node = New_Node(field=next_field, player_num=player_num, finite_state_flg=True,
                                 depth=node.depth + 1)
            # exist_flg = next_field in child_node_fields
            for child in node.child_nodes:
                if node.node_id_2_edge_action[id(child)] == Action_Code.TURN_END.value:
                    exist_flg = True
                    break

        else:
            if move[0] == Action_Code.PLAY_CARD.value:
                if move[2] is None and node.regal_targets[move[1]] != []:
                    mylogger.info("in_node:{}".format(node.regal_targets[move[1]]))
                    assert False, "null target error"
                elif move[2] is not None and \
                        move[2] not in next_field.get_regal_targets(next_field.players[player_num].hand[move[1]],
                                                                    target_type=1, player_num=player_num):
                    assert False, "ill-target error"
            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = next_field.check_game_end()
            next_node = New_Node(field=next_field, player_num=player_num,
                                 finite_state_flg=flg, depth=node.depth + 1)
            # exist_flg = next_field in child_node_fields
            for child_field in child_node_fields:
                if child_field.eq(next_field) and child_field.players[player_num].eq(next_field.players[player_num]):
                    node.children_moves.remove(move)
                    exist_flg = True
                    break

        return next_node, move, exist_flg

    def best(self, node, player_num=0):
        children = node.child_nodes
        # uct_values = {}
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            uct = self.uct(children[i], node, player_num=player_num)
            if action == (0,0,0):
                uct = 0
            action_uct_values[action].append(uct)
            action_2_node[action].append(children[i])
        max_value = None
        max_value_action = None
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            weights = [cell.visit_num for cell in action_2_node[key]]
            visit_num_sum = sum(weights)
            # mean_value = sum(values)/len(values)
            mean_value = sum([values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            if max_value is None or mean_value > max_value:
                max_value = mean_value
                max_value_action = key
        weights = [cell.visit_num for cell in action_2_node[max_value_action]]
        max_value_node = random.choices(action_2_node[max_value_action], weights=weights)[0]
        #if max_value_action not in node.action_visit_num_dict:
        #    node.action_visit_num_dict[max_value_action]=0
        #    node.action_visit_num_dict[max_value_action] += 1
        return max_value_node, max_value_action

    def execute_best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            # action_uct_values[action].append(children[i].value)
            action_uct_values[action].append(children[i].visit_num)
            action_2_node[action].append(children[i])
        max_value = None
        max_value_action = None
        max_ave_value = 0
        assert len(action_uct_values) > 0, "non-action-error"
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            weights = [cell.visit_num for cell in action_2_node[key]]
            ave_values_list = [cell.value/cell.visit_num for cell in action_2_node[key]]

            visit_num_sum = sum(weights)
            # mean_value = sum(values)/len(values)
            mean_value = sum([values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            ave_value = sum([values[node_id] * (ave_values_list[node_id] / visit_num_sum) for node_id in range(len(values))])
            #mylogger.info("action:{}".format(key))
            #mylogger.info("weighted_visit_num:{} ave_value:{}".format(mean_value,ave_value))
            #mylogger.info("")
            if max_value is None or mean_value > max_value:
                max_value = mean_value
                max_value_action = key
                max_ave_value = max_ave_value
            elif mean_value == max_value:
                if ave_value > max_ave_value:
                    max_value_action = key
                    max_ave_value = max_ave_value




        weights = [cell.visit_num for cell in action_2_node[max_value_action]]
        max_value_node = random.choices(action_2_node[max_value_action], weights=weights)[0]
        return max_value_node, max_value_action

    def uct(self, child_node, node, player_num=0):
        over_all_n = node.visit_num
        n = child_node.visit_num
        w = child_node.value
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        # mylogger.info("exploitation_value:{}".format(exploitation_value))
        if over_all_n == 0:
            over_all_n = 1
        if n == 0:
            mylogger.warning("zero child visit_num!")
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / (n + epsilon))

        value = exploitation_value + exploration_value
        # mylogger.info("value:{}".format(value))
        child_node.uct = value

        return value

    def back_up(self, last_visited, reward, player_num=0):
        current = last_visited
        while True:
            #prev_value = float(current.value)
            current.visit_num += 1
            current.value += reward
            #assert abs(reward) <= 1.0,"reward:{}".format(reward)
            #assert abs(current.value/current.visit_num) <= 1.0,"{} to {},{}".format(prev_value,current.value,current.visit_num)
            if current.is_root:
                break
            if current.parent_node is None:
                current.print_tree(single=True)
                assert False
            if current.parent_node.is_root:
                best_childen = current.parent_node.max_child_visit_num
                best_node = best_childen[1]
                second_node = best_childen[0]
                if best_node is None:  # ベストが空
                    best_node = current
                elif second_node is None:  # ベストが1つのみ
                    if current != best_node:
                        if best_node.visit_num < current.visit_num:
                            best_childen = [best_node, current]
                        else:
                            best_childen = [current, best_node]
                else:
                    if second_node.visit_num > best_node.visit_num:
                        best_children = [best_node, second_node]

                    different_flg = current not in best_children[0]
                    if different_flg:
                        if current.visit_num > best_children[1].visit_num:
                            best_children = [best_children[1], current]

                        elif current.visit_num > best_children[0].visit_num:
                            best_children = [current, best_children[1]]

            current = current.parent_node
            if current is None:
                break


class Flexible_Iteration_MCTSPolicy(MCTSPolicy):
    def __init__(self, N=100):
        super().__init__()
        self.name = self.name = "MCTS(n={})Policy".format(N)
        self.iteration = N


class Flexible_Iteration_Aggro_MCTSPolicy(Aggro_MCTSPolicy):
    def __init__(self, N=100):
        super().__init__()
        self.name = self.name = "Aggro(n={})_MCTSPolicy".format(N)
        self.iteration = N


class Flexible_Iteration_Information_Set_MCTSPolicy(Information_Set_MCTSPolicy):
    def __init__(self, N=100):
        super().__init__()
        self.name = self.name = "IS(n={})_MCTSPolicy".format(N)
        self.iteration = N


class Time_bounded_Information_Set_MCTSPolicy(Information_Set_MCTSPolicy):
    def __init__(self, limit=90, playout=RandomPolicy()):
        super().__init__()
        self.name = "Time_bounded(limit={})_Information_Set_MCTSPolicy".format(limit)
        self.limit = limit

    def uct_search(self, player, opponent, field):
        field.get_regal_target_dict(player, opponent)
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = New_Node(field=starting_field, player_num=player.player_num)
        starting_node.is_root = True
        self.starting_node = starting_node
        self.current_node = starting_node
        if starting_node.children_moves == [(0, 0, 0)]:
            return 0, 0, 0
        t1 = time.time()
        while time.time() - t1 < self.limit:
            node = self.tree_policy(starting_node, player_num=player_num)
            value = self.default_policy(node, player_num=player_num)
            self.back_up(node, value, player_num=player_num)

        _, move = self.execute_best(self.current_node, player_num=player_num)

        return move


def default_state_value(field, player_num):
    return advanced_state_value(field,player_num)
    """
    if field.check_game_end():
        if field.players[player_num].life <= 0 or field.players[player_num].lib_out_flg:
            return 0
        else:
            return 1
    power_sum = 0
    for card_id in field.get_creature_location()[1 - player_num]:
        power_sum += field.card_location[1 - player_num][card_id].power
    if power_sum >= field.players[player_num].life and not field.check_ward()[player_num]:
        return 0
    cum_damage = (field.players[1 - player_num].max_life - field.players[1 - player_num].life) * 50
    board_ad = (len(field.card_location[player_num]) - len(
        field.card_location[1 - player_num])) * 50
    hand_len = len(field.players[player_num].hand)
    value = cum_damage + board_ad + hand_len + 250
    max_value = 1509  # 1000+250+9+250

    return value / max_value
    """


class Opponent_Modeling_MCTSPolicy(MCTSPolicy):
    # デッキ公開制のときのみ
    def __init__(self,iteration=100):
        super().__init__()
        self.name = "OM_D_MCTS(iteration={})Policy".format(iteration)
        self.main_player_num = 0
        self.play_out_policy = AggroPolicy()
        self.iteration = iteration

    def state_value(self, field, player_num):
        return default_state_value(field,player_num)

    def decide(self, player, opponent, field):
        self.main_player_num = player.player_num
        if self.current_node is None:
            action = self.uct_search(player, opponent, field)
            if not field.secret:
                mylogger.info("generate new tree")

            tmp_move = action
            if len(self.current_node.child_nodes) > 0:
                next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
                self.prev_node = self.current_node
                self.current_node = next_node
            if tmp_move[0] == Action_Code.TURN_END.value:
                self.current_node = None

            return tmp_move  # action
        else:
            self.type = "blanch"
            hit_flg = False
            for child_node in self.prev_node.child_nodes:
                sim_player_hand = child_node[1].field.players[player.player_num].hand
                if child_node[1].field.eq(field) and player.compare_hand(sim_player_hand):
                    hit_flg = True
                    self.current_node = child_node[1]
                    break

            if hit_flg:
                if not field.secret:
                    mylogger.info("corresponding node is not found(visit_num:{},child_num:{})"
                                  .format(self.current_node.visit_num,
                                          len(self.current_node.child_nodes)))
                self.uct_search(player, opponent, field,use_existed_node=True)

            else:
                return Action_Code.ERROR.value,None,None
            assert len(self.current_node.child_nodes) > 0,"current_node have no child!\n{}"\
                .format(self.current_node.print_tree(single=True))
            next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
            self.prev_node = self.current_node
            self.current_node = next_node
            if tmp_move == (0, 0, 0):
                if not field.secret:
                    if self.current_node.parent_node is not None:
                        if self.current_node.parent_node.children_moves != [(0,0,0)]:
                            mylogger.info("turn end is prior to other moves")
                            mylogger.info("children_moves:{}".format(self.current_node.parent_node.children_moves))
                            self.current_node.parent_node.print_tree(single=True)
                self.current_node = None
            return tmp_move  # action

    def uct_search(self, player, opponent, field,use_existed_node=False):
        field.get_regal_target_dict(player, opponent)
        player_num = player.player_num

        if use_existed_node:
            starting_node = self.current_node
            starting_node.parent_node = None
            starting_node.is_root = True
        else:
            starting_field = Field_setting.Field(5)
            starting_field.set_data(field)
            starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                                 starting_field.players[opponent.player_num])
            starting_node = Node(field=starting_field, player_num=player.player_num)
            starting_node.is_root = True
            self.starting_node = starting_node
        self.current_node = starting_node
        self.decide_node_seq = []
        self.decide_node_seq.append(starting_node)
        if starting_node.get_able_action_list() == [(0, 0, 0)]:
            return 0, 0, 0
        for i in range(self.iteration):
            if time.time() - field.time > 89:
                break
            node = self.tree_policy(starting_node, player_num=player_num)
            if node.field.players[1 - player_num].life <= 0 and node.field.players[player_num].life > 0:
                self.back_up(node, 1.0, player_num=player_num)
                break
            value = self.default_policy(node, player_num=player_num)
            self.back_up(node, value, player_num=player_num)

            if starting_node.max_child_visit_num[0] is not None and starting_node.max_child_visit_num[1] is not None:
                if starting_node.max_child_visit_num[1].visit_num - \
                        starting_node.max_child_visit_num[0].visit_num > self.iteration - i:
                    break

            elif starting_node.max_child_visit_num[1] is not None:
                if starting_node.max_child_visit_num[1].visit_num > int(self.iteration / 2):
                    break
        if len(self.starting_node.child_nodes) > 0:
            _, move = self.execute_best(self.starting_node, player_num=player_num)
        else:
            move = Action_Code.TURN_END.value, 0, 0

        assert self.starting_node is not None and self.current_node is not None, "{},{}".format(self.starting_node,
                                                                                                self.current_node)
        #if not field.secret:
        #    mylogger.info("new tree detail")
        #    self.starting_node.print_tree()
        return move

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and not check:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            current_player_num = node.field.turn_player_num
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    node, _ = self.best(node, player_num=current_player_num)
                else:
                    check = self.fully_expand(node, player_num=current_player_num)
                    if not check:
                        return self.expand(node, player_num=current_player_num)
                    else:
                        node, _ = self.best(node, player_num=current_player_num)
                length_of_children = len(node.child_nodes)
            else:
                check = self.fully_expand(node, player_num=current_player_num)
                if not check:
                    return self.expand(node, player_num=current_player_num)
                break

            count += 1
            if count > 100:
                field = node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node

    def default_policy(self, node, player_num=0):
        sum_of_value = 0
        if node.finite_state_flg:
            return self.state_value(node.field, player_num=self.main_player_num)
        end_count = 0
        current_field = Field_setting.Field(5)
        opponent_player_num = 1 - self.main_player_num
        for i in range(self.default_iteration):
            current_field.set_data(node.field)
            if self.main_player_num == player_num:
                current_field.turn_player_num = player_num
                self.simulate_playout(current_field, player_num=self.main_player_num)
                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field,self.main_player_num)
                    end_count += 1
                    if end_count >= 5:
                        return sum_of_value / end_count
                    continue
                opponent = current_field.players[opponent_player_num]
                current_field.turn_player_num = opponent_player_num
                current_field.untap(opponent_player_num)
                current_field.increment_cost(opponent_player_num)
                current_field.start_of_turn(opponent_player_num, virtual=True)
                opponent_hand = opponent.hand
                opponent_deck = opponent.deck
                hand_len = len(opponent.hand)
                while len(opponent.hand) > 0:
                    opponent_deck.append(opponent_hand.pop())
                opponent_deck.shuffle()
                opponent.draw(opponent_deck, hand_len + 1)
            self.simulate_playout(current_field, player_num=1 - self.main_player_num)
            sum_of_value += self.state_value(current_field,self.main_player_num)

        return sum_of_value / self.default_iteration

    def simulate_playout(self, current_field, player_num=0):
        current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
        current_field.players[1].deck.shuffle()
        player = current_field.players[player_num]
        opponent = current_field.players[1 - player_num]
        current_field.get_regal_target_dict(player, opponent)

        action_count = 0
        while True:
            (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

            end_flg = player.execute_action(current_field, opponent,
                                            action_code=(action_num, card_id, target_id), virtual=True)

            if current_field.check_game_end() or end_flg:
                break

            current_field.get_regal_target_dict(player, opponent)
            action_count += 1
            if action_count > 100:
                player.show_hand()
                current_field.show_field()
                assert False
        if current_field.check_game_end():
            return
        current_field.end_of_turn(player_num, virtual=True)
        if current_field.check_game_end():
            return

    def fully_expand(self, node, player_num=0):
        if len(node.get_able_action_list()) == 0:
            return True
        if node.field.check_game_end():
            return True
        if self.main_player_num == player_num:
            if node.parent_node is not None:
                for child in node.parent_node.child_nodes:
                    if child[1] == node and child[0][0] == Action_Code.TURN_END.value:
                        return True
        return len(node.child_nodes) == len(node.children_moves)  # turn_endの場合を追加

    def expand(self, node, player_num=0):

        field = node.field

        new_choices = node.get_able_action_list()
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.players[0].deck.shuffle()
        next_field.players[1].deck.shuffle()
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])
        # regal_targetsの更新のため
        next_node = None
        if move[0] == 0:
            next_field.end_of_turn(player_num, virtual=True)
            opponent_player_num = 1 - player_num
            opponent = next_field.players[opponent_player_num]
            next_field.untap(opponent_player_num)
            next_field.increment_cost(opponent_player_num)
            next_field.start_of_turn(opponent_player_num, virtual=True)
            next_field.turn_player_num = opponent_player_num
            hand_len = len(opponent.hand)
            while len(opponent.hand) > 0:
                opponent.deck.append(opponent.hand.pop())
            opponent.deck.shuffle()
            opponent.draw(opponent.deck,num=hand_len+1)
            flg = next_field.check_game_end()
            if player_num != self.main_player_num:
                flg = True
            next_node = Node(field=next_field, player_num=opponent_player_num,
                             finite_state_flg=flg, depth=node.depth + 1)
        else:
            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = next_field.check_game_end()
            next_node = Node(field=next_field, player_num=player_num,
                             finite_state_flg=flg, depth=node.depth + 1)
        next_node.parent_node = node
        next_node.field_value = self.state_value(next_field, self.main_player_num)
        node.child_nodes.append((move, next_node))
        return next_node

    def best(self, node, player_num=0):
        children = node.child_nodes
        uct_values = {}
        for i in range(len(children)):
            if children[i][0] == (0, 0, 0) and len(children) > 1:
                uct_values[i] = 0
                continue
            uct_values[i] = self.uct(children[i][1], node, player_num=self.main_player_num)


        uct_values_list = list(uct_values.values())
        max_uct_value = max(uct_values_list)
        max_list_index = uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]

        return max_value_node, action

    def execute_best(self, node, player_num=0):
        children = node.child_nodes
        uct_values = {}
        for i in range(len(children)):
            if children[i][0] == (0,0,0) and len(children)>1:
                uct_values[i] = (children[i][1].value/children[i][1].visit_num)/10
                continue
            uct_values[i] = children[i][1].value/children[i][1].visit_num
            if children[i][1].finite_state_flg and children[i][1].field.check_game_end():
                uct_values[i] = self.state_value(children[i][1].field,player_num=self.main_player_num)
                break

        uct_values_list = list(uct_values.values())
        max_uct_value = max(uct_values_list)
        max_list_index = uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]
        return max_value_node, action

    def uct(self, child_node, node, player_num=0):
        over_all_n = node.visit_num
        n = child_node.visit_num
        w = child_node.value
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        if player_num != self.main_player_num:
            exploitation_value = 1 - exploitation_value
        if over_all_n == 0:
            over_all_n = 1
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / (n + epsilon))

        value = exploitation_value + exploration_value

        child_node.uct = value

        return value


class Improved_Aggro_MCTSPolicy(MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Improved_Aggro_MCTSPolicy"
        # RandomPolicyとAggroPolicyのPlayOutでの比較
        self.play_out_policy = AggroPolicy()
        self.parameters = {
            0: 10, 1: 30,
            2: 1, 3: 5,
            4: 20, 5: 100
        }

    def state_value(self, field, player_num):
        partial_observable_data = field.get_observable_data(player_num=player_num)
        if partial_observable_data["opponent"]["life"] <= 0:
            return 1  # WIN_BONUS
        able_attack_power_sum = 0
        if field.get_ward_list(player_num) == []:
            for card in field.card_location[player_num]:
                if card.card_category == "Creatrue" and card.can_attack_to_player():
                    able_attack_power_sum += card.power
            if able_attack_power_sum >= partial_observable_data["opponent"]["life"]:
                return 1  # WIN_BONUS
        opponent_power_sum = 0
        for card_id in field.get_creature_location()[1 - player_num]:
            opponent_power_sum += field.card_location[1 - player_num][card_id].power
        if opponent_power_sum >= partial_observable_data["player"]["life"]:
            return 0  # -WIN_BONUS

        life_ad = partial_observable_data["player"]["life"] * self.parameters[0] \
                  - partial_observable_data["opponent"]["life"] * self.parameters[1]

        hand_ad = partial_observable_data["player"]["hand_len"] * self.parameters[2] \
                  - partial_observable_data["opponent"]["hand_len"] * self.parameters[3]

        creature_location = field.get_creature_location()
        board_ad = len(creature_location[player_num]) * self.parameters[4] \
                   - len(creature_location[1 - player_num]) * self.parameters[5]
        tmp = life_ad + hand_ad + board_ad + 20 * self.parameters[0] + 9 * self.parameters[2] + 5 * self.parameters[4]
        value = 1 / (1 + np.exp(-tmp / 100))
        return value

    def decide(self, player, opponent, field):
        if self.current_node is None:
            self.main_player_num = player.player_num
            action = self.uct_search(player, opponent, field)
            self.end_count += 1
            tmp_move = action
            if len(self.current_node.child_nodes) > 0:
                next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
                # self.best(self.current_node, player_num=player.player_num)
                self.current_node = next_node
            self.node_index = 0
            self.type = "root"
            if tmp_move == (0, 0, 0):
                self.current_node = None
            return tmp_move  # action
        else:
            self.type = "blanch"
            self.node_index += 1

            if self.action_seq != [] and self.action_seq[-1] != (0, 0, 0):
                tmp_move = self.action_seq.pop(0)
                new_field = Field_setting.Field(5)
                new_field.set_data(field)
                new_field.get_regal_target_dict(new_field.players[player.player_num],
                                                new_field.players[opponent.player_num])
                self.current_node = Node(field=new_field, player_num=player.player_num)


            else:
                if self.current_node.child_nodes == []:
                    return Action_Code.ERROR.value, 0, 0
                # next_node, tmp_move = self.best(self.current_node, player_num=player.player_num)
                next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
                # self.next_node = next_node
                self.current_node = next_node
            if tmp_move == (0, 0, 0):
                self.current_node = None
            return tmp_move

    def execute_best(self, node, player_num=0):
        children = node.child_nodes
        uct_values = {}
        for i in range(len(children)):
            uct_values[i] = children[i][1].visit_num
            if children[i].finite_state_flg and children[i].field.check_game_end():
                uct_values[i] = self.state_value(children[i].field,player_num=self.main_player_num)
                break

        uct_values_list = list(uct_values.values())
        max_uct_value = max(uct_values_list)
        max_list_index = uct_values_list.index(max_uct_value)

        max_value_node = children[max_list_index][1]

        action = children[max_list_index][0]

        return max_value_node, action


class Opponent_Modeling_ISMCTSPolicy(Information_Set_MCTSPolicy):
    # デッキ公開制のときのみ
    def __init__(self, iteration=100):
        super().__init__()
        self.name = "OM(n={})_ISMCTSPolicy".format(iteration)
        self.play_out_policy = AggroPolicy()
        self.main_player_num = 0
        self.iteration = iteration
        # self.uct_c = 1. / (np.sqrt(2) * 100)
        self.parameters = {
            0: 10, 1: 30,
            2: 1, 3: 5,
            4: 20, 5: 100
        }
        self.error_count = 0

    def state_value(self, field, player_num):
        return default_state_value(field,player_num)

    def decide(self, player, opponent, field):
        self.main_player_num = player.player_num
        if self.current_node is None:
            action = self.uct_search(player, opponent, field)
            if not field.secret:
                mylogger.info("generate new tree")
                self.current_node.print_tree(single=True)
            tmp_move = action
            if len(self.current_node.child_nodes) > 0:
                next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
            self.node_index = 0
            self.type = "root"
            if tmp_move[0] == Action_Code.TURN_END.value:
                if not field.secret:
                    if self.prev_node is not None:
                        mylogger.info("children_moves:{}".format(self.prev_node.children_moves))
                    if self.prev_node is not None and len(self.prev_node.children_moves) > 1:
                        for child in self.prev_node.child_nodes:
                            mylogger.info("action:{} value:{}".format(self.prev_node.node_id_2_edge_action[id(child)],
                                                                      child.value/max(1,child.visit_num)))
                self.prev_node = self.current_node
                self.current_node = None
                self.error_count = 0
            else:
                self.prev_node = self.current_node
                self.current_node = next_node

            return tmp_move  # action
        else:
            self.type = "blanch"
            hit_flg = False
            for child in self.prev_node.child_nodes:
                sim_player_hand = child.field.players[player.player_num].hand
                if child.field.eq(field) and player.compare_hand(sim_player_hand):
                    hit_flg = True
                    self.current_node = child
                    break

            if hit_flg:
                if not field.secret:
                    mylogger.info("corresponding node is not found(visit_num:{},child_num:{})"
                                  .format(self.current_node.visit_num,
                                          len(self.current_node.child_nodes)))
                if not field.secret:
                    mylogger.info("reuse current_node as root")
                self.uct_search(player, opponent, field,use_existed_node=True)

            else:
                return Action_Code.ERROR.value,None,None

            if not field.secret:
                mylogger.info("use existed tree")
            next_node, tmp_move = self.execute_best(self.current_node, player_num=player.player_num)
            if tmp_move[0] == Action_Code.TURN_END.value:
                if len(self.prev_node.children_moves) > 1:
                    if not field.secret:
                        mylogger.info("choices:{}".format(self.prev_node.children_moves))
                self.error_count = 0
                self.current_node = None
            else:
                self.prev_node = self.current_node
                self.current_node = next_node
            return tmp_move  # action

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and check is False:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            current_player_num = node.field.turn_player_num
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    parent_node = node
                    node, move = self.best(node, player_num=current_player_num)
                    exist_flg = False
                    for child in parent_node.child_nodes:
                        if move == parent_node.node_id_2_edge_action[id(child)]:
                            if child.field.eq(node.field):
                                exist_flg = True
                                break
                    if not exist_flg:
                        # mylogger.info("new_node")
                        return self.expand(node, player_num=current_player_num)

                else:
                    check = self.fully_expand(node, player_num=current_player_num)
                    if not check:
                        return self.expand(node, player_num=current_player_num)
                    else:
                        parent_node = node
                        node, move = self.best(node, player_num=current_player_num)
                        exist_flg = False
                        for child in parent_node.child_nodes:
                            if move == parent_node.node_id_2_edge_action[id(child)]:
                                if child.field.eq(node.field):
                                    exist_flg = True
                                    break
                        if not exist_flg:
                            # mylogger.info("new_node")
                            return self.expand(node, player_num=current_player_num)
                length_of_children = len(node.child_nodes)
            else:
                return self.expand(node, player_num=current_player_num)

            count += 1
            if count > 100:
                field = node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node

    # @jit
    def default_policy(self, node, player_num=0):
        sum_of_value = 0
        if node.finite_state_flg:
            return self.state_value(node.field, player_num=self.main_player_num)
        end_count = 0
        current_field = Field_setting.Field(5)
        opponent_player_num = 1 - self.main_player_num
        for i in range(self.default_iteration):
            current_field.set_data(node.field)
            if self.main_player_num == player_num:
                current_field.turn_player_num = player_num
                self.simulate_playout(current_field, player_num=self.main_player_num)
                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field,self.main_player_num)
                    end_count += 1
                    if end_count >= 5:
                        return sum_of_value / end_count
                    continue
                opponent = current_field.players[opponent_player_num]
                current_field.turn_player_num = opponent_player_num
                current_field.untap(opponent_player_num)
                current_field.increment_cost(opponent_player_num)
                current_field.start_of_turn(opponent_player_num, virtual=True)
                opponent_hand = opponent.hand
                opponent_deck = opponent.deck
                hand_len = len(opponent.hand)
                while len(opponent.hand) > 0:
                    opponent_deck.append(opponent_hand.pop())
                opponent_deck.shuffle()
                opponent.draw(opponent_deck, hand_len + 1)
            self.simulate_playout(current_field, player_num=1 - self.main_player_num)
            sum_of_value += self.state_value(current_field, player_num=self.main_player_num)
        if node.parent_node.node_id_2_edge_action[id(node)][0] == Action_Code.TURN_END.value:
            if len(node.parent_node.edge_action_2_node_id) > 1:
                sum_of_value /= 10
        return sum_of_value / self.default_iteration

    def simulate_playout(self, current_field, player_num=0):
        current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
        current_field.players[1].deck.shuffle()
        player = current_field.players[player_num]
        opponent = current_field.players[1 - player_num]
        current_field.get_regal_target_dict(player, opponent)
        action_count = 0
        while True:
            (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

            end_flg = player.execute_action(current_field, opponent,
                                            action_code=(action_num, card_id, target_id), virtual=True)

            if current_field.check_game_end() or end_flg:
                break

            current_field.get_regal_target_dict(player, opponent)
            action_count += 1
            if action_count > 100:
                player.show_hand()
                current_field.show_field()
                assert False
        if current_field.check_game_end():
            return
        current_field.end_of_turn(player_num, virtual=True)
        if current_field.check_game_end():
            return

    def fully_expand(self, node, player_num=0):
        if node.field.check_game_end():
            return True
        if self.main_player_num == player_num:
            if node.parent_node is not None:
                for child in node.parent_node.child_nodes:
                    if child is node:
                        action = node.parent_node.node_id_2_edge_action[id(child)]
                        if action[0] == Action_Code.TURN_END.value:
                            return True
            turn_end_counter_flg = True
            for child in node.child_nodes:
                action = node.node_id_2_edge_action[id(child)]
                if action[0] == Action_Code.TURN_END.value:
                    if turn_end_counter_flg:
                        turn_end_counter_flg = False
                    else:
                        if len(node.children_moves) == len(list(node.edge_action_2_node_id.keys())):
                            return True
        return False
        remain_actions = list(set(node.children_moves) - set(node.child_actions))
        if remain_actions != []:
            remain_num = len(remain_actions)
            player = node.field.players[player_num]
            for action in remain_actions:
                if action[0] == Action_Code.TURN_END.value:
                    check = action in node.edge_action_2_node_id
                    if check:
                        remain_num -= 1

                if remain_num <= 0:
                    return True
                if action[0] == Action_Code.EVOLVE.value:
                    evo_actions = list(filter(lambda cell: cell[0] == -1, node.child_actions))
                    side = node.field.card_location[player_num]
                    opponent_side = node.field.card_location[1 - player_num]
                    for evo_action in evo_actions:
                        if side[action[1]].eq(side[evo_action[1]]):
                            if action[2] is None:
                                # return True
                                node.children_moves.remove(action)
                                remain_num -= 1
                                break
                            else:
                                if side[action[1]].evo_target == Target_Type.ENEMY_FOLLOWER.value:
                                    if opponent_side[action[2]].eq(opponent_side[evo_action[2]]):
                                        node.children_moves.remove(action)
                                        remain_num -= 1
                                        break

                if remain_num <= 0:
                    return True
                if action[0] == Action_Code.PLAY_CARD.value:
                    play_actions = list(filter(lambda cell: cell[0] == Action_Code.PLAY_CARD.value, node.child_actions))
                    for play_action in play_actions:
                        if player.hand[action[1]].eq(player.hand[play_action[1]]):
                            # return True
                            node.children_moves.remove(action)
                            remain_num -= 1
                            break
                if remain_num <= 0:
                    return True
                if action[0] == Action_Code.ATTACK_TO_FOLLOWER.value:
                    attack_actions = \
                        list(filter(lambda cell: cell[0] == Action_Code.ATTACK_TO_FOLLOWER.value, node.child_actions))
                    field = node.field
                    side = field.card_location[player_num]
                    opponent_side = field.card_location[1 - player_num]
                    length = len(opponent_side)
                    assert action[2] < length, "out-of-range(action):{},{}".format(action[2], length)
                    for attack_action in attack_actions:
                        assert attack_action[2] < length, "out-of-range(attack_action):{},{}".format(action[2], length)
                        if side[action[1]].power == 0 and KeywordAbility.BANE.value not in side[action[1]].ability:
                            # return True
                            node.children_moves.remove(action)
                            remain_num -= 1
                            break
                        if side[action[1]].eq(side[attack_action[1]]):
                            if opponent_side[action[2]].eq(opponent_side[attack_action[2]]):
                                node.children_moves.remove(action)
                                remain_num -= 1
                                break
                if remain_num <= 0:
                    return True
            return False
        return True

    def expand(self, node, player_num=0):
        child_node_fields = []
        for cell in node.child_nodes:
            child_node_fields.append(cell.field)
        old_choices = node.children_moves[:]
        new_choices = node.children_moves[:]
        if (0,0,0) in node.edge_action_2_node_id:
            exist_flg = False
            for node_id in node.edge_action_2_node_id[(0,0,0)]:
                for child in node.child_nodes:
                    if id(child) == node_id and child.visit_num >= 5:
                        old_choices.remove((0,0,0))
                        new_choices.remove((0, 0, 0))
                        exist_flg = True
                        break
                if exist_flg:break
        #assert len(new_choices) > 0, "non-choice-error"
        if len(new_choices) == 0:
            return node
        next_node, move, exist_flg = self.execute_single_action(node, new_choices, child_node_fields,
                                                                player_num=player_num)
        if exist_flg:
            while exist_flg:
                new_choices = list(set(node.children_moves) - set(node.child_actions))
                if len(new_choices) == 0:
                    return self.best(node, player_num)[0]
                next_node, move, exist_flg = self.execute_single_action(node, new_choices, child_node_fields,
                                                                        player_num=player_num)

        node.node_id_2_edge_action[id(next_node)] = move
        next_node.parent_node = node
        node.child_nodes.append(next_node)
        if move in node.edge_action_2_node_id:
            node.edge_action_2_node_id[move].append(next_node)
            # node.edge_action_2_node_id[move].append(id(next_node))
        else:
            node.edge_action_2_node_id[move] = [next_node]
            # node.edge_action_2_node_id[move] = [id(next_node)]
        if move not in node.child_actions:
            node.child_actions.append(move)
        return next_node

    def execute_single_action(self, node, new_choices, child_node_fields, player_num=0):
        field = node.field
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.players[0].deck.shuffle()
        next_field.players[1].deck.shuffle()
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])
        next_node = None
        exist_flg = False
        if move[0] == Action_Code.TURN_END.value:
            next_field.end_of_turn(player_num, virtual=True)
            opponent_player_num = 1 - player_num
            opponent = next_field.players[opponent_player_num]
            next_field.untap(opponent_player_num)
            next_field.increment_cost(opponent_player_num)
            next_field.start_of_turn(opponent_player_num, virtual=True)
            next_field.turn_player_num = opponent_player_num
            hand_len = len(opponent.hand)
            while len(opponent.hand) > 0:
                opponent.deck.append(opponent.hand.pop())
            opponent.deck.shuffle()
            opponent.draw(opponent.deck, num=hand_len + 1)
            flg = next_field.check_game_end()
            if player_num != self.main_player_num:
                flg = True
            next_node = New_Node(field=next_field, player_num=1 - player_num, finite_state_flg=flg,
                                 depth=node.depth + 1)
            for child in node.child_nodes:
                if node.node_id_2_edge_action[id(child)] == Action_Code.TURN_END.value:
                    if child.field.eq(next_field):#and opponent.eq(child.field.players[1 - player_num]):
                        if move not in node.action_counter:
                            node.action_counter[move] = 0
                        node.action_counter[move] += 1
                        if node.action_counter[move] >= 5:
                            node.children_moves.remove(move)
                        exist_flg = True
                        break

        else:
            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = next_field.check_game_end()
            next_node = New_Node(field=next_field, player_num=player_num,
                                 finite_state_flg=flg, depth=node.depth + 1)
            for child_field in child_node_fields:
                if child_field.eq(next_field) and \
                        child_field.players[player_num].eq(next_field.players[player_num]):
                    if move not in node.action_counter:
                        node.action_counter[move] = 0
                    node.action_counter[move] += 1
                    if node.action_counter[move] >= 5:
                        node.children_moves.remove(move)
                    exist_flg = True
                    break

        return next_node, move, exist_flg

    def best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            if action == (0,0,0) and len(node.edge_action_2_node_id) > 1:
                value = -1
            else:
                value = self.uct(children[i], node, player_num=player_num)
            action_uct_values[action].append(value)
            action_2_node[action].append(children[i])
        max_value = None
        max_value_action = None
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            weights = [cell.visit_num for cell in action_2_node[key]]
            visit_num_sum = sum(weights)
            mean_value = sum([values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            if max_value is None or mean_value > max_value:
                max_value = mean_value
                max_value_action = key
        # weights = [cell.visit_num for cell in action_2_node[max_value_action]]
        # max_value_node = random.choices(action_2_node[max_value_action], weights=weights)[0]
        max_value_node = random.choices(action_2_node[max_value_action])[0]
        return max_value_node, max_value_action

    def execute_best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            if action == (0,0,0) and len(node.edge_action_2_node_id) > 1:
                action_uct_values[action].append(0)
                action_2_node[action].append(children[i])
                continue
            if children[i].finite_state_flg and children[i].field.check_game_end():
                player = children[i].field.players[player_num]
                if player.life <= 0 or player.lib_out_flg:
                    action_uct_values[action].append(0)
                else:
                    action_uct_values[action].append(100)
            else:
                action_uct_values[action].append(children[i].value/children[i].visit_num)
            action_2_node[action].append(children[i])
        max_value = None
        max_value_action = None
        if len(action_uct_values) == 0:
            return None, (Action_Code.ERROR.value, 0, 0)
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            weights = [cell.visit_num for cell in action_2_node[key]]
            visit_num_sum = sum(weights)
            # mean_value = sum(values)/len(values)
            mean_value = sum([values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            if max_value is None or mean_value > max_value:
                max_value = mean_value
                max_value_action = key
        assert max_value_action is not None, "action is None!"
        # weights = [cell.visit_num for cell in action_2_node[max_value_action]]
        # max_value_node = random.choices(action_2_node[max_value_action], weights=weights)[0]
        #max_value_node = random.choices(action_2_node[max_value_action])[0]
        max_value_node = random.choice(action_2_node[max_value_action])
        return max_value_node, max_value_action

    def uct(self, child_node, node, player_num=0):
        over_all_n = node.visit_num
        n = child_node.visit_num
        w = child_node.value
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        if player_num != self.main_player_num:
            exploitation_value = 1 - exploitation_value
        if node.node_id_2_edge_action[id(child_node)][0] == Action_Code.TURN_END.value:
            exploitation_value = max(exploitation_value / 10, exploitation_value - 0.5)
        # assert over_all_n>0,"over_all_n:{}".format(over_all_n)
        if over_all_n == 0:
            over_all_n = 1
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / (n + epsilon))

        value = exploitation_value + exploration_value

        child_node.uct = value

        return value


def get_draw_probability(prev_field, current_field, player_num=0):
    prev_hand = prev_field.players[player_num].hand
    current_hand = current_field.players[player_num].hand
    diffarance = len(current_hand) - len(prev_hand)
    if diffarance > 0:
        prev_deck = prev_field.players[player_num].deck
        card_set = prev_deck.get_remain_card_set()
        remain_nums = {}
        draw_cards = {}
        last_id = len(current_hand) - 1
        for i in range(diffarance):
            card_name = current_hand[last_id - i].name
            if card_name in card_set:
                remain_nums[card_name] = card_set[card_name]
                if card_name not in draw_cards:
                    draw_cards[card_name] = 1
                else:
                    draw_cards[card_name] += 1

        denominator = comb(len(prev_deck.deck), sum(draw_cards.values()), exact=True)
        numerator = 1
        for key in list(draw_cards.keys()):
            assert remain_nums[key] >= draw_cards[key], "over-draw-error"
            numerator *= comb(remain_nums[key], draw_cards[key], exact=True)

        return numerator / denominator

    return 0


class Alter_Opponent_Modeling_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__(iteration=iteration)
        self.name = "Alter_OM(n={})_ISMCTSPolicy".format(iteration)

    def best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            value = self.uct(children[i], node, player_num=player_num)
            action_uct_values[action].append(value)
            action_2_node[action].append(children[i])
        max_value = None
        max_value_action = None
        max_value_node = None
        draw_actions = {}
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            prev_hand_len = len(node.field.players[player_num].hand)
            next_hand_len = -1
            flg = True
            for child in action_2_node[key]:
                if next_hand_len == -1:
                    next_hand_len = len(child.field.players[player_num].hand)
                    if next_hand_len <= prev_hand_len:
                        flg = False
                        break
                else:
                    if len(child.field.players[player_num].hand) != next_hand_len:
                        flg = False
            estimated_value = 0
            if flg:
                distribution = [get_draw_probability(node.field, cell.field, player_num=player_num)
                                for cell in action_2_node[key]]
                draw_actions[key] = distribution
                sum_of_distribution = sum(distribution)
                assert sum_of_distribution > 0, "distribution:{}".format(distribution)
                estimated_value = sum(
                    [values[node_id] * (distribution[node_id] / sum_of_distribution) for node_id in range(len(values))])

            else:
                weights = [cell.visit_num for cell in action_2_node[key]]
                visit_num_sum = sum(weights)
                estimated_value = sum(
                    [values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            if max_value is None or estimated_value > max_value:
                max_value = estimated_value
                max_value_action = key
        if max_value_action in draw_actions:
            max_value_node = random.choices(action_2_node[max_value_action], weights=draw_actions[max_value_action])[0]
        else:
            max_value_node = random.choices(action_2_node[max_value_action])[0]
        return max_value_node, max_value_action


class Neo_MCTSPolicy(Flexible_Iteration_MCTSPolicy):
    def __init__(self, N=100, probability=0.5):
        super().__init__(N=N)
        self.name = "Neo(n={},p={})_MCTSPolicy".format(N, probability)
        self.iteration = N
        self.play_out_policy = AggroPolicy()
        self.sub_play_out_policy = RandomPolicy()
        self.probability = probability

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        current_field = Field_setting.Field(5)
        for i in range(self.default_iteration):
            if not node.finite_state_flg:
                current_field.set_data(node.field)
                current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
                current_field.players[1].deck.shuffle()
                player = current_field.players[player_num]
                opponent = current_field.players[1 - player_num]
                current_field.get_regal_target_dict(player, opponent)

                action_count = 0
                action_num, card_id, target_id = 0, 0, 0
                while True:
                    if random.random() < self.probability:
                        (action_num, card_id, target_id) = self.sub_play_out_policy.decide(player, opponent,
                                                                                           current_field)
                    else:
                        (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent,
                                                                                       current_field)
                    end_flg = player.execute_action(current_field, opponent,
                                                    action_code=(action_num, card_id, target_id), virtual=True)

                    if current_field.check_game_end() or end_flg:
                        break

                    current_field.get_regal_target_dict(player, opponent)
                    action_count += 1
                    if action_count > 100:
                        player.show_hand()
                        current_field.show_field()
                        assert False
                if current_field.check_game_end():
                    sum_of_value += WIN_BONUS
                    return sum_of_value
                current_field.end_of_turn(player_num, virtual=True)
                if current_field.check_game_end():
                    sum_of_value += WIN_BONUS
                    return sum_of_value
                else:
                    sum_of_value += self.state_value(current_field, player_num)

        return sum_of_value / self.default_iteration


class Neo_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100, probability=0.5):
        super().__init__(iteration=iteration)
        self.sub_play_out_policy = RandomPolicy()
        self.probability = probability
        self.name = "Neo(n={},p={})_OM_ISMCTSPolicy".format(iteration, probability)

    def simulate_playout(self, current_field, player_num=0):
        current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
        current_field.players[1].deck.shuffle()
        player = current_field.players[player_num]
        opponent = current_field.players[1 - player_num]
        current_field.get_regal_target_dict(player, opponent)
        action_count = 0
        action_num, card_id, target_id = 0, 0, 0
        while True:
            if random.random() < self.probability:
                (action_num, card_id, target_id) = self.sub_play_out_policy.decide(player, opponent, current_field)
            else:
                (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

            end_flg = player.execute_action(current_field, opponent,
                                            action_code=(action_num, card_id, target_id), virtual=True)

            if current_field.check_game_end() or end_flg:
                break

            current_field.get_regal_target_dict(player, opponent)
            action_count += 1
            if action_count > 100:
                player.show_hand()
                current_field.show_field()
                assert False
        if current_field.check_game_end():
            return
        current_field.end_of_turn(player_num, virtual=True)
        if current_field.check_game_end():
            return


class MAST_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__(iteration=iteration)
        self.name = "MAST(n={})_OM_ISMCTSPolicy".format(iteration)
        self.Q_dict = {}
        self.previous_Q_dict = {}

    def simulate_playout(self, current_field, player_num=0):
        current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
        current_field.players[1].deck.shuffle()
        player = current_field.players[player_num]
        opponent = current_field.players[1 - player_num]
        current_field.get_regal_target_dict(player, opponent)
        action_count = 0
        end_flg = False
        action_history = []
        previous_action = None
        while True:
            if player_num == self.main_player_num:
                able_actions = self.get_able_actions(current_field, player_num=player_num)
                action_code_2_key = {}
                key_2_action_code = {}
                able_choices = {}
                if previous_action is not None and previous_action not in self.previous_Q_dict:
                    self.previous_Q_dict[previous_action] = {}
                for cell_id, action in enumerate(able_actions):
                    origin = (action[0], action[1], action[2])
                    # mylogger.info("action:{}".format(action))
                    key = None
                    if origin[0] == Action_Code.PLAY_CARD.value:
                        hand_id = able_actions[cell_id][1]
                        key = (action[0], player.hand[hand_id].name)
                    elif origin[0] == Action_Code.ATTACK_TO_FOLLOWER.value or origin[
                        0] == Action_Code.ATTACK_TO_PLAYER.value:
                        attacker_id = able_actions[cell_id][1]
                        key = (action[0], current_field.card_location[player_num][attacker_id].name)
                    elif origin[0] == Action_Code.EVOLVE.value:
                        evo_id = able_actions[cell_id][1]
                        key = (action[0], current_field.card_location[player_num][evo_id].name)
                    else:
                        key = (0, 0)
                    action_code_2_key[origin] = key
                    key_2_action_code[key] = origin
                    if key not in self.Q_dict:
                        self.Q_dict[key] = [0, 0]
                    if previous_action is not None and key not in self.previous_Q_dict[previous_action]:
                        self.previous_Q_dict[previous_action][key] = [0, 0]
                    if key not in able_choices:
                        if previous_action is None:
                            able_choices[key] = self.Q_dict[key]
                        else:
                            able_choices[key] = self.previous_Q_dict[previous_action][key]
                max_value = None
                max_keys = []
                assert len(able_choices) > 0, "{}".format(able_choices)
                n = max(1, sum([able_choices[key][1] for key in list(able_choices.keys())]))
                for key in list(able_choices.keys()):
                    uct_value = self.sim_uct(key, over_all_n=n, player_num=player_num, previous_action=previous_action)
                    if max_value is None or max_value < uct_value:
                        max_keys = [key]
                        max_value = uct_value
                    elif max_value == uct_value:
                        max_keys.append(key)
                max_key = random.choice(max_keys)
                action_history.append(max_key)
                (action_num, card_id, target_id) = key_2_action_code[max_key]

                before_value = self.state_value(current_field, self.main_player_num)
                end_flg = player.execute_action(current_field, opponent,
                                                action_code=(action_num, card_id, target_id), virtual=True)
                after_value = self.state_value(current_field, self.main_player_num)
                win_flg = after_value > 0.99
                self.Q_dict[max_key][0] += int(win_flg) \
                                           + (1 - int(win_flg)) * (after_value - before_value)
                self.Q_dict[max_key][1] += 1
                if previous_action is not None:
                    self.previous_Q_dict[previous_action][max_key][0] += int(win_flg) \
                                                                         + (1 - int(win_flg)) * (
                                                                                 after_value - before_value)
                    self.previous_Q_dict[previous_action][max_key][1] += 1
                previous_action = max_key
            else:
                (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)
                end_flg = player.execute_action(current_field, opponent,
                                                action_code=(action_num, card_id, target_id), virtual=True)

            if current_field.check_game_end() or end_flg:
                break

            current_field.get_regal_target_dict(player, opponent)
            action_count += 1
            if action_count > 100:
                player.show_hand()
                current_field.show_field()
                assert False
        if current_field.check_game_end():
            return
        current_field.end_of_turn(player_num, virtual=True)
        if current_field.check_game_end():
            return

    def sim_uct(self, key, over_all_n=None, player_num=0, previous_action=None):
        n, w = 0, 0
        if previous_action is None:
            n = self.Q_dict[key][1]
            w = self.Q_dict[key][0]
        else:
            n = self.previous_Q_dict[previous_action][key][1]
            w = self.previous_Q_dict[previous_action][key][0]
        if player_num != self.main_player_num:
            w = -w
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        exploration_value = 2.0 * c * np.sqrt(np.log(over_all_n) / (n + epsilon))

        value = exploitation_value + exploration_value

        return value

    def get_able_actions(self, field, player_num=0):
        children_moves = []
        end_flg = field.check_game_end()
        if not end_flg:
            player = field.players[player_num]
            field.update_hand_cost(player_num=player_num)
            (ward_list, _, can_be_attacked, regal_targets) = \
                field.get_situation(player, field.players[1 - player_num])

            (_, _, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) = \
                field.get_flag_and_choices(player, field.players[1 - player_num], regal_targets)
            children_moves.append((0, 0, 0))
            remain_able_to_play = able_to_play[:]
            hand_id = 0
            while hand_id < len(player.hand):
                if hand_id in remain_able_to_play:
                    other_hand_ids = remain_able_to_play[:]
                    other_hand_ids.remove(hand_id)
                    if other_hand_ids is not None:
                        for other_id in other_hand_ids:
                            if player.hand[hand_id].eq(player.hand[other_id]):
                                remain_able_to_play.remove(other_id)
                hand_id += 1
            able_to_play = remain_able_to_play

            remain_able_to_creature_attack = able_to_creature_attack[:]
            remain_able_to_attack = able_to_attack[:]
            remain_able_to_evo = able_to_evo[:]
            location_id = 0
            side = field.card_location[player_num]

            while location_id < len(side):
                if location_id in remain_able_to_creature_attack:
                    other_follower_ids = remain_able_to_creature_attack[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_creature_attack.remove(other_id)
                if location_id in remain_able_to_attack:
                    other_follower_ids = remain_able_to_attack[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_attack.remove(other_id)
                if location_id in remain_able_to_evo:
                    other_follower_ids = remain_able_to_evo[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_able_to_evo.remove(other_id)
                location_id += 1
            able_to_creature_attack = remain_able_to_creature_attack
            able_to_attack = remain_able_to_attack
            able_to_evo = remain_able_to_evo
            remain_can_be_attacked = can_be_attacked[:]
            remain_ward_list = ward_list[:]
            side = field.card_location[1 - player_num]
            location_id = 0
            while location_id < len(side):
                if location_id in remain_can_be_attacked:
                    other_follower_ids = remain_can_be_attacked[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_can_be_attacked.remove(other_id)
                if location_id in remain_ward_list:
                    other_follower_ids = remain_ward_list[:]
                    other_follower_ids.remove(location_id)
                    if len(other_follower_ids) > 0:
                        for other_id in other_follower_ids:
                            if side[location_id].eq(side[other_id]):
                                remain_ward_list.remove(other_id)
                location_id += 1
            can_be_attacked = remain_can_be_attacked
            ward_list = remain_ward_list

            for play_id in able_to_play:
                if field.players[player_num].hand[play_id].cost > field.remain_cost[player_num]:
                    continue
                if field.players[player_num].hand[play_id].card_category != "Spell" and len(
                        field.card_location[player_num]) >= field.max_field_num:
                    continue
                if len(regal_targets[play_id]) > 0:
                    for i in range(len(regal_targets[play_id])):
                        children_moves.append((1, play_id, regal_targets[play_id][i]))
                else:
                    if field.players[player_num].hand[play_id].card_category == "Spell":
                        if field.players[player_num].hand[play_id].have_target == 0:
                            children_moves.append((1, play_id, None))
                    else:
                        children_moves.append((1, play_id, None))

            if len(ward_list) == 0:
                for attacker_id in able_to_creature_attack:
                    for target_id in can_be_attacked:
                        children_moves.append((2, attacker_id, target_id))
                for attacker_id in able_to_attack:
                    children_moves.append((3, attacker_id, None))
            else:
                for attacker_id in able_to_creature_attack:
                    for target_id in ward_list:
                        children_moves.append((2, attacker_id, target_id))

            if can_evo:
                for evo_id in able_to_evo:
                    evo_creature = field.card_location[player_num][evo_id]
                    if evo_creature.evo_target is None:
                        children_moves.append((-1, evo_id, None))
                    else:
                        targets = field.get_regal_targets(evo_creature, target_type=0, player_num=player_num)
                        for target_id in targets:
                            children_moves.append((-1, evo_id, target_id))

            return children_moves


class Damped_Sampling_MCTS(Flexible_Iteration_Aggro_MCTSPolicy):
    def __init__(self, N=100):
        super().__init__(N=N)
        self.name = "Damped_Sampling(n={})_MCTSPolicy".format(N)

    def default_policy(self, node, player_num=0):
        if node.visit_num >= 10:
            return node.value / node.visit_num
        return super().default_policy(node, player_num=player_num)


class Damped_Sampling_ISMCTS(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__(iteration=iteration)
        self.name = "Damped_Sampling(n={})_ISMCTSPolicy".format(iteration)

    def default_policy(self, node, player_num=0):
        if node.visit_num >= 10:
            return node.value / node.visit_num
        return super().default_policy(node, player_num=player_num)


class Sampling_ISMCTS(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100, th=5):
        super().__init__(iteration=iteration)
        self.name = "Sampling(n={},th={})_ISMCTSPolicy".format(iteration, th)
        self.th = th

    def tree_policy(self, node, player_num=0):

        length_of_children = len(node.child_nodes)
        check = self.fully_expand(node, player_num=player_num)
        if length_of_children == 0 and check is False:
            return self.expand(node, player_num=player_num)
        count = 0
        while not node.finite_state_flg:
            current_player_num = node.field.turn_player_num
            if length_of_children > 0:
                if random.uniform(0, 1) < .5:
                    # parent_node = node
                    node, move = self.best(node, player_num=current_player_num)
                    """
                    exist_flg = False
                    for child in parent_node.child_nodes:
                        if move == parent_node.node_id_2_edge_action[id(child)]:
                            if child.field.eq(node.field):
                                exist_flg = True
                                break
                    if not exist_flg:
                        # mylogger.info("new_node")
                        return self.expand(node, player_num=current_player_num)
                    """

                else:
                    check = self.fully_expand(node, player_num=current_player_num)
                    if not check:
                        return self.expand(node, player_num=current_player_num)
                    else:
                        # parent_node = node
                        node, move = self.best(node, player_num=current_player_num)
                        """
                        exist_flg = False
                        for child in parent_node.child_nodes:
                            if move == parent_node.node_id_2_edge_action[id(child)]:
                                if child.field.eq(node.field):
                                    exist_flg = True
                                    break
                        if not exist_flg:
                            # mylogger.info("new_node")
                            return self.expand(node, player_num=current_player_num)
                        """
                length_of_children = len(node.child_nodes)
            else:
                return self.expand(node, player_num=current_player_num)

            count += 1
            if count > 100:
                field = node.field
                node.field.show_field()
                mylogger.info("finite:{}".format(node.finite_state_flg))
                assert False,"infinite loop!"

        return node

    def expand(self, node, player_num=0):
        child_node_fields = []
        for cell in node.child_nodes:
            child_node_fields.append(cell.field)
        old_choices = node.children_moves
        new_choices = node.children_moves
        assert len(new_choices) > 0, "non-choice-error"
        next_node, move, exist_flg = self.execute_single_action(node, new_choices, child_node_fields,
                                                                player_num=player_num)

        if exist_flg:
            if move in node.edge_action_2_node_id:
                threthold = []
                for child in node.edge_action_2_node_id[move]:
                    threthold.append(child.visit_num)
                if sum(threthold) > self.th:
                    return random.choices(node.edge_action_2_node_id[move], weights=threthold, k=1)[0]
            while exist_flg:
                new_choices = list(set(node.children_moves) - set(node.child_actions))
                # assert len(new_choices) > 0, "non-choice-error({},{})".format(node.children_moves,node.child_actions)
                if len(new_choices) == 0:
                    return self.best(node, player_num)[0]
                next_node, move, exist_flg = self.execute_single_action(node, new_choices, child_node_fields,
                                                                        player_num=player_num)

        node.node_id_2_edge_action[id(next_node)] = move
        next_node.parent_node = node
        node.child_nodes.append(next_node)
        if move in node.edge_action_2_node_id:
            node.edge_action_2_node_id[move].append(next_node)
            # node.edge_action_2_node_id[move].append(id(next_node))
        else:
            node.edge_action_2_node_id[move] = [next_node]
            # node.edge_action_2_node_id[move] = [id(next_node)]
        if move not in node.child_actions:
            node.child_actions.append(move)
            assert move in node.children_moves, "illegal-move!"
        return next_node

    def best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        for action in list(node.edge_action_2_node_id.keys()):
            action_uct_values[action] = []
            for child in node.edge_action_2_node_id[action]:
                value = self.uct(child, node, player_num=player_num)
                action_uct_values[action].append(value)
        max_value = None
        max_value_action = None
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            weights = [cell.visit_num for cell in node.edge_action_2_node_id[key]]
            visit_num_sum = sum(weights)
            mean_value = sum([values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            if max_value is None or mean_value > max_value:
                max_value = mean_value
                max_value_action = key
        max_value_node = random.choices(node.edge_action_2_node_id[max_value_action])[0]
        return max_value_node, max_value_action


def simple_state_value(field, player_num):
    if field.check_game_end():
        if field.players[player_num].life <= 0 or field.players[player_num].lib_out_flg:
            return 0
        return 1
    return 0.5


class Simple_value_function_A_MCTSPolicy(New_Aggro_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Simple_value_function_New_A_MCTSPolicy"

    def state_value(self, field, player_num):
        return simple_state_value(field,player_num)


class Simple_value_function_ISMCTSPolicy(Information_Set_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Simple_value_function_ISMCTSPolicy"

    def state_value(self, field, player_num):
        return simple_state_value(field,player_num)


class Simple_value_function_OM_MCTSPolicy(Opponent_Modeling_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Simple_value_function_OM_MCTSPolicy"

    def state_value(self, field, player_num):
        return simple_state_value(field,player_num)


class Simple_value_function_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Simple_value_function_OM_ISMCTSPolicy"

    def state_value(self, field, player_num):
        return simple_state_value(field,player_num)


def second_state_value(field, player_num):

    partial_observable_data = field.get_observable_data(player_num=player_num)
    if field.check_game_end():
        if partial_observable_data["player"]["life"] <= 0 or field.players[player_num].lib_out_flg:
            return 0
        elif partial_observable_data["opponent"]["life"] <= 0 or field.players[1 - player_num].lib_out_flg:
            return 1
    card_location = field.card_location
    life_diff = partial_observable_data["player"]["life"] - partial_observable_data["opponent"]["life"]
    life_diff /= 40
    hand_diff = partial_observable_data["player"]["hand_len"] - partial_observable_data["opponent"]["hand_len"]
    hand_diff /= 18
    board_diff = len(card_location[player_num]) - len(card_location[1 - player_num])
    board_diff /= 10
    value = board_diff * 0.45 + life_diff * 0.45 + hand_diff * 0.1
    return 1 / (1 + np.exp(-(value * 3)))


class Second_value_function_A_MCTSPolicy(New_Aggro_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Second_value_function_New_A-MCTSPolicy"

    def state_value(self, field, player_num):
        return second_state_value(field,player_num)


class Second_value_function_OM_MCTSPolicy(Opponent_Modeling_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Second_value_function_OM_MCTSPolicy"

    def state_value(self, field, player_num):
        return second_state_value(field,player_num)


class Second_value_function_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Second_value_function_OM-ISMCTSPolicy"

    def state_value(self, field, player_num):
        return second_state_value(field,player_num)


class Flexible_Simulation_A_MCTSPolicy(Aggro_MCTSPolicy):
    def __init__(self, sim_num=10):
        super().__init__()
        self.name = "{}_Simulation_A_MCTSPolicy".format(sim_num)
        self.default_iteration = sim_num


class Flexible_Simulation_MO_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, sim_num=10):
        super().__init__()
        self.name = "{}_Simulation_MO_ISMCTSPolicy".format(sim_num)
        self.default_iteration = sim_num


class Cheating_MO_MCTSPolicy(Opponent_Modeling_MCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__()
        self.name = "Cheating_MO_MCTS(iteration={})Policy".format(iteration)
        self.iteration = iteration
    """
    def default_policy(self, node, player_num=0):
        sum_of_value = 0
        if node.finite_state_flg:
            return self.state_value(node.field, player_num=self.main_player_num)
        end_count = 0
        current_field = Field_setting.Field(5)
        opponent_player_num = 1 - self.main_player_num
        t1 = None
        for i in range(self.default_iteration):
            current_field.set_data(node.field)
            if self.main_player_num == player_num:
                current_field.turn_player_num = player_num
                self.simulate_playout(current_field, player_num=self.main_player_num)
                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field,self.main_player_num)
                    end_count += 1
                    if end_count >= 5:
                        return sum_of_value / end_count
                    continue
                opponent = current_field.players[opponent_player_num]
                current_field.turn_player_num = opponent_player_num
                current_field.untap(opponent_player_num)
                current_field.increment_cost(opponent_player_num)
                current_field.start_of_turn(opponent_player_num, virtual=True)
                opponent.draw(opponent.deck,1)
            self.simulate_playout(current_field, player_num=1 - self.main_player_num)
            sum_of_value += self.state_value(current_field, player_num=self.main_player_num)

        return sum_of_value / self.default_iteration

    def simulate_playout(self, current_field, player_num=0):
        player = current_field.players[player_num]
        opponent = current_field.players[1 - player_num]
        current_field.get_regal_target_dict(player, opponent)

        action_count = 0
        while True:
            (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

            end_flg = player.execute_action(current_field, opponent,
                                            action_code=(action_num, card_id, target_id), virtual=True)

            if current_field.check_game_end() or end_flg:
                break

            current_field.get_regal_target_dict(player, opponent)
            action_count += 1
            if action_count > 100:
                player.show_hand()
                current_field.show_field()
                assert False
        if current_field.check_game_end():
            return
        current_field.end_of_turn(player_num, virtual=True)
        if current_field.check_game_end():
            return
    """
    def expand(self, node, player_num=0):
        field = node.field

        new_choices = node.get_able_action_list()
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])
        # regal_targetsの更新のため
        next_node = None
        if move[0] == 0:
            next_field.end_of_turn(player_num, virtual=True)
            opponent_player_num = 1 - player_num
            opponent = next_field.players[opponent_player_num]
            next_field.untap(opponent_player_num)
            next_field.increment_cost(opponent_player_num)
            next_field.start_of_turn(opponent_player_num, virtual=True)
            next_field.turn_player_num = opponent_player_num
            opponent.draw(opponent.deck,num=1)
            flg = next_field.check_game_end()
            if player_num != self.main_player_num:
                flg = True
            next_node = Node(field=next_field, player_num=opponent_player_num,
                             finite_state_flg=flg, depth=node.depth + 1)
        else:
            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = next_field.check_game_end()
            next_node = Node(field=next_field, player_num=player_num,
                             finite_state_flg=flg, depth=node.depth + 1)
        next_node.parent_node = node
        next_node.field_value = self.state_value(next_field, self.main_player_num)
        node.child_nodes.append((move, next_node))
        return next_node


class Cheating_MO_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__()
        self.name = "Cheating_MO_ISMCTS(iteration={},expand_only)Policy".format(iteration)
        self.iteration = iteration
    """
    def default_policy(self, node, player_num=0):
        sum_of_value = 0
        if node.finite_state_flg:
            return self.state_value(node.field, player_num=self.main_player_num)
        end_count = 0
        current_field = Field_setting.Field(5)
        opponent_player_num = 1 - self.main_player_num
        for i in range(self.default_iteration):
            current_field.set_data(node.field)
            if self.main_player_num == player_num:
                current_field.turn_player_num = player_num
                self.simulate_playout(current_field, player_num=self.main_player_num)
                if current_field.check_game_end():
                    sum_of_value += self.state_value(current_field,self.main_player_num)
                    end_count += 1
                    if end_count >= 5:
                        return sum_of_value / end_count
                    continue
                opponent = current_field.players[opponent_player_num]
                current_field.turn_player_num = opponent_player_num
                current_field.untap(opponent_player_num)
                current_field.increment_cost(opponent_player_num)
                current_field.start_of_turn(opponent_player_num, virtual=True)
                opponent.draw(opponent.deck, 1)
            self.simulate_playout(current_field, player_num=1 - self.main_player_num)
            sum_of_value += self.state_value(current_field, player_num=self.main_player_num)
        return sum_of_value / self.default_iteration
    
    def simulate_playout(self, current_field, player_num=0):
        player = current_field.players[player_num]
        opponent = current_field.players[1 - player_num]
        current_field.get_regal_target_dict(player, opponent)
        action_count = 0
        while True:
            (action_num, card_id, target_id) = self.play_out_policy.decide(player, opponent, current_field)

            end_flg = player.execute_action(current_field, opponent,
                                            action_code=(action_num, card_id, target_id), virtual=True)

            if current_field.check_game_end() or end_flg:
                break

            current_field.get_regal_target_dict(player, opponent)
            action_count += 1
            if action_count > 100:
                player.show_hand()
                current_field.show_field()
                assert False
        if current_field.check_game_end():
            return
        current_field.end_of_turn(player_num, virtual=True)
        if current_field.check_game_end():
            return
    """
    def execute_single_action(self, node, new_choices, child_node_fields, player_num=0):
        field = node.field
        move = random.choice(new_choices)
        next_field = Field_setting.Field(5)
        next_field.set_data(field)
        next_field.get_situation(next_field.players[player_num],
                                 next_field.players[1 - player_num])
        next_node = None
        exist_flg = False
        if move[0] == Action_Code.TURN_END.value:
            next_field.end_of_turn(player_num, virtual=True)
            opponent_player_num = 1 - player_num
            opponent = next_field.players[opponent_player_num]
            next_field.untap(opponent_player_num)
            next_field.increment_cost(opponent_player_num)
            next_field.start_of_turn(opponent_player_num, virtual=True)
            next_field.turn_player_num = opponent_player_num
            opponent.draw(opponent.deck, num= 1)
            flg = next_field.check_game_end()
            if player_num != self.main_player_num:
                flg = True
            next_node = New_Node(field=next_field, player_num=1 - player_num, finite_state_flg=flg,
                                 depth=node.depth + 1)
            for child in node.child_nodes:
                if node.node_id_2_edge_action[id(child)] == Action_Code.TURN_END.value:
                    if child.field.eq(next_field):
                        exist_flg = True
                        break

        else:
            next_field.players[player_num].execute_action(next_field, next_field.players[1 - player_num],
                                                          action_code=move, virtual=True)
            flg = next_field.check_game_end()
            next_node = New_Node(field=next_field, player_num=player_num,
                                 finite_state_flg=flg, depth=node.depth + 1)
            for child_field in child_node_fields:
                if child_field.eq(next_field) and \
                        child_field.players[player_num].eq(next_field.players[player_num]):
                    node.children_moves.remove(move)
                    exist_flg = True
                    break

        return next_node, move, exist_flg


def advanced_state_value(field, player_num):
    if field.check_game_end():
        if field.players[player_num].life <= 0 or field.players[player_num].lib_out_flg:
            return 0
        return 1
    player = field.players[player_num]
    opponent = field.players[1-player_num]
    player_side = field.card_location[player_num]
    opponent_side = field.card_location[1-player_num]
    stats_sum = [0, 0]
    amulet_counter = [0, 0]
    can_attack_to_player_power = [0,0]
    ward_flg = field.check_ward()
    for i, side in enumerate([player_side, opponent_side]):
        for follower in side:
            if follower.card_category == "Creature":
                stats_sum[i] += follower.power*2 + follower.get_current_toughness()

            else:
                if amulet_counter[i] < 3:
                    stats_sum[i] += follower.origin_cost
                else:
                    stats_sum[i] -= 10
                amulet_counter[i] += 1
            stats_sum[i] += (len(side)-amulet_counter[i])*5

    value = 0
    a = 0.0004
    if player.life >= 15:
        if opponent.life <= 10:
            value = (25 - player.life)*(stats_sum[0]-stats_sum[1]) + (20-opponent.life)*10
        else:
            value = (25 - player.life)*(stats_sum[0]-stats_sum[1]) + (20-opponent.life)*5
    else:
        if opponent.life <= 10:
            value = (40 - player.life) * (stats_sum[0] - stats_sum[1]) + (20-opponent.life)*20
        else:
            value = (40 - player.life) * (stats_sum[0] - stats_sum[1]) + (20 - opponent.life)*10

    return 1/(1+np.exp(-value*a))


class Advanced_value_function_A_MCTSPolicy(New_Aggro_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Advanced_value_function_New_A-MCTSPolicy"

    def state_value(self, field, player_num):
        value = advanced_state_value(field, player_num)
        #mylogger.info("state_value:{}".format(value))
        return value


class Advanced_value_function_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "Advanced_value_function_OM-ISMCTSPolicy"

    def state_value(self, field, player_num):
        value = advanced_state_value(field, player_num)
        #mylogger.info("state_value:{}".format(value))
        return value


class Non_Rollout_MCTSPolicy(Flexible_Iteration_MCTSPolicy):
    def __init__(self,iteration=100):
        super().__init__(N=iteration)
        self.name = "NR_MCTS(n={})policy".format(iteration)

    def default_policy(self, node, player_num=0):
        return self.state_value(node.field,player_num)


class Non_Rollout_A_MCTSPolicy(New_Aggro_MCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__()
        self.name = "NR_New_A_MCTS(n={})policy".format(iteration)
        self.iteration = iteration

    def default_policy(self, node, player_num=0):
        return self.state_value(node.field, player_num)


class Non_Rollout_ISMCTSPolicy(Flexible_Iteration_Information_Set_MCTSPolicy):
    def __init__(self,iteration=100):
        super().__init__(N=iteration)
        self.name = "NR_ISMCTS(n={})policy".format(iteration)

    def default_policy(self, node, player_num=0):
        return self.state_value(node.field,player_num)


class Non_Rollout_OM_MCTSPolicy(Opponent_Modeling_MCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__(iteration=iteration)
        self.name = "NR_MO_MCTS(n={})policy".format(iteration)

    def default_policy(self, node, player_num=0):
        return self.state_value(node.field, self.main_player_num)


class Non_Rollout_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self, iteration=100):
        super().__init__(iteration=iteration)
        self.name = "NR_MO_ISMCTS(n={})policy".format(iteration)

    def default_policy(self, node, player_num=0):
        value = self.state_value(node.field, self.main_player_num)
        if node.parent_node.node_id_2_edge_action[id(node)][0] == Action_Code.TURN_END.value:
            if len(node.parent_node.edge_action_2_node_id) > 1:
                value /= 10
        return value


class New_GreedyPolicy(Default_GreedyPolicy):

    def __init__(self):
        self.name = "New_GreedyPolicy(obligated)"
        self.policy_type = 2

    def decide(self, player, opponent, field):
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field, player_num=player.player_num)
        able_actions = starting_node.children_moves
        max_value_action = (0,0,0)
        max_state_value = -np.inf

        if not field.secret:
            mylogger.info("able_actions:{}".format(able_actions))
        for action in able_actions:
            if action == (0,0,0):
                continue
            sim_field = Field_setting.Field(5)
            sim_field.set_data(starting_field)
            sim_player = sim_field.players[player_num]
            sim_opponent = sim_field.players[1-player_num]
            sim_player.execute_action(sim_field, sim_opponent, action_code=action, virtual=True)
            value = self.state_value(sim_field,player_num)
            if not field.secret:
                mylogger.info("action:{},state_value:{:.3f}".format(action,value))
            if value > max_state_value:
                max_value_action = action
                max_state_value = value
                if value == 1:
                    break

        return max_value_action
    

class until_game_end_MCTSPolicy(New_Aggro_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "UGE_MCTSPolicy"

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            current_field = Field_setting.Field(5)
            current_field.set_data(node.field)
            current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
            current_field.players[1].deck.shuffle()
            player = current_field.players[player_num]
            current_field.turn_player_num = player_num
            opponent = current_field.players[1 - player_num]
            opponent_hand = opponent.hand
            opponent_deck = opponent.deck
            hand_len = len(opponent.hand)
            while len(opponent.hand) > 0:
                opponent_deck.append(opponent_hand.pop())
            opponent_deck.shuffle()
            opponent.draw(opponent_deck, hand_len)
            player.policy = self.play_out_policy
            opponent.policy = self.play_out_policy
            while True:
                end_flg = player.decide(player, opponent, current_field, virtual=True)
                if end_flg:
                    break
            if current_field.check_game_end():
                sum_of_value += self.state_value(current_field,player.player_num)
                continue
            current_field.end_of_turn(player_num,True)
            if current_field.check_game_end():
                sum_of_value += self.state_value(current_field,player.player_num)
                continue
            while True:
                current_field.play_turn(opponent.player_num, 0, 0, 0,0, True)
                if current_field.check_game_end():
                    break
                current_field.play_turn(player.player_num, 0, 0, 0, 0, True)
                if current_field.check_game_end():
                    break
            sum_of_value += self.state_value(current_field, player.player_num)
        return sum_of_value / self.default_iteration


class until_game_end_OM_MCTSPolicy(Opponent_Modeling_MCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "UGE_OM_MCTSPolicy"

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            current_field = Field_setting.Field(5)
            current_field.set_data(node.field)
            current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
            current_field.players[1].deck.shuffle()
            player = current_field.players[player_num]
            current_field.turn_player_num = player_num
            opponent = current_field.players[1 - player_num]
            opponent_hand = opponent.hand
            opponent_deck = opponent.deck
            hand_len = len(opponent.hand)
            while len(opponent.hand) > 0:
                opponent_deck.append(opponent_hand.pop())
            opponent_deck.shuffle()
            opponent.draw(opponent_deck, hand_len)
            player.policy = self.play_out_policy
            opponent.policy = self.play_out_policy
            while True:
                end_flg = player.decide(player, opponent, current_field, virtual=True)
                if end_flg:
                    break
            if current_field.check_game_end():
                sum_of_value += self.state_value(current_field,player.player_num)
                continue
            current_field.end_of_turn(player_num,True)
            if current_field.check_game_end():
                sum_of_value += self.state_value(current_field,player.player_num)
                continue
            while True:
                current_field.play_turn(opponent.player_num, 0, 0, 0,0, True)
                if current_field.check_game_end():
                    break
                current_field.play_turn(player.player_num, 0, 0, 0, 0, True)
                if current_field.check_game_end():
                    break
            sum_of_value += self.state_value(current_field, player.player_num)
        return sum_of_value / self.default_iteration


class until_game_end_OM_ISMCTSPolicy(Opponent_Modeling_ISMCTSPolicy):
    def __init__(self):
        super().__init__()
        self.name = "UGE_OM_ISMCTSPolicy"

    def default_policy(self, node, player_num=0):
        if node.finite_state_flg:
            if not node.field.check_game_end():
                current_field = Field_setting.Field(5)
                current_field.set_data(node.field)
                current_field.end_of_turn(player_num, virtual=True)
            return self.state_value(node.field, player_num)
        sum_of_value = 0
        end_flg = False
        for i in range(self.default_iteration):
            current_field = Field_setting.Field(5)
            current_field.set_data(node.field)
            current_field.players[0].deck.shuffle()  # デッキの並びは不明だから
            current_field.players[1].deck.shuffle()
            player = current_field.players[player_num]
            current_field.turn_player_num = player_num
            opponent = current_field.players[1 - player_num]
            opponent_hand = opponent.hand
            opponent_deck = opponent.deck
            hand_len = len(opponent.hand)
            while len(opponent.hand) > 0:
                opponent_deck.append(opponent_hand.pop())
            opponent_deck.shuffle()
            opponent.draw(opponent_deck, hand_len)
            player.policy = self.play_out_policy
            opponent.policy = self.play_out_policy
            while True:
                end_flg = player.decide(player, opponent, current_field, virtual=True)
                if end_flg:
                    break
            if current_field.check_game_end():
                sum_of_value += self.state_value(current_field, player.player_num)
                continue
            current_field.end_of_turn(player_num, True)
            if current_field.check_game_end():
                sum_of_value += self.state_value(current_field, player.player_num)
                continue
            while True:
                current_field.play_turn(opponent.player_num, 0, 0, 0, 0, True)
                if current_field.check_game_end():
                    break
                current_field.play_turn(player.player_num, 0, 0, 0, 0, True)
                if current_field.check_game_end():
                    break
            sum_of_value += self.state_value(current_field, player.player_num)
        return sum_of_value / self.default_iteration

class NN_GreedyPolicy(New_GreedyPolicy):
    def __init__(self,model_name = None):
        short_name = model_name.split(".pth")[0]
        self.name = "NN_GreedyPolicy(model_name={})".format(short_name)
        self.policy_type = 2
        #self.net = Net(10173,10,1)
        self.net = Net(19218, 10, 1)
        self.model_name = PATH
        if model_name is not None:
            self.model_name = 'model/{}'.format(model_name)
        self.net.load_state_dict(torch.load(self.model_name))
        from Network_model import Field_START,LIFE_START
        self.Field_START = Field_START
        self.LIFE_START = LIFE_START


    def decide(self, player, opponent, field):
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field, player_num=player.player_num)
        able_actions = starting_node.children_moves
        max_value_action = (0, 0, 0)
        max_state_value = -np.inf

        if not field.secret:
            mylogger.info("able_actions:{}".format(able_actions))
        for action in able_actions:
            if action == (0, 0, 0):
                continue
            sim_field = Field_setting.Field(5)
            sim_field.set_data(starting_field)
            sim_player = sim_field.players[player_num]
            sim_opponent = sim_field.players[1 - player_num]
            sim_player.execute_action(sim_field, sim_opponent, action_code=action, virtual=True)
            inputs = self.get_data(sim_field,player_num=player.player_num)
            #inputs = torch.Tensor(inputs)
            value = float(self.net(inputs))
            if not field.secret:
                mylogger.info("action:{},state_value:{:.3f}".format(action, value))
            if value > max_state_value:
                max_value_action = action
                max_state_value = value
                if value == 1.0:
                    break

        return max_value_action

    def get_data(self,f,player_num = 0):
        tmp = Game_setting.get_data(f,player_num=player_num)
        return self.single_state_change_to_full(tmp)[0]

    def single_state_change_to_full(self,origin):
        # [card_category,cost,card_id]*9*2 + [card_id,power,toughness,[ability]]*5*2+[life,life,turn]
        # 3*9 + 4*10 + 3 = 27 + 40 +3 = 70
        convert_states = []

        cell = origin
        assert len(cell) == 70, "cell_len:{}".format(len(cell))
        tmp = []
        for i in range(9):
            tmp.extend(list(np.identity(4)[cell[3 * i]]))
            tmp.append(cell[3 * i + 1])
            tmp.extend(list(np.identity(1000)[3 * i + 2 + 500]))
        # 9*(4+1+1000) = 9*1005 = 9045
        for i in range(10):
            j = self.Field_START + 4 * i
            assert j % 4 == 3, "j={}".format(j)
            assert type(cell[j]) == int and type(cell[j + 1]) == int and type(cell[j + 2]) == int \
                   and type(cell[j + 3]) == list, "cell={}".format(cell[j:j + 4])
            tmp.extend(list(np.identity(1000)[cell[j] + 500]))
            tmp.extend([cell[j + 1], cell[j + 2]])
            embed_ability = [int(ability_id in cell[j + 3]) for ability_id in range(1, 16)]
            tmp.extend(embed_ability)
        # 10 *(1000+2+15) = 10170
        # 9045 + 10170 = 19215
        assert len(cell[self.LIFE_START:]) == 3, "data:{}".format(cell[self.LIFE_START:])
        tmp.extend(cell[self.LIFE_START:])
        convert_states.append(torch.Tensor(tmp))

        return convert_states


class NN_A_MCTSPolicy(Flexible_Iteration_Aggro_MCTSPolicy,NN_GreedyPolicy):
    def __init__(self,model_name = None):
        super().__init__()
        short_name = model_name.split(".pth")[0]
        self.name = "NN-A-MCTS(model_name={})Policy".format(short_name)
        self.net = Net(19218, 10, 1)
        self.model_name = PATH
        if model_name is not None:
            self.model_name = 'model/{}'.format(model_name)
        self.net.load_state_dict(torch.load(self.model_name))
        from Network_model import Field_START,LIFE_START
        self.Field_START = Field_START
        self.LIFE_START = LIFE_START

    def state_value(self, field, player_num):
        inputs = self.get_data(field, player_num=player_num)
        value = float(self.net(inputs))
        return value

    def get_data(self,field,player_num = 0):
        return super().get_data(field,player_num = player_num)

class NN_Non_Rollout_MCTSPolicy(Non_Rollout_A_MCTSPolicy,NN_GreedyPolicy):
    def __init__(self,model_name = None):
        super().__init__()
        short_name = model_name.split(".pth")[0]
        self.name = "NN_Non-Rollout_MCTS(model_name={})Policy".format(short_name)
        self.net = Net(19218, 10, 1)
        self.model_name = PATH
        if model_name is not None:
            self.model_name = 'model/{}'.format(model_name)
        self.net.load_state_dict(torch.load(self.model_name))
        self.net.eval()
        from Network_model import Field_START,LIFE_START
        self.Field_START = Field_START
        self.LIFE_START = LIFE_START

    def state_value(self, field, player_num):
        inputs = self.get_data(field, player_num=player_num)
        value = float(self.net(inputs))
        return value

    def get_data(self,field,player_num = 0):
        return super().get_data(field,player_num = player_num)


class Dual_NN_Non_Rollout_OM_ISMCTSPolicy(Non_Rollout_OM_ISMCTSPolicy):
    def __init__(self,model_name = None,origin_model = None):
        super().__init__()

        from Dual_Network_model import Dual_Net
        from Network_model import state_change_to_full
        self.net = None
        self.model_name = PATH
        if model_name is not None:
            short_name = model_name.split(".pth")[0]
            self.name = "DN_NR_OMISMCTS(model_name={})Policy".format(short_name)
            self.net = Dual_Net(19218, 100, 1)
            if torch.cuda.is_available():
                self.net = self.net.cuda()
            self.model_name = 'model/{}'.format(model_name)
            self.net.load_state_dict(torch.load(self.model_name))
            self.net.eval()
        else:
            if origin_model is not None:
                self.net = origin_model
            else:
                assert False,"non-model error"

        self.state_convertor = state_change_to_full

    def default_policy(self, node, player_num=0):
        value = self.state_value(node.field, self.main_player_num)
        if node.parent_node.node_id_2_edge_action[id(node)][0] == Action_Code.TURN_END.value:
            if len(node.parent_node.edge_action_2_node_id) > 1:
                value /= 10
        return value

    def state_value(self, field, player_num):
        if field.check_game_end():
            player = field.players[0]
            return 2*int(player.life > 0 and not player.lib_out_flg) - 1
        inputs = self.get_data(field, player_num=player_num)
        _, value = self.net(inputs)
        return float(value[0])

    def get_data(self,f,player_num = 0):
        tmp = Game_setting.get_data(f,player_num=player_num)
        states = [State_tuple(tmp)]*2
        states = self.state_convertor(states,cuda=self.cuda)
        states = torch.stack(states, dim=0)
        return states


class New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(Non_Rollout_OM_ISMCTSPolicy):
    def __init__(self,model_name = None,origin_model = None, cuda=False):
        super().__init__()

        from Embedd_Network_model import New_Dual_Net, Detailed_State_data_2_Tensor
        self.net = None
        self.model_name = PATH
        if model_name is not None:
            short_name = model_name.split(".pth")[0]
            self.name = "N_DN_NR_OMISMCTS(model_name={})Policy".format(short_name)
            self.net = New_Dual_Net(100)
            if torch.cuda.is_available() and cuda:
                self.net = self.net.cuda()
            self.model_name = 'model/{}'.format(model_name)
            self.net.load_state_dict(torch.load(self.model_name))
            self.net.eval()
        else:
            if origin_model is not None:
                self.net = origin_model
                self.net.eval()
            else:
                assert False,"non-model error"

        self.state_convertor = Detailed_State_data_2_Tensor
        self.policy_type = 3
        self.cuda = cuda

    def decide(self, player, opponent, field):
        action = super().decide(player,opponent, field)

        if not field.secret and self.prev_node is not None:
            pai = self.prev_node.pai
            if type(pai) != list:
                mylogger.info("action probability distribution")
                #print(pai)
                for action_code_id in range(len(pai)):
                    if pai[action_code_id] != 0:
                        txt = ""
                        if action_code_id == 0:
                            txt = "{:<40}".format(Action_Code.TURN_END.name)

                        elif action_code_id >= 1 and action_code_id <= 9:
                            txt = "{},{}".format(Action_Code.PLAY_CARD.name, player.hand[action_code_id-1].name)

                        elif action_code_id >= 10 and action_code_id <= 34:
                            attacking_id  = (action_code_id - 10) // 5
                            attacked_id = action_code_id % 5
                            txt = "{},{},{}".format(Action_Code.ATTACK_TO_FOLLOWER.name,
                                                 field.card_location[player.player_num][attacking_id].name,
                                                 field.card_location[opponent.player_num][attacked_id].name)
                        elif action_code_id >= 35 and action_code_id <= 39:
                            attacking_id = action_code_id - 35
                            txt = "{},{}".format(Action_Code.ATTACK_TO_PLAYER.name,
                                                 field.card_location[player.player_num][attacking_id].name)
                        elif action_code_id <= 44:
                            evolving_id = action_code_id - 40
                            txt = "{},{}".format(Action_Code.EVOLVE.name,
                                                 field.card_location[player.player_num][evolving_id].name)
                        else:
                            assert False

                        mylogger.info("{:<60}:{:.3%}".format(txt, pai[action_code_id]))

            #self.prev_node.print_tree(single=True)
        return action

    def uct_search(self, player, opponent, field,use_existed_node=False):
        if not use_existed_node:
            current_field = Field_setting.Field(5)
            current_field.set_data(field)
            current_field.get_regal_target_dict(current_field.players[player.player_num],
                                                 current_field.players[opponent.player_num])
            self.current_node = New_Node(field=current_field, player_num=player.player_num,root=True)
            self.state_value(self.current_node, player.player_num)

        action = super().uct_search(player, opponent, field, use_existed_node=True)

        return action

    def default_policy(self, node, player_num=0):
        value = self.state_value(node, player_num)
        if player_num != self.main_player_num:
            value = -value
        if node.parent_node.node_id_2_edge_action[id(node)][0] == Action_Code.TURN_END.value:
            if len(node.parent_node.edge_action_2_node_id) > 1:
                value = -1.0
        assert abs(value) <= 1.0,"value:{}".format(value)
        return value

    def state_value(self, node, player_num):
        field = node.field
        if field.check_game_end():
            player = field.players[self.main_player_num]
            return 2*int(player.life > 0 and not player.lib_out_flg) - 1
        if node.state_value is not None:
            return node.state_value

        states = self.get_data(field, player_num=player_num)

        states['detailed_action_codes'] = Embedd_Network_model.Detailed_action_code_2_Tensor\
            ([field.get_detailed_action_code(field.players[player_num])])
        pai, value = self.net(states)
        node.state_value = float(value[0])
        node.pai = pai[0]

        return float(value[0])

    def get_data(self,f,player_num = 0):
        tmp = Game_setting.get_data(f,player_num=player_num)
        states = [tmp]
        states = self.state_convertor(states)

        return states

    def execute_best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            if action == (0,0,0) and len(node.edge_action_2_node_id) > 1:
                action_uct_values[action].append(-100)
                action_2_node[action].append(children[i])
                continue

            if children[i].finite_state_flg and children[i].field.check_game_end():
                player = children[i].field.players[player_num]
                if player.life <= 0 or player.lib_out_flg:
                    action_uct_values[action].append(-100)
                else:
                    action_uct_values[action].append(100)
            else:
                action_uct_values[action].append(children[i].visit_num)
            action_2_node[action].append(children[i])
        #print(action_uct_values)
        max_value = -1000
        max_value_action = (0, 0, 0)
        for key in list(action_uct_values.keys()):
            sum_of_value = sum(action_uct_values[key])
            if max_value < sum_of_value:
                max_value = sum_of_value
                max_value_action = key
        if max_value_action not in action_2_node:
            return node, (Action_Code.ERROR.value,0,0)

        max_value_node = random.choice(action_2_node[max_value_action])
        return max_value_node, max_value_action

    def best(self, node, player_num=0):
        children = node.child_nodes
        action_uct_values = {}
        action_2_node = {}
        for i in range(len(children)):
            action = node.node_id_2_edge_action[id(children[i])]
            if action not in action_uct_values:
                action_uct_values[action] = []
            if action not in action_2_node:
                action_2_node[action] = []
            if action == (0,0,0) and len(node.edge_action_2_node_id) > 1:
                value = -1
            else:
                value = self.uct(children[i], node, player_num=player_num)
            action_uct_values[action].append(value)
            action_2_node[action].append(children[i])
        max_value = None
        max_value_action = None
        for key in list(action_uct_values.keys()):
            values = action_uct_values[key]
            weights = [cell.visit_num for cell in action_2_node[key]]
            visit_num_sum = sum(weights)
            mean_value = sum([values[node_id] * (weights[node_id] / visit_num_sum) for node_id in range(len(values))])
            if max_value is None or mean_value > max_value:
                max_value = mean_value
                max_value_action = key

        max_value_node = random.choices(action_2_node[max_value_action])[0]
        return max_value_node, max_value_action


    def uct(self, child_node, node, player_num=0):
        #Q(s,a) + C(s)P(a|s)(√N(s)/(1+N(s,a)))
        #C(s) = log((1+N(s)+c_base)/c_base) + c_init
        over_all_n = node.visit_num #N(s)
        n = child_node.visit_num #N(s,a)
        w = child_node.value #Q(s,a)
        c_base = 1
        c_init = 1
        c = np.log((1+over_all_n+c_base)/c_base) + c_init
        epsilon = EPSILON
        exploitation_value = (2*int( player_num == self.main_player_num)-1)*(w / (n + epsilon))
        action_code = node.node_id_2_edge_action[id(child_node)]#(action_category,play_id,target)
        action_id = Action_Code.TURN_END.value
        if action_code[0] == Action_Code.PLAY_CARD.value:
            action_id = 1 + action_code[1]
        elif action_code[0] == Action_Code.ATTACK_TO_FOLLOWER.value:
            action_id = 10 + action_code[1]*5 + action_code[2]
        elif action_code[0] == Action_Code.ATTACK_TO_PLAYER.value:
            action_id = 35 + action_code[1]
        elif action_code[0] == Action_Code.EVOLVE.value:
            action_id = 40 + action_code[1]

        probability = node.pai[action_id]
        exploration_value = c * probability * (np.sqrt(over_all_n) / (1 + n))

        value = exploitation_value + exploration_value
        child_node.uct = value

        return value


class Dual_NN_GreedyPolicy(New_GreedyPolicy):
    def __init__(self, model_name=None, origin_model=None):
        super().__init__()
        self.policy_type = 2
        from Embedd_Network_model import New_Dual_Net, Detailed_State_data_2_Tensor
        self.net = None
        self.model_name = PATH
        if model_name is not None:
            short_name = model_name.split(".pth")[0]
            self.name = "DN_Greedy(model_name={})Policy".format(short_name)
            self.net = New_Dual_Net(100)
            self.model_name = 'model/{}'.format(model_name)
            self.net.load_state_dict(torch.load(self.model_name))
            self.net.eval()
        else:
            if origin_model is not None:
                self.net = origin_model
            else:
                assert False, "non-model error"

        self.state_convertor = Detailed_State_data_2_Tensor

    def decide(self, player, opponent, field):
        player_num = player.player_num
        starting_field = Field_setting.Field(5)
        starting_field.set_data(field)
        starting_field.get_regal_target_dict(starting_field.players[player.player_num],
                                             starting_field.players[opponent.player_num])
        starting_node = Node(field=starting_field, player_num=player.player_num)
        able_actions = starting_node.children_moves
        max_value_action = (0, 0, 0)
        max_state_value = -np.inf

        if not field.secret:
            mylogger.info("able_actions:{}".format(able_actions))
        for action in able_actions:
            if action == (0, 0, 0):
                continue
            sim_field = Field_setting.Field(5)
            sim_field.set_data(starting_field)
            sim_player = sim_field.players[player_num]
            sim_opponent = sim_field.players[1 - player_num]
            sim_player.execute_action(sim_field, sim_opponent, action_code=action, virtual=True)
            value = self.state_value(sim_field, player_num)
            if not field.secret:
                mylogger.info("action:{},state_value:{:.3f}".format(action, value))
            if value > max_state_value:
                max_value_action = action
                max_state_value = value
                if value == 1.0:
                    break

        return max_value_action

    def state_value(self, field, player_num):

        if field.check_game_end():
            player = field.players[player_num]
            return 2*int(player.life > 0 and not player.lib_out_flg) - 1
        states = self.get_data(field, player_num=player_num)

        states['detailed_action_codes'] = Embedd_Network_model.Detailed_action_code_2_Tensor\
            ([field.get_detailed_action_code(field.players[player_num])])
        _, value = self.net(states)

        return float(value[0])

    def get_data(self,f,player_num = 0):
        tmp = Game_setting.get_data(f,player_num=player_num)
        #states = [tmp]*2
        states = [tmp]
        states = self.state_convertor(states)

        return states

