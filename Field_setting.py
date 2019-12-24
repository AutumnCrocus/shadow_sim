import numpy as np
import random
import math
import copy
from card_setting import *
import collections
from collections import deque
import itertools
from my_moduler import get_module_logger

mylogger = get_module_logger(__name__)
from util_ability import *
# mylogger = get_module_logger('mylogger')
from my_enum import *
import time
import os


class Field:
    def __init__(self, max_field_num):
        self.card_location = [[], []]
        self.card_num = [0, 0]
        self.cost = [0, 0]
        self.max_cost = 10
        self.remain_cost = [0, 0]
        self.max_field_num = max_field_num
        self.turn_end = False
        self.graveyard = Graveyard()
        self.play_cards = Play_Cards()
        self.drawn_cards = Drawn_Cards()
        self.players = [None, None]
        self.evo_point = [2, 3]
        self.able_to_evo_turn = [5, 4]
        self.current_turn = [0, 0]
        self.evo_flg = False
        self.ex_turn_count = [0, 0]
        self.turn_player_num = 0
        self.stack = deque()
        self.chain_num = 0
        self.players_play_num = 0
        self.player_ability = [[], []]
        self.state_log = deque()
        self.stack_num = 0
        self.time = time.time()
        self.secret = False

    def eq(self, other):
        if type(self) != type(other):
            assert False, "NoImplemented"
        if len(self.card_location[0]) != len(other.card_location[0]): return False
        if len(self.card_location[1]) != len(other.card_location[1]): return False
        for i in range(2):
            observable = self.get_observable_data(player_num=i)
            other_observable = other.get_observable_data(player_num=i)
            for key in list(observable.keys()):
                for second_key in list(observable[key].keys()):
                    if observable[key][second_key] != other_observable[key][second_key]:
                        return False
        #仮実装
        self.play_cards.play_cards_set()
        other.play_cards.play_cards_set()
        for i in range(2):
            for j in range(len(self.card_location[i])):
                first_card = self.card_location[i][j]
                second_card = other.card_location[i][j]
                if not first_card.eq(second_card):
                    return False
            for cost_key in sorted(list(self.play_cards.name_list[i].keys())):
                if cost_key not in other.play_cards.name_list[i]:
                    return False
                for category_key in sorted(list(self.play_cards.name_list[i][cost_key].keys())):
                    if category_key not in other.play_cards.name_list[i][cost_key]:
                        return False
                    for name_key in sorted(list(self.play_cards.name_list[i][cost_key][category_key].keys())):
                        if name_key not in other.play_cards.name_list[i][cost_key][category_key]:
                            return False
                        if other.play_cards.name_list[i][cost_key][category_key][name_key] !=\
                            self.play_cards.name_list[i][cost_key][category_key][name_key]:
                            return False


        return True

    def set_data(self, field):
        # self.card_location=copy.deepcopy(field.card_location)
        assert len(self.card_location[0]) <= self.max_field_num, "card_num:{}".format(len(self.card_location[0]))
        assert len(self.card_location[1]) <= self.max_field_num, "card_num:{}".format(len(self.card_location[1]))
        self.card_location[0].clear()
        self.card_location[1].clear()
        for i in range(2):
            for card in field.card_location[i]:
                self.card_location[i].append(card.get_copy())

        self.card_num = field.card_num[:]
        self.cost = field.cost[:]
        self.remain_cost = field.remain_cost[:]
        self.turn_end = field.turn_end
        self.graveyard = copy.deepcopy(field.graveyard)
        self.play_cards = copy.deepcopy(field.play_cards)
        self.drawn_cards = copy.deepcopy(field.drawn_cards)
        self.players[0] = field.players[0].get_copy(field)
        self.players[1] = field.players[1].get_copy(field)
        self.players[0].field = self
        self.players[1].field = self
        self.update_hand_cost(player_num=0)
        self.update_hand_cost(player_num=1)
        self.evo_point = field.evo_point[:]
        self.current_turn = field.current_turn[:]
        self.evo_flg = field.evo_flg
        self.ex_turn_count = field.ex_turn_count[:]
        self.turn_player_num = int(field.turn_player_num)
        self.players_play_num = int(field.players_play_num)
        if self.player_ability[0] != []:
            field.player_ability[0] = copy.deepcopy(self.player_ability[0])
        if self.player_ability[1] != []:
            field.player_ability[1] = copy.deepcopy(self.player_ability[1])
        self.reset_time_stamp()

    def solve_lastword_ability(self, virtual=False, player_num=0):
        while len(self.stack) > 0:
            (ability, player_num, itself) = self.stack.pop()
            if not virtual:
                mylogger.info("{}'s lastword ability actived".format(itself.name))
            ability(self, self.players[player_num], self.players[1 - player_num], virtual, None, itself)
            if self.check_game_end():
                break

            # if virtual==False:
            #    mylogger.info("rest_num={}".format(len(self.stack)))

    def solve_field_trigger_ability(self, virtual=False, player_num=0):
        ability_list = deque()
        # ability_list.apendleft()
        if len(self.state_log) == 0: return
        index = len(self.state_log) - 1
        while index >= 0:
            target_state_log = self.state_log[index]
            for i in range(2):
                side_id = (i + player_num) % 2
                for player_ability in self.player_ability[side_id]:
                    player_ability(self, self.players[side_id], virtual, state_log=target_state_log)
                side = self.card_location[side_id]
                location_id = len(side) - 1
                while location_id >= 0:
                    if side[location_id].time_stamp < target_state_log[-1]:
                        for ability in side[location_id].trigger_ability:
                            argument = [self, self.players[side_id], self.players[1 - side_id], virtual, None,
                                        side[location_id], target_state_log]
                            ability_list.appendleft((ability, argument))
                            # ability(self,self.players[side_id],self.players[1-side_id],virtual,None,thing,state_log=target_state_log)
                    location_id -= 1
            index -= 1

        self.state_log.clear()
        while len(ability_list) > 0:
            tmp_ability_pair = ability_list.popleft()
            ability = tmp_ability_pair[0]
            argument = tmp_ability_pair[1]
            ability(argument[0], argument[1], argument[2], argument[3], argument[4], argument[5], state_log=argument[6])

    def ability_resolution(self, virtual=False, player_num=0):
        chain_len = 0

        self.check_active_ability()
        while len(self.stack) > 0 or len(self.state_log) > 0:
            self.check_death(player_num, virtual=virtual)
            self.solve_lastword_ability(virtual=virtual, player_num=player_num)
            self.check_death(player_num, virtual=virtual)
            self.solve_field_trigger_ability(virtual=virtual, player_num=player_num)
            self.check_death(player_num, virtual=virtual)
            chain_len += 1
            assert chain_len < 100, "infinite_chain_error"

    def check_active_ability(self):
        for i in range(2):
            for card in self.card_location[i]:
                if card.have_active_ability == True:
                    if card.active_ability_check_func(self.players[i]) == True:
                        card.get_active_ability()
                    else:
                        card.lose_active_ability()

    def get_observable_data(self, player_num=0):
        observable_data_dict = {"player": {}, "opponent": {}}
        for key in list(observable_data_dict.keys()):
            player_id = (1 - 2 * int(key != "player")) * player_num + int(key != "player")
            observable_data_dict[key]["leader_class"] = self.players[player_id].deck.leader_class.name
            observable_data_dict[key]["life"] = self.players[player_id].life
            observable_data_dict[key]["max_life"] = self.players[player_id].max_life
            observable_data_dict[key]["hand_len"] = len(self.players[player_id].hand)
            observable_data_dict[key]["deck_len"] = len(self.players[player_id].deck.deck)
            observable_data_dict[key]["shadows"] = self.graveyard.shadows[player_id]
            observable_data_dict[key]["pp/max_pp"] = (self.remain_cost[player_id], self.cost[player_id])
            observable_data_dict[key]["evo_point"] = self.evo_point[player_id]

        return observable_data_dict

    def reset_time_stamp(self):
        self.stack.clear()
        self.stack_num = 0
        for i in range(2):
            for card in self.card_location[i]:
                card.time_stamp = 0

    def restore_player_life(self, player=None, num=0, virtual=False, at_once=False):
        tmp = num
        if player.max_life - player.life < tmp:
            tmp = player.max_life - player.life
        player.life += tmp
        if virtual == False:
            mylogger.info("Player {} restore {} life".format(player.player_num + 1, tmp))
        if not at_once:
            self.stack_num += 1
            self.state_log.append([State_Code.RESTORE_PLAYER_LIFE.value, player.player_num, player.field.stack_num])

    def restore_follower_toughness(self, follower=None, num=0, virtual=False, at_once=False):
        side_id = 0
        if follower in self.card_location[0]:
            side_id = 0
        elif follower in self.card_location[1]:
            side_id = 1
        else:
            assert False, "follower does not exist!"
        amount = follower.restore_toughness(num)
        if virtual == False:
            mylogger.info("{} restore {} life".format(follower.name, amount))
        if not at_once:
            self.stack_num += 1
            self.state_log.append([State_Code.RESTORE_FOLLOWER_TOUGHNESS.value, side_id, self.stack_num])

    def gain_max_pp(self, player_num=0, num=0, virtual=False):
        if self.cost[player_num] < self.max_cost:
            if not virtual:
                mylogger.info("Player {} gain {} max PP".format(player_num + 1, num))
            self.cost[player_num] += 1

    def resotre_pp(self, player_num=0, num=0, virtual=False):
        value = int(self.remain_cost[player_num])
        self.remain_cost[player_num] += num
        self.remain_cost[player_num] = min(self.remain_cost[player_num], self.cost[player_num])
        value = self.cost[player_num] - value
        if not virtual:
            mylogger.info("Player {} gain {} PP".format(player_num + 1, value))

    def check_death(self, player_num=0, virtual=False):
        for j in range(2):
            i = 0
            while i < len(self.card_location[(player_num + j) % 2]):
                card = self.card_location[(player_num + j) % 2][i]
                if not card.is_in_field or card.is_in_graveyard:
                    self.remove_card([(player_num + j) % 2, i], virtual=virtual)
                elif card.card_category == "Creature":
                    assert card.get_current_toughness() > 0, "minus-toughness_error:{}".format(
                        card.get_current_toughness())
                    i += 1
                else:
                    i += 1
    def append_played_turn(self,card_name=None):
        assert card_name is not None
        if card_name not in self.play_cards.played_turn_dict[self.turn_player_num]:
            self.play_cards.played_turn_dict[self.turn_player_num][card_name] = \
                [self.current_turn[self.turn_player_num]]
        else:
            self.play_cards.played_turn_dict[self.turn_player_num][card_name].append(\
                self.current_turn[self.turn_player_num])

    def play_creature(self, hand, card_id, player_num, player, opponent, virtual=False, target=None):
        if self.card_num[player_num] < self.max_field_num:
            tmp = hand.pop(card_id)
            self.stack_num += 1
            self.state_log.append(
                [State_Code.PLAY.value, (player_num, tmp.card_category, tmp.card_id), self.stack_num])  # 1はプレイ
            tmp.is_in_field = True
            tmp.is_tapped = True
            self.set_card(tmp, player_num, virtual=virtual)
            if tmp.fanfare_ability != None:
                tmp.fanfare_ability(self, player, opponent, virtual, target, tmp)
            self.play_cards.append(tmp.card_category, tmp.card_id, player_num)
            self.append_played_turn(card_name=tmp.name)
            self.check_death(player_num=player_num, virtual=virtual)
        else:
            self.players[player_num].show_hand()
            mylogger.info("card_id:{}".format(card_id))
            self.show_field()
            raise Exception('field is full!\n')

    def play_spell(self, hand, card_id, player_num, player, opponent, virtual=False, target=None):
        tmp = hand.pop(card_id)
        self.stack_num += 1
        self.state_log.append([State_Code.PLAY.value, (player_num, tmp.card_category, tmp.card_id), self.stack_num])
        self.spell_boost(player.player_num)
        for ability in tmp.triggered_ability:
            ability(self, player, opponent, virtual, target, tmp)
        tmp.is_in_graveyard = True
        self.graveyard.append(tmp.card_category, tmp.card_id, player_num)
        self.play_cards.append(tmp.card_category, tmp.card_id, player_num)
        self.append_played_turn(card_name=tmp.name)
        self.check_death(player_num=player_num, virtual=virtual)

    def play_amulet(self, hand, card_id, player_num, player, opponent, virtual=False, target=None):
        if self.card_num[player_num] < self.max_field_num:
            tmp = hand.pop(card_id)
            self.stack_num += 1
            self.state_log.append([State_Code.PLAY.value, (player_num, tmp.card_category, tmp.card_id), self.stack_num])

            tmp.is_in_field = True
            self.set_card(tmp, player_num, virtual=virtual)
            if tmp.fanfare_ability != None:
                tmp.fanfare_ability(self, player, opponent, virtual, target, tmp)
            self.play_cards.append(tmp.card_category, tmp.card_id, player_num)
            self.append_played_turn(card_name=tmp.name)
            self.check_death(player_num=player_num, virtual=virtual)
        else:
            self.players[player_num].show_hand()
            mylogger.info("card_id:{}".format(card_id))
            self.show_field()
            raise Exception('field is full!\n')

    def spell_boost(self, player_num):
        hand = self.players[player_num].hand
        for card in hand:
            if card.card_class.name == "RUNE" and card.spell_boost != None:
                card.spell_boost += 1
                # if card.cost_down==True:
                #    card.cost=max(0,card.origin_cost-card.spell_boost)

    def play_as_other_card(self, hand, card_id, player_num, virtual=False, target=None):
        player = self.players[player_num]
        opponent = self.players[1 - player_num]
        play_card = player.hand[card_id]
        if play_card.have_accelerate and play_card.active_accelerate_code[0]:
            if not virtual: mylogger.info("Accelerate")
            play_card = self.players[player_num].hand.pop(card_id)
            assert play_card.active_accelerate_code[1] in play_card.accelerate_card_id
            new_card_id = play_card.accelerate_card_id[play_card.active_accelerate_code[1]]
            new_card = Spell(new_card_id)
            self.stack_num += 1
            self.state_log.append([State_Code.PLAY.value, (player_num, "Spell", new_card_id), self.stack_num])
            self.spell_boost(player_num)
            for ability in new_card.triggered_ability:
                ability(self, player, opponent, virtual, target, new_card)
            self.append_played_turn(card_name = play_card.name)
            new_card.is_in_graveyard = True
            self.graveyard.append(new_card.card_category, new_card_id, player_num)

    def set_card(self, card, player_num, virtual=False):
        if len(self.card_location[player_num]) < self.max_field_num:
            self.stack_num += 1
            self.state_log.append([State_Code.SET.value, (player_num, card.card_category, card.card_id, id(card)),
                                   self.stack_num])  # 2は場に出たとき
            self.card_location[player_num].append(card)
            self.card_num[player_num] += 1
            card.is_tapped = True
            card.is_in_field = True

            card.time_stamp = self.stack_num
        else:
            if not virtual:
                mylogger.info("{} is vanished".format(card.name))

    def remove_card(self, location, virtual=False, by_effects=False):
        assert self.card_location[location[0]][location[1]] is not None
        tmp = self.card_location[location[0]][location[1]]

        if KeywordAbility.BANISH_WHEN_LEAVES.value in tmp.ability:
            self.banish_card(location, virtual=virtual)
            return
        tmp = self.card_location[location[0]].pop(location[1])
        self.card_num[location[0]] -= 1
        self.stack_num += 1
        self.state_log.append(
            [State_Code.DESTROYED.value, (location[0], tmp.card_category, tmp.card_id), self.stack_num])  # 3は破壊されたとき
        if not virtual:
            if tmp.card_category == "Creature":
                mylogger.info("Player {}'s {}(location_id={}) is dead".format(location[0] + 1, tmp.name, location[1]))
            else:
                mylogger.info("Player {}'s {} is broken".format(location[0] + 1,
                                                                tmp.name))
        for i in range(len(tmp.lastword_ability)):
            self.stack.appendleft((tmp.lastword_ability[i], location[0], copy.deepcopy(tmp)))
        tmp.is_in_field = False
        tmp.is_in_graveyard = True
        self.graveyard.append(tmp.card_category, tmp.card_id, location[0])

    def return_card_to_hand(self, target_location, virtual=False):
        assert len(self.card_location[target_location[0]]) > target_location[1]
        tmp = self.card_location[target_location[0]][target_location[1]]

        if KeywordAbility.BANISH_WHEN_LEAVES.value in tmp.ability:
            self.banish_card(target_location, virtual=virtual)
            return
        self.card_num[target_location[0]] -= 1
        card_id = tmp.card_id
        card_category = tmp.card_category
        if virtual == False:
            mylogger.info("Player {}'s {} return to hand".format(target_location[0] + 1,
                                                                 tmp.name))
        del self.card_location[target_location[0]][target_location[1]]

        card = None
        if card_category == "Creature":
            card = Creature(card_id)
        elif card_category == "Amulet":
            card = Amulet(card_id)

        if card != None:
            self.players[target_location[0]].hand.append(card)

    def banish_card(self, location, virtual=False):
        assert self.card_location[location[0]][location[1]] is not None
        if not virtual:
            mylogger.info(
                "{}(location_id={}) is banished".format(self.card_location[location[0]][location[1]].name, location[1]))
        del self.card_location[location[0]][location[1]]
        self.card_num[location[0]] -= 1

    def transform_card(self, location, card=None, virtual=False):
        assert self.card_location[location[0]][location[1]] is not None and card is None
        if not virtual:
            mylogger.info(
                "{}(location_id={}) is transformed into {}".format(self.card_location[location[0]][location[1]].name,
                                                                   location[1], card.name))
        self.card_location[location[0]][location[1]] = card

    def show_field(self):
        for i in range(2):
            print("player", i + 1, "'s field")
            for j in range(len(self.card_location[i])):
                print(j, ": ", self.card_location[i][j])
            print("\n")

    def attack_to_follower(self, attack, defence, field, virtual=False):

        assert attack[1] < len(self.card_location[attack[0]]) and defence[1] < len(self.card_location[defence[0]])
        attacking_follower = self.card_location[attack[0]][attack[1]]
        if KeywordAbility.AMBUSH.value in attacking_follower.ability:
            attacking_follower.ability.remove(KeywordAbility.AMBUSH.value)
        defencing_follower = self.card_location[defence[0]][defence[1]]
        assert attacking_follower.can_attack_to_follower() and defencing_follower.can_be_attacked(), \
            "attack:{} defence:{}".format(attacking_follower.can_attack_to_follower(),
                                          defencing_follower.can_be_attacked())
        attacking_follower.current_attack_num += 1
        if not virtual:
            mylogger.info("Player {}'s {} attacks Player {}'s {}".format(attack[0] + 1, attacking_follower.name
                                                                         , defence[0] + 1, defencing_follower.name))

        for ability in attacking_follower.in_battle_ability:
            ability(self, self.players[attack[0]], self.players[defence[0]], attacking_follower, defencing_follower,
                    situation_num=[0, 1, 3], virtual=virtual)
        for ability in defencing_follower.in_battle_ability:
            ability(self, self.players[defence[0]], self.players[attack[0]], defencing_follower, attacking_follower,
                    situation_num=[3], virtual=virtual)
        self.stack_num += 1
        self.state_log.append([State_Code.ATTACK_TO_FOLLOWER.value, attack[0], attacking_follower, defencing_follower,
                               self.stack_num])  # 5はフォロワーに攻撃したとき

        self.check_death(player_num=attack[0], virtual=virtual)
        self.ability_resolution(virtual=virtual, player_num=attack[0])
        if not attacking_follower.is_in_field or not defencing_follower.is_in_field:
            return
        amount = defencing_follower.get_damage(attacking_follower.power)
        attacking_follower.get_damage(defencing_follower.power)
        if KeywordAbility.DRAIN.value in attacking_follower.ability:
            restore_player_life(self.players[attack[0]], virtual, num=amount)
        if defencing_follower.is_in_field:
            if KeywordAbility.BANE.value in attacking_follower.ability and \
                    KeywordAbility.CANT_BE_DESTROYED_BY_EFFECTS.value not in defencing_follower.ability:  # 必殺効果処理
                new_def_index = [defence[0], self.card_location[defence[0]].index(defencing_follower)]
                self.remove_card(new_def_index, virtual)
        if attacking_follower.is_in_field:
            if KeywordAbility.BANE.value in defencing_follower.ability and \
                    KeywordAbility.CANT_BE_DESTROYED_BY_EFFECTS.value not in attacking_follower.ability:
                new_atk_index = [attack[0], self.card_location[attack[0]].index(attacking_follower)]
                self.remove_card(new_atk_index, virtual)
        # self.solve_lastword_ability(virtual=virtual,player_num=attack[0])
        self.ability_resolution(virtual=virtual, player_num=attack[0])

    def attack_to_player(self, attacker, defence_player, visible=False, virtual=False):
        attacking_follower = self.card_location[attacker[0]][attacker[1]]
        if KeywordAbility.AMBUSH.value in attacking_follower.ability:
            attacking_follower.ability.remove(KeywordAbility.AMBUSH.value)
        assert attacking_follower.can_attack_to_player()
        attacking_follower.current_attack_num += 1
        for ability in attacking_follower.in_battle_ability:
            ability(self, self.players[attacker[0]], defence_player, attacking_follower, defence_player,
                    situation_num=[0, 2], virtual=virtual)
        self.stack_num += 1
        self.state_log.append(
            [State_Code.ATTACK_TO_PLAYER.value, attacker[0], attacking_follower, self.stack_num])  # 5はプレイヤーに攻撃したとき
        self.check_death(player_num=attacker[0], virtual=virtual)
        self.ability_resolution(virtual=virtual, player_num=attacker[0])
        if not attacking_follower.is_in_field:
            return
        if visible:
            print("Player", attacker[0] + 1, "'s", attacking_follower.name,
                  "attacks directly Player", defence_player.player_num + 1)
        if not virtual:
            mylogger.info("Player {}'s {} attacks directly Player {}".format(attacker[0] + 1, attacking_follower.name
                                                                             , 2 - attacker[0]))
        amount = defence_player.get_damage(attacking_follower.power)
        if KeywordAbility.DRAIN.value in attacking_follower.ability:
            restore_player_life(self.players[attacker[0]], virtual, num=amount)

        # self.solve_lastword_ability(virtual=virtual,player_num=attacker[0])
        # self.ability_resolution(virtual=virtual,player_num=attacker[0])
        if visible:
            print("Player", defence_player.player_num + 1, "life: ", defence_player.life)

    def start_of_turn(self, player_num, virtual=False):
        self.state_log.clear()
        self.reset_time_stamp()
        self.stack_num += 1
        self.state_log.append([State_Code.START_OF_TURN.value, player_num, self.stack_num])
        self.ability_resolution(virtual=virtual, player_num=player_num)
        i = 0
        while i < len(self.card_location[player_num]):
            thing = self.card_location[player_num][i]
            if thing.turn_start_ability != []:
                for ability in thing.turn_start_ability:
                    # if not virtual:
                    #    mylogger.info("{}'s start-of-turn ability acive".format(thing.name))

                    before = len(self.card_location[player_num])
                    ability(self, self.players[player_num], self.players[1 - player_num], virtual, None, \
                            thing)
                    after = len(self.card_location[player_num])
                    i += int(before == after)

                    if self.check_game_end():
                        break
                if self.check_game_end():
                    break
            else:
                i += 1
        self.ability_resolution(virtual=virtual, player_num=player_num)
        if self.check_game_end():
            return
        i = 0
        while i < len(self.card_location[1 - player_num]):
            thing = self.card_location[1 - player_num][i]
            if thing.turn_start_ability != []:
                for ability in thing.turn_start_ability:
                    # if virtual==False:
                    #    mylogger.info("{}'s start-of-turn ability acive".format(thing.name))

                    before = len(self.card_location[1 - player_num])
                    ability(self, self.players[1 - player_num], self.players[player_num], virtual, None,
                            thing)
                    after = len(self.card_location[1 - player_num])
                    i += int(before == after)

                    if self.check_game_end():
                        break
                if self.check_game_end():
                    break
            else:
                i += 1
        for thing in self.card_location[player_num]:
            thing.down_count(num=1, virtual=virtual)
        self.check_death(player_num, virtual=virtual)
        self.ability_resolution(virtual=virtual, player_num=player_num)

    def end_of_turn(self, player_num, virtual=False):
        self.state_log.clear()
        self.reset_time_stamp()
        self.stack_num += 1
        self.state_log.append([State_Code.END_OF_TURN.value, player_num, self.stack_num])
        self.ability_resolution(virtual=virtual, player_num=player_num)
        for creature_id in self.get_creature_location()[player_num]:
            creature = self.card_location[player_num][creature_id]

            if creature.until_turn_end_buff != [0, 0]:
                creature.power = \
                    max(0, creature.power - creature.until_turn_end_buff[0])
                creature.buff[0] = \
                    max(0, creature.buff[0] - creature.until_turn_end_buff[0])
                creature.until_turn_end_buff = [0, 0]

        i = 0
        while i < len(self.card_location[player_num]):
            thing = self.card_location[player_num][i]
            if thing.turn_end_ability != []:
                for ability in thing.turn_end_ability:
                    # if not virtual:
                    #    mylogger.info("{}'s end-of-turn ability acive".format(thing.name))
                    before = len(self.card_location[player_num])
                    ability(self, self.players[player_num], self.players[1 - player_num], virtual, None, thing)
                    after = len(self.card_location[player_num])
                    i += int(before == after)

                    if self.check_game_end():
                        break
                if self.check_game_end():
                    break
            else:
                i += 1
        self.ability_resolution(virtual=virtual, player_num=player_num)
        i = 0
        while i < len(self.card_location[1 - player_num]):
            thing = self.card_location[1 - player_num][i]
            if thing.turn_end_ability != []:
                for ability in thing.turn_end_ability:
                    # if not virtual:
                    #    mylogger.info("{}'s end-of-turn ability acive".format(thing.name))
                    before = len(self.card_location[1 - player_num])
                    ability(self, self.players[1 - player_num], self.players[player_num], virtual, None, thing)
                    after = len(self.card_location[1 - player_num])
                    i += int(before == after)

                    if self.check_game_end():
                        break
                if self.check_game_end():
                    break
            else:
                i += 1
        if self.check_game_end():
            return
        self.check_death(player_num=player_num, virtual=virtual)

        self.ability_resolution(virtual=virtual, player_num=player_num)
        self.players_play_num = 0

    def check_game_end(self):
        lib_flg = False
        for player in self.players:
            if player.lib_out_flg:
                lib_flg = True
                player.lib_out_flg = False
                break
        return self.players[0].life <= 0 or self.players[1].life <= 0 or lib_flg == True

    def untap(self, player_num):
        self.turn_end = False
        self.current_turn[player_num] += 1
        self.turn_player_num = player_num
        self.evo_flg = False
        for card in self.card_location[player_num]:
            card.untap()

    def reset_remain_cost(self, num):
        self.remain_cost[num] = int(self.cost[num])

    def increment_cost(self, player_num):
        if self.cost[player_num] < self.max_cost:
            self.cost[player_num] += 1
        self.reset_remain_cost(player_num)

    def evolve(self, creature, virtual=False, target=None):
        # if virtual==False:mylogger.info("evo_check")
        if creature.evolved == True or self.evo_flg == True:
            first = creature in self.card_location[0]

            mylogger.info("first:{} policy:{}".format(first, self.players[1 - int(first)].policy.name))
            self.show_field()
            mylogger.info(" name:{} evolved:{} evo_flg:{} able_to_evo:{}"
                          .format(creature.name, creature.evolved, self.evo_flg,
                                  self.get_able_to_evo(self.players[self.turn_player_num])))
            assert False
        card_index = int(creature not in self.card_location[0])
        self.stack_num += 1
        self.state_log.append([State_Code.EVOLVE.value, (card_index, id(creature)), self.stack_num])  # 5は進化したとき
        if not virtual:
            mylogger.info("{} evolve".format(creature.name))
            mylogger.info("remain evo point:{}".format(self.evo_point[self.turn_player_num]))
        creature.evolve(self, target, player_num=self.turn_player_num, virtual=virtual)
        self.evo_flg = True

    def auto_evolve(self, creature, virtual=False):
        assert not creature.evolved, "Already evolved!"
        card_index = int(creature not in self.card_location[0])
        self.stack_num += 1
        self.state_log.append([State_Code.EVOLVE.value, (card_index, id(creature)), self.stack_num])  # 5は進化したとき
        if not virtual:
            mylogger.info("{} evolve".format(creature.name))
        creature.evolve(self, None, player_num=self.turn_player_num, virtual=virtual, auto=True)

    def check_word(self):
        location = self.get_creature_location()
        ans = [False, False]
        for i, side in enumerate(location):
            for j in side:
                creature = self.card_location[i][j]
                if KeywordAbility.WARD.value in creature.ability:
                    ans[i] = True
                    break

        return ans

    def get_creature_location(self):
        ans = [[], []]
        for i in range(2):
            for j, thing in enumerate(self.card_location[i]):
                if thing.card_category == "Creature":
                    ans[i].append(j)
        return ans

    def update_hand_cost(self, player_num=0):
        for i, card in enumerate(self.players[player_num].hand):
            if card.card_class.name == "RUNE" and card.spell_boost is not None and card.cost_down:
                card.cost = max(0, card.origin_cost - card.spell_boost)
            if card.cost_change_ability is not None:
                card.cost_change_ability(card, self, self.players[player_num])
            if card.have_enhance:
                enhance_flg = False
                for cost in card.enhance_cost:
                    if cost <= self.remain_cost[player_num]:
                        enhance_flg = True
                        card.active_enhance_code = [True, cost]
                if not enhance_flg:
                    card.active_enhance_code = [False, 0]
            elif card.have_accelerate:
                accelerate_flg = False
                if self.remain_cost[player_num] >= card.cost:
                    card.active_accelerate_code = [False, 0]
                    continue
                for cost in card.accelerate_cost:
                    if cost <= self.remain_cost[player_num]:
                        accelerate_flg = True
                        card.active_accelerate_code = [True, cost]
                if not accelerate_flg:
                    card.active_accelerate_code = [False, 0]

    def get_can_be_targeted(self, player_num=0):
        can_be_targeted = []
        opponent_side_creature = self.get_creature_location()[1 - player_num]
        for ele in opponent_side_creature:
            creature = self.card_location[1 - player_num][ele]
            if creature.can_be_targeted():
                can_be_targeted.append(ele)
        return can_be_targeted

    def get_can_be_attacked(self, player_num=0):
        can_be_attacked = []
        opponent_side_creature = self.get_creature_location()[1 - player_num]
        for ele in opponent_side_creature:
            creature = self.card_location[1 - player_num][ele]
            if creature.can_be_attacked(): can_be_attacked.append(ele)
        return can_be_attacked

    def get_ward_list(self, player_num=0):
        ward_list = []
        can_be_attacked = self.get_can_be_attacked(player_num=player_num)
        for i in can_be_attacked:
            creature = self.card_location[1 - player_num][i]
            if KeywordAbility.WARD.value in creature.ability:
                ward_list.append(i)

        return ward_list

    def get_situation(self, player, opponent):
        ward_list = self.get_ward_list(player_num=player.player_num)
        can_be_targeted = self.get_can_be_targeted(player_num=player.player_num)
        can_be_attacked = self.get_can_be_attacked(player_num=player.player_num)
        regal_targets = self.get_regal_target_dict(player, opponent)

        return ward_list, can_be_targeted, can_be_attacked, regal_targets

    def get_regal_target_dict(self, player, opponent):
        self.update_hand_cost(player_num=player.player_num)
        regal_targets = {}
        for i, target_card in enumerate(player.hand):
            regal_targets[i] = self.get_regal_targets(target_card, target_type=1, player_num=player.player_num)

        return regal_targets

    def get_able_to_play(self, player, regal_targets=None):
        if regal_targets is None:
            regal_targets = self.get_regal_target_dict()
        full_flg = len(self.card_location[player.player_num]) == self.max_field_num
        # mylogger.info("full_flg={}".format(full_flg))
        able_to_play = []
        for i, hand_card in enumerate(player.hand):
            if hand_card.have_enhance is True and hand_card.active_enhance_code[0] is True:
                if hand_card.active_enhance_code[1] <= self.remain_cost[player.player_num]:
                    if hand_card.card_category == "Spell":
                        if hand_card.enhance_target != 0:
                            if regal_targets[i] != []:
                                able_to_play.append(i)
                        else:
                            able_to_play.append(i)
                    else:
                        if not full_flg:
                            able_to_play.append(i)

            elif hand_card.cost <= self.remain_cost[player.player_num]:
                if hand_card.card_category == "Spell":
                    if hand_card.have_target == 0:
                        able_to_play.append(i)
                    else:
                        if i in regal_targets and regal_targets[i] != []:
                            able_to_play.append(i)
                else:
                    if not full_flg:
                        able_to_play.append(i)
            elif hand_card.have_accelerate is True and hand_card.active_accelerate_code[0] is True:
                if hand_card.active_accelerate_code[1] <= self.remain_cost[player.player_num]:
                    if hand_card.accelerate_target != 0:
                        if regal_targets[i] != []:
                            able_to_play.append(i)
                    else:
                        able_to_play.append(i)
        return able_to_play

    def get_able_to_attack(self, player):
        able_to_attack = []
        for i in self.get_creature_location()[player.player_num]:
            creature = self.card_location[player.player_num][i]
            if creature.can_attack_to_player():
                if creature.player_attack_regulation is not None:
                    if creature.player_attack_regulation(player):
                        able_to_attack.append(i)
                else:
                    able_to_attack.append(i)
        return able_to_attack

    def get_able_to_creature_attack(self, player):
        able_to_creature_attack = []

        for i in self.get_creature_location()[player.player_num]:
            creature = self.card_location[player.player_num][i]
            if creature.can_attack_to_follower():
                if creature.can_only_attack_target is not None:
                    if creature.can_only_attack_target(self, player):
                        able_to_creature_attack.append(i)
                else:
                    able_to_creature_attack.append(i)

            # if creature.evolved==False:able_to_evo.append(i)
        return able_to_creature_attack

    def get_able_to_evo(self, player):
        if self.current_turn[player.player_num] < self.able_to_evo_turn[player.player_num] or self.evo_point[
            player.player_num] == 0 \
                or self.evo_flg == True:
            return []
        able_to_evo = []
        for i in self.get_creature_location()[player.player_num]:
            creature = self.card_location[player.player_num][i]
            if not creature.evolved:
                able_to_evo.append(i)

        return able_to_evo

    def get_flag_and_choices(self, player, opponent, regal_targets):
        self.update_hand_cost(player_num=player.player_num)
        can_attack = True
        can_play = True
        can_evo = True

        able_to_play = self.get_able_to_play(player, regal_targets=regal_targets)
        if len(able_to_play) == 0:
            can_play = False

        able_to_attack = self.get_able_to_attack(player)
        able_to_creature_attack = self.get_able_to_creature_attack(player)
        able_to_evo = self.get_able_to_evo(player)
        if len(able_to_creature_attack) == 0 and len(able_to_attack) == 0:
            can_attack = False
        if len(able_to_evo) == 0 or self.evo_flg == True:
            can_evo = False

        return (can_play, can_attack, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo)

    def get_regal_targets(self, card, target_type=0, player_num=0, human=False):
        if card.card_category == "Creature":
            if card.card_id in creature_ability_condition:
                ability_id = creature_ability_condition[card.card_id]
                if not creature_ability_condition_dict[ability_id](self, player_num, card):
                    return []
        # 0は進化効果の対象取得,1はプレイ時の対象選択
        can_be_targeted = self.get_can_be_targeted(player_num=player_num)
        regal_targets = []
        assert target_type in [0, 1]
        if target_type == 0:
            target_category = card.evo_target
            if human and card.evo_target is not None:
                mylogger.info(
                    "name:{} target_category:{} card.evo_target_regulation:{}".format(card.name, target_category,
                                                                                      card.evo_target_regulation))
            if card.evo_target is None:
                return []
            elif card.evo_target_regulation is None:
                if target_category == Target_Type.ENEMY_FOLLOWER.value:
                    for card_id in can_be_targeted:
                        regal_targets.append(card_id)

                elif target_category == Target_Type.ALLIED_FOLLOWER.value:
                    for card_id in self.get_creature_location()[player_num]:
                        regal_targets.append(card_id)

                elif target_category == Target_Type.ENEMY.value:
                    regal_targets = [-1]
                    for card_id in can_be_targeted:
                        regal_targets.append(card_id)

                elif target_category == Target_Type.FOLLOWER.value:
                    for card_id in can_be_targeted:
                        regal_targets.append((1 - player_num, card_id))
                    for card_id in self.get_creature_location()[player_num]:
                        regal_targets.append((player_num, card_id))

                elif target_category == Target_Type.CARD.value:
                    for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                        if target_thing.can_be_targeted():
                            regal_targets.append((1 - player_num, card_id))

                    for card_id, target_thing in enumerate(self.card_location[player_num]):
                        regal_targets.append((player_num, card_id))

                elif target_category == Target_Type.ALLIED_CARD.value:
                    for card_id, target_thing in enumerate(self.card_location[player_num]):
                        regal_targets.append(card_id)

                elif target_category == Target_Type.ALLIED_CARD_AND_ENEMY_FOLLOWER.value:
                    for player_target_id, opponent_target_id in itertools.product(
                            range(len(self.card_location[player_num])), can_be_targeted):
                        regal_targets.append((player_target_id, opponent_target_id))
                elif target_category == Target_Type.CARD_IN_HAND.value:
                    for i, hand_card in enumerate(self.players[player_num].hand):
                        regal_targets.append(i)
                elif target_category == Target_Type.ENEMY_CARD.value:
                    for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                        if target_thing.can_be_targeted():
                            regal_targets.append(card_id)
            else:
                evo_target_regulation = card.evo_target_regulation
                if target_category == Target_Type.ENEMY_FOLLOWER.value:
                    for card_id in can_be_targeted:
                        target_thing = self.card_location[1 - player_num][card_id]
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append(card_id)

                elif target_category == Target_Type.ALLIED_FOLLOWER.value:
                    for card_id in self.get_creature_location()[player_num]:
                        target_thing = self.card_location[player_num][card_id]
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append(card_id)

                elif target_category == Target_Type.ENEMY.value:
                    regal_targets = [-1]
                    for card_id in can_be_targeted:
                        target_thing = self.card_location[1 - player_num][card_id]
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append(card_id)

                elif target_category == Target_Type.FOLLOWER.value:
                    for card_id in can_be_targeted:
                        target_thing = self.card_location[1 - player_num][card_id]
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append((1 - player_num, card_id))
                    for card_id in self.get_creature_location()[player_num]:
                        target_thing = self.card_location[1 - player_num][card_id]
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append((player_num, card_id))

                elif target_category == Target_Type.CARD.value:
                    for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                        if target_thing.can_be_targeted():
                            if evo_target_regulation(target_thing, card):
                                regal_targets.append((1 - player_num, card_id))

                    for card_id, target_thing in enumerate(self.card_location[player_num]):
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append((player_num, card_id))

                elif target_category == Target_Type.ALLIED_CARD.value:
                    for card_id, target_thing in enumerate(self.card_location[player_num]):
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append(card_id)

                elif target_category == Target_Type.ALLIED_CARD_AND_ENEMY_FOLLOWER.value:
                    for player_target_id, opponent_target_id in itertools.product(
                            range(len(self.card_location[player_num])), can_be_targeted):
                        target_creature = self.card_location[1 - player_num][opponent_target_id]
                        if evo_target_regulation(target_creature, card):
                            regal_targets.append((player_target_id, opponent_target_id))

                elif target_category == Target_Type.CARD_IN_HAND.value:
                    for i, hand_card in enumerate(self.players[player_num].hand):
                        if evo_target_regulation(hand_card, card):
                            regal_targets.append(i)

                elif target_category == Target_Type.ENEMY_CARD.value:
                    for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                        if target_thing.can_be_targeted() and evo_target_regulation(target_thing, card):
                            regal_targets.append(card_id)

                elif target_category == Target_Type.ALLY.value:
                    regal_targets = [-1]
                    for card_id, target_thing in enumerate(self.card_location[player_num]):
                        if evo_target_regulation(target_thing, card):
                            regal_targets.append(card_id)

                elif target_category == Target_Type.ALLIED_AMULET.value:
                    for card_id, target_thing in enumerate(self.card_location[player_num]):
                        if target_thing.card_category == "Amulet":
                            if evo_target_regulation(target_thing, card):
                                regal_targets.append(card_id)

        elif target_type == 1 and \
                ((card.have_enhance is True and card.active_enhance_code[0] is True
                  and card.enhance_target_regulation is None)
                 or (card.have_target != 0 and card.target_regulation is None)
                 or (card.have_accelerate is True and card.active_accelerate_code[0]
                     and card.accelerate_target_regulation is None)):
            target_category = None
            if card.have_enhance is True and card.active_enhance_code[0] is True:
                target_category = card.enhance_target
            elif card.have_accelerate is True and card.active_accelerate_code[0] is True:
                target_category = card.accelerate_target
            else:
                target_category = card.have_target

            if target_category is None or target_category == 0:
                # raise Exception("name:{} target_category:{}".format(card.name,target_category))
                return []
            if target_category == Target_Type.ENEMY_FOLLOWER.value:
                regal_targets = can_be_targeted[:]

            elif target_category == Target_Type.ALLIED_FOLLOWER.value:
                regal_targets = self.get_creature_location()[player_num]

            elif target_category == Target_Type.ENEMY.value:
                regal_targets = [-1] + can_be_targeted[:]

            elif target_category == Target_Type.FOLLOWER.value:
                for card_id in can_be_targeted:
                    regal_targets.append((1 - player_num, card_id))
                for card_id in self.get_creature_location()[player_num]:
                    regal_targets.append((player_num, card_id))

            elif target_category == Target_Type.CARD.value:
                for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                    if target_thing.can_be_targeted():
                        regal_targets.append((1 - player_num, card_id))

                for card_id, target_thing in enumerate(self.card_location[player_num]):
                    regal_targets.append((player_num, card_id))

            elif target_category == Target_Type.ALLIED_CARD.value:
                for card_id, target_thing in enumerate(self.card_location[player_num]):
                    regal_targets.append(card_id)

            elif target_category == Target_Type.ALLIED_CARD_AND_ENEMY_FOLLOWER.value:
                for player_target_id, opponent_target_id in itertools.product(
                        range(len(self.card_location[player_num])), can_be_targeted):
                    regal_targets.append((player_target_id, opponent_target_id))

            elif target_category == Target_Type.CARD_IN_HAND.value:
                itself_index = self.players[player_num].hand.index(card)
                for i, hand_card in enumerate(self.players[player_num].hand):
                    if i == itself_index:
                        continue
                    if i <= itself_index:
                        regal_targets.append(i)
                    else:
                        regal_targets.append(i - 1)

            elif target_category == Target_Type.ENEMY_CARD.value:
                for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                    if target_thing.can_be_targeted():
                        regal_targets.append(card_id)
            elif target_category == Target_Type.ALLY.value:
                regal_targets = [-1]
                regal_targets.extend(self.get_creature_location()[player_num])
            elif target_category == Target_Type.ALLIED_AMULET.value:
                for card_id, target_thing in enumerate(self.card_location[player_num]):
                    if target_thing.card_category == "Amulet":
                        regal_targets.append(card_id)

        elif target_type == 1:
            target_category = None
            regulation_func = None
            if card.have_enhance is True and card.active_enhance_code[0] is True:
                target_category = card.enhance_target
                regulation_func = card.enhance_target_regulation
            elif card.have_accelerate is True and card.active_accelerate_code[0] is True:
                target_category = card.accelerate_target
                regulation_func = card.accelerate_target_regulation
            else:
                target_category = card.have_target
                regulation_func = card.target_regulation

            if target_category is None or target_category == 0:
                return []

            if regulation_func is None:
                assert False
            if target_category == Target_Type.ENEMY_FOLLOWER.value:
                for card_id in can_be_targeted:
                    target_follower = self.card_location[1 - player_num][card_id]
                    if regulation_func(target_follower):
                        regal_targets.append(card_id)

            elif target_category == Target_Type.ALLIED_FOLLOWER.value:
                for card_id in self.get_creature_location()[player_num]:
                    target_follower = self.card_location[player_num][card_id]
                    if regulation_func(target_follower):
                        regal_targets.append(card_id)

            elif target_category == Target_Type.ENEMY.value:
                regal_targets = [-1]
                for card_id in can_be_targeted:
                    target_follower = self.card_location[1 - player_num][card_id]
                    if regulation_func(target_follower):
                        regal_targets.append(card_id)

            elif target_category == Target_Type.FOLLOWER.value:
                for card_id in can_be_targeted:
                    target_thing = self.card_location[1 - player_num][card_id]
                    if regulation_func(target_thing):
                        regal_targets.append((1 - player_num, card_id))
                for card_id in self.get_creature_location()[player_num]:
                    target_thing = self.card_location[1 - player_num][card_id]
                    if regulation_func(target_thing):
                        regal_targets.append((player_num, card_id))

            elif target_category == Target_Type.CARD.value:
                for card_id, target_thing in enumerate(self.card_location[1 - player_num]):
                    if target_thing.can_be_targeted():
                        if regulation_func(target_thing):
                            regal_targets.append((1 - player_num, card_id))

                for card_id, target_thing in enumerate(self.card_location[player_num]):
                    if regulation_func(target_thing):
                        regal_targets.append((player_num, card_id))

            elif target_category == Target_Type.ALLIED_CARD:
                for card_id, target_thing in enumerate(self.card_location[player_num]):
                    if regulation_func(target_thing):
                        regal_targets.append(card_id)

            elif target_category == Target_Type.ALLIED_CARD_AND_ENEMY_FOLLOWER.value:
                for player_target_id, opponent_target_id in itertools.product(
                        range(len(self.card_location[player_num])), can_be_targeted):
                    target_creature = self.card_location[1 - player_num][opponent_target_id]
                    if regulation_func(target_creature):
                        regal_targets.append((player_target_id, opponent_target_id))

            elif target_category == Target_Type.CARD_IN_HAND.value:
                itself_index = self.players[player_num].hand.index(card)
                for i in range(len(self.players[player_num].hand)):
                    if i == itself_index:
                        continue
                    if i <= itself_index:
                        if regulation_func(self.players[player_num].hand[i]):
                            regal_targets.append(i)
                    else:
                        if regulation_func(self.players[player_num].hand[i]):
                            regal_targets.append(i - 1)

            elif target_category == Target_Type.ENEMY_CARD.value:
                for target_id, card in enumerate(self.card_location[1 - player_num]):
                    if card.can_be_targeted() and regulation_func(card):
                        regal_targets.append(target_id)

            elif target_category == Target_Type.ALLY.value:
                regal_targets = [-1]
                regal_targets.extend(self.get_creature_location()[player_num])

            elif target_category == Target_Type.ALLIED_AMULET.value:
                for card_id, target_thing in enumerate(self.card_location[player_num]):
                    if target_thing.card_category == "Amulet":
                        if regulation_func(target_thing):
                            regal_targets.append(card_id)

        return regal_targets

    def play_turn(self, turn_player_num, win, lose, lib_num, turn, virtual_flg):
        while True:
            can_play = True
            can_attack = True
            self.untap(turn_player_num)
            self.increment_cost(turn_player_num)
            if not virtual_flg:
                mylogger.info("Turn {}".format(turn))
                mylogger.info("Player{} turn start cost:{}".format(turn_player_num + 1, self.cost[turn_player_num]))
            self.start_of_turn(turn_player_num, virtual=virtual_flg)
            if self.check_game_end():
                if self.players[turn_player_num].life <= 0 or self.players[turn_player_num].deck.deck == []:
                    if turn_player_num == 0:
                        lose += 1
                    else:
                        win += 1
                elif self.players[1 - turn_player_num].life <= 0 or self.players[1 - turn_player_num].deck.deck == []:
                    if turn_player_num == 0:
                        win += 1
                    else:
                        lose += 1
                break

            if turn_player_num == 1 and self.current_turn[turn_player_num] == 1:
                draw_cards(self.players[turn_player_num], virtual_flg, num=1)
            draw_cards(self.players[turn_player_num], virtual_flg, num=1)
            if self.check_game_end():
                if turn_player_num == 0:
                    lose += 1
                else:
                    win += 1
                lib_num += 1
                return win, lose, lib_num, turn, True
            self.time = time.time()
            while time.time() - self.time < 90:
                end_flg = self.players[turn_player_num].decide(self.players[turn_player_num],
                                                               self.players[1 - turn_player_num], self,
                                                               virtual=virtual_flg)
                if end_flg:
                    break
            # if virtual_flg==False:
            #    os.system('clear')

            if not virtual_flg:
                observable_data = self.get_observable_data(player_num=turn_player_num)
                for key in list(observable_data.keys()):
                    print("{}".format(key))
                    for sub_key in list(observable_data[key].keys()):
                        print("{}:{}".format(sub_key, observable_data[key][sub_key]))
                self.players[turn_player_num].show_hand()
                self.show_field()

            if self.check_game_end():
                if self.players[turn_player_num].life <= 0 or self.players[turn_player_num].lib_out_flg == True:
                    if turn_player_num == 0:
                        lose += 1
                    else:
                        win += 1
                elif self.players[1 - turn_player_num].life <= 0 or self.players[
                    1 - turn_player_num].lib_out_flg == True:
                    if turn_player_num == 0:
                        win += 1
                    else:
                        lose += 1
                break

            self.end_of_turn(turn_player_num, virtual=virtual_flg)

            if self.check_game_end():
                if self.players[turn_player_num].life <= 0 or self.players[turn_player_num].lib_out_flg == True:
                    if turn_player_num == 0:
                        lose += 1
                    else:
                        win += 1
                elif self.players[1 - turn_player_num].life <= 0 or self.players[
                    1 - turn_player_num].lib_out_flg == True:
                    if turn_player_num == 0:
                        win += 1
                    else:
                        lose += 1
                break
            if not virtual_flg:
                mylogger.info("Player{} turn end".format(turn_player_num + 1))
            turn += 1
            if self.ex_turn_count[turn_player_num] > 0:
                if not virtual_flg:
                    mylogger.info("remain_turn:{}".format(self.ex_turn_count[turn_player_num]))
                self.ex_turn_count[turn_player_num] -= 1
            else:
                break

        return win, lose, lib_num, turn, self.check_game_end()


class Graveyard:
    def __init__(self):
        self.graveyard = [[], []]
        self.shadows = [0, 0]
        self.name_list = None

    def append(self, card_category, card_id, player_num):
        self.graveyard[player_num].append((card_category, card_id))
        self.shadows[player_num] += 1

    def show_graveyard(self):
        for i in range(2):
            print("Player", i + 1, "Graveyard")
            for j in range(len(self.graveyard[i])):
                if self.graveyard[i][j][0] == "Creature":
                    print('{:<2}'.format(j), ":", creature_list[self.graveyard[i][j]][-1])
                elif self.graveyard[i][j][0] == "Spell":
                    print('{:<2}'.format(j), ":", spell_list[self.graveyard[i][j]][-1])
                elif self.graveyard[i][j][0] == "Amulet":
                    print('{:<2}'.format(j), ":", amulet_list[self.graveyard[i][j]][-1])

    def graveyard_set(self):

        set_of_graveyard = [None, None]
        name_list = [{}, {}]
        set_of_graveyard[0] = list(set(self.graveyard[0]))
        set_of_graveyard[1] = list(set(self.graveyard[1]))
        counter = [collections.Counter(self.graveyard[0]), collections.Counter(self.graveyard[1])]
        items = [dict(list(counter[0].items())), dict(list(counter[1].items()))]

        for i in range(2):
            for ele in set_of_graveyard[i]:
                card_list = None
                if ele[0] == "Creature":
                    card_list = creature_list
                elif ele[0] == "Spell":
                    card_list = spell_list
                elif ele[0] == "Amulet":
                    card_list = amulet_list
                else:
                    assert False
                if card_list[ele[1]][0] not in name_list[i]:
                    name_list[i][card_list[ele[1]][0]] = {}
                if ele[0] not in name_list[i][card_list[ele[1]][0]]:
                    name_list[i][card_list[ele[1]][0]][ele[0]] = {}

                name_list[i][card_list[ele[1]][0]][ele[0]][card_list[ele[1]][-1]] = items[i][ele]
        self.name_list = name_list

    def show_formated_list(self):
        self.graveyard_set()
        for i in range(2):
            print("Player{} graveyards".format(i + 1))
            for cost_key in sorted(list(self.name_list[i].keys())):
                print("cost {}:".format(cost_key))
                for category_key in sorted(list(self.name_list[i][cost_key].keys())):
                    print("category:{}".format(category_key))
                    for name_key in sorted(list(self.name_list[i][cost_key][category_key].keys())):
                        print("{}×{}".format(name_key, self.name_list[i][cost_key][category_key][name_key]))
            print("\n")


class Play_Cards:
    def __init__(self):
        self.play_cards = [[], []]
        self.name_list = None
        self.played_turn_dict = [{},{}]

    def append(self, card_category, card_id, player_num):
        self.play_cards[player_num].append((card_category, card_id))

    def show_play_cards(self):
        for i in range(2):
            print("Player", i + 1, "Graveyard")
            for j in range(len(self.play_cards[i])):
                if self.play_cards[i][j][0] == "Creature":
                    print('{:<2}'.format(j), ":", creature_list[self.play_cards[i][j]][-1])
                elif self.play_cards[i][j][0] == "Spell":
                    print('{:<2}'.format(j), ":", spell_list[self.play_cards[i][j]][-1])
                elif self.play_cards[i][j][0] == "Amulet":
                    print('{:<2}'.format(j), ":", amulet_list[self.play_cards[i][j]][-1])

    def play_cards_set(self):

        set_of_play_cards = [None, None]
        name_list = [{}, {}]
        set_of_play_cards[0] = list(set(self.play_cards[0]))
        set_of_play_cards[1] = list(set(self.play_cards[1]))
        counter = [collections.Counter(self.play_cards[0]), collections.Counter(self.play_cards[1])]
        items = [dict(list(counter[0].items())), dict(list(counter[1].items()))]
        # name_list[0]=["{:<15}".format(creature_list[i][-1])+":"+str(items[0][i]) for i in set_of_play_cards[0]]
        # name_list[1]=["{:<15}".format(creature_list[i][-1])+":"+str(items[1][i]) for i in set_of_play_cards[1]]
        for i in range(2):
            for ele in set_of_play_cards[i]:
                card_list = None
                if ele[0] == "Creature":
                    card_list = creature_list
                elif ele[0] == "Spell":
                    card_list = spell_list
                elif ele[0] == "Amulet":
                    card_list = amulet_list
                else:
                    assert False
                if card_list[ele[1]][0] not in name_list[i]:
                    name_list[i][card_list[ele[1]][0]] = {}
                if ele[0] not in name_list[i][card_list[ele[1]][0]]:
                    name_list[i][card_list[ele[1]][0]][ele[0]] = {}
                name_list[i][card_list[ele[1]][0]][ele[0]][card_list[ele[1]][-1]] = items[i][ele]
        self.name_list = name_list

    def show_play_list(self):
        self.play_cards_set()
        for i in range(2):
            print("Player{} play_cards".format(i + 1))
            for cost_key in sorted(list(self.name_list[i].keys())):
                print("cost {}:".format(cost_key))
                for category_key in sorted(list(self.name_list[i][cost_key].keys())):
                    print("category:{}".format(category_key))
                    for name_key in sorted(list(self.name_list[i][cost_key][category_key].keys())):
                        print("{}×{}".format(name_key, self.name_list[i][cost_key][category_key][name_key]))
            print("\n")


class Drawn_Cards:
    def __init__(self):
        self.name_list = [[], []]

    def append(self, card, player_num):
        if card.name not in self.name_list[player_num]:
            self.name_list[player_num].append(card.name)
