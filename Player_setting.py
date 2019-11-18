import numpy as np
import random
import math
from Policy import *
from my_moduler import get_module_logger, get_state_logger
from mulligan_setting import *

mylogger = get_module_logger(__name__)
# mylogger = get_module_logger('mylogger')
import itertools

statelogger = get_state_logger('state')
from my_enum import *


class Player:
    def __init__(self, max_hand_num, first=True, policy=RandomPolicy(), mulligan=Random_mulligan_policy()):
        self.hand = []
        self.max_hand_num = max_hand_num
        self.is_first = first
        self.player_num = 1 - int(self.is_first)
        self.life = 20
        self.max_life = 20
        self.policy = policy
        self.mulligan_policy = mulligan
        self.deck = None
        self.lib_out_flg = False
        self.field = None
        self.name = None
        self.class_num = None
        self.effect = []

    def set_field(self, field):
        self.field = field

    def get_copy(self, field):
        player = Player(self.max_hand_num, first=self.is_first, policy=self.policy, mulligan=self.mulligan_policy)
        # player.hand=copy.deepcopy(self.hand)
        for card in self.hand:
            player.hand.append(card.get_copy())
        player.life = self.life
        # player.deck=copy.deepcopy(self.deck)
        player.deck = Deck()
        for card in self.deck.deck:
            player.deck.append(card)
        if len(player.deck.deck) == 0 and len(self.deck.deck) > 0:
            raise Exception()

        player.set_field(field)
        player.name = self.name
        player.class_num = self.class_num
        if len(self.effect) > 0:
            player.effect = copy.deepcopy(self.effect)
        return player

    def get_damage(self, damage):
        if len(self.effect) > 0:
            tmp = int(damage)
            priority_list = list(set([effect.proirity for effect in self.effect]))
            priority_list = sorted(priority_list, reverse=True)

            for i in priority_list:
                for effect in self.effect:
                    if effect.priority == i:
                        tmp = effect(argument=tmp, state_code=State_Code.GET_DAMAGE.value)
            return tmp
        else:
            self.life -= damage
            return damage

    def restore_player_life(self, num=0, virtual=False):
        self.field.restore_player_life(player=self, num=num, virtual=virtual)
        """
            tmp=num
            if self.max_life-self.life<tmp:
                tmp=self.max_life-self.life
            self.life+=tmp
            if virtual==False:
                mylogger.info("Player {} restore {} life".format(self.player_num+1,tmp))
            self.field.stack_num+=1
            self.field.state_log.append([State_Code.RESTORE_PLAYER_LIFE.value,self.player_num,self.field.stack_num])
            """

    def check_vengeance(self):
        return self.life <= 10

    def check_overflow(self):
        return self.field.cost[self.player_num] >= 7

    def check_resonance(self):
        return len(self.deck.deck) % 2 == 0

    def draw(self, deck, num):
        for i in range(num):
            if deck.deck == []:
                self.lib_out_flg = True
                return
            self.hand.append(deck.draw())
            if len(self.hand) > self.max_hand_num:
                self.field.graveyard.shadows[self.player_num] += 1
                self.hand.pop(-1)

    def append_cards_to_hand(self, cards):
        for card in cards:
            self.hand.append(card)
            if len(self.hand) > self.max_hand_num:
                self.field.graveyard.shadows[self.player_num] += 1
                self.hand.pop(-1)

    def show_hand(self):
        length = 0
        print("Player", self.player_num + 1, "'s hand")
        print("====================================================================")
        for i in range(len(self.hand)):
            print(i, ": ", self.hand[i])
            length = i
        print("====================================================================")
        for i in range(9 - length):
            print("")

    def mulligan(self, deck, virtual=False):
        change_cards_id = self.mulligan_policy.decide(self.hand, deck)
        if virtual == False:
            mylogger.info("Player{}'s hand".format(self.player_num + 1))
            self.show_hand()
            mylogger.info("change card_id:{}".format(change_cards_id))
        return_cards = [self.hand.pop(i) for i in change_cards_id[::-1]]
        self.draw(deck, len(return_cards))
        if virtual == False:
            self.show_hand()
        for i in range(len(return_cards)):
            deck.append(return_cards.pop())

        deck.shuffle()

    def play_card(self, field, card_id, player, opponent, virtual=False, target=None):
        if self.hand[card_id].have_enhance == True and self.hand[card_id].active_enhance_code[0] == True:
            field.remain_cost[self.player_num] -= self.hand[card_id].active_enhance_code[1]
            if virtual == False:
                mylogger.info("Enhance active!")
        elif self.hand[card_id].have_accelerate == True and self.hand[card_id].active_accelerate_code[0] == True:
            field.remain_cost[self.player_num] -= self.hand[card_id].active_accelerate_code[1]
            if virtual == False:
                mylogger.info("Accelerate active!")
            field.play_as_other_card(self.hand, card_id, self.player_num, virtual=virtual, target=target)
            return
        else:
            field.remain_cost[self.player_num] -= self.hand[card_id].cost

        if virtual == False:
            mylogger.info("Player {} plays {}".format(self.player_num + 1, self.hand[card_id].name))
        if self.hand[card_id].card_category == "Creature":
            field.play_creature(self.hand, card_id, self.player_num, player, opponent, virtual=virtual, target=target)
        elif self.hand[card_id].card_category == "Spell":
            field.play_spell(self.hand, card_id, self.player_num, player, opponent, virtual=virtual, target=target)
        elif self.hand[card_id].card_category == "Amulet":
            field.play_amulet(self.hand, card_id, self.player_num, player, opponent, virtual=virtual, target=target)
        field.players_play_num += 1
        field.ability_resolution(virtual=virtual, player_num=self.player_num)
        # if field.stack!=[]:
        #    field.solve_lastword_ability(virtual=virtual,player_num=self.player_num)

    def attack_to_follower(self, field, attacker_id, target_id, virtual=False):
        field.attack_to_follower([self.player_num, attacker_id], [1 - self.player_num, target_id], field,
                                 virtual=virtual)

    def attack_to_player(self, field, attacker_id, opponent, virtual=False):
        # mylogger.info("location:{}".format([self.player_num,attacker_id]))
        field.attack_to_player([self.player_num, attacker_id], opponent, virtual=virtual)

    def creature_evolve(self, creature, field, target=None, virtual=False):
        assert field.evo_point[self.player_num] > 0
        if virtual == False: mylogger.info("evo_check")
        field.evo_point[self.player_num] -= 1
        field.evolve(creature, virtual=virtual, target=target)

    def discard(self, id, field):
        del self.hand[id]

    def decide(self, player, opponent, field, virtual=False):
        field.stack = []
        # field.get_previous_field()
        (_, _, can_be_attacked, regal_targets) = field.get_situation(self, opponent)
        # mylogger.info("regal_targets1:{}".format(regal_targets))
        (can_play, can_attack, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) \
            = field.get_flag_and_choices(self, opponent, regal_targets)
        # mylogger.info("regal_targets2:{}".format(regal_targets))
        if not virtual:
            self.show_hand()
            field.show_field()
            mylogger.info(
                "Player1 life:{} Player2 life:{} remain_cost:{}".format(field.players[0].life, field.players[1].life, \
                                                                        field.remain_cost[self.player_num]))
            if able_to_play != []:
                mylogger.info("able_to_play:{}".format(able_to_play))
            if able_to_creature_attack != []:
                mylogger.info(
                    "able_to_creature_attack:{} can_be_attacked:{}".format(able_to_creature_attack, can_be_attacked))
            mylogger.info("regal_targets:{}".format(regal_targets))

        # self.show_hand()
        re_check = False
        action_num = 0
        card_id = 0
        target_id = None
        msg = ""
        policy_count = 0
        while True:
            policy_count += 1
            if policy_count > 20:
                mylogger.info("action_code:{}".format((action_num, card_id, target_id)))
                self.show_hand()
                field.show_field()
                assert False, "infinite loop!"
            re_check = False
            sim_hand = None
            if self.policy.policy_type == 3 and self.policy.current_node is not None:
                sim_field = self.policy.current_node.field
                real_hand = self.hand
                sim_hand = sim_field.players[self.player_num].hand
                if len(real_hand) != len(sim_hand):
                    msg = "diff hand len error({}!={})".format(len(real_hand),len(sim_hand))
                    re_check = True
                else:
                    for tmp_id in range(len(self.hand)):
                        if not real_hand[tmp_id].eq(sim_hand[tmp_id]):
                            msg = "diff hand card error"
                            re_check = True
                            break
                    if re_check is False and not field.eq(sim_field):
                        msg = "diff field error"
                        re_check = True

                if re_check:
                    if not virtual:
                        mylogger.info("{}".format(msg))
                    self.policy.current_node=None
                    continue
            (action_num, card_id, target_id) = self.policy.decide(self, opponent, field)
            if action_num == Action_Code.ERROR.value:
                self.policy.current_node = None
                continue

            assert action_num in [-1, 0, 1, 2, 3]
            re_check = False
            if action_num == Action_Code.TURN_END.value:
                break

            if action_num == Action_Code.EVOLVE.value:
                if not virtual:
                    mylogger.info("able_to_evo:{} card_id:{}".format(field.get_able_to_evo(self), card_id))
                if card_id not in field.get_able_to_evo(self):
                    msg = "evo_error(able_to_evo:{} card_id:{})".format(field.get_able_to_evo(self), card_id)
                    re_check = True
                else:
                    evo_creature = field.card_location[self.player_num][card_id]
                    if target_id is not None and\
                        target_id not  in field.get_regal_targets(evo_creature, target_type=0, player_num=player.player_num):
                        msg = "illegal evo-target error"
                        re_check = True
            elif action_num == Action_Code.PLAY_CARD.value:
                if card_id not in regal_targets:
                    msg = "illigal_hand_card_error({} not in {})".format(card_id,regal_targets.keys())
                    re_check = True
                elif regal_targets[card_id] != [] and target_id is None:
                    msg = "non-target-error"
                    #assert False,"{} regal_targets:{}".format((action_num,card_id,target_id),regal_targets[card_id])
                    re_check = True
                if card_id in regal_targets and target_id is not None:
                    if sim_hand is not None and self.hand[card_id].have_target == Target_Type.CARD_IN_HAND.value:
                        check_targets=field.get_regal_targets(self.hand[card_id], target_type=1, player_num=self.player_num)
                        #mylogger.info("regal_targets:{}".format(check_targets))
                        #assert False
                        if sim_hand[target_id].name != self.hand[target_id].name:
                            msg = "diff target name error(target_id:{},{} is not {})"\
                                .format(target_id,self.hand[target_id].name,sim_hand[target_id].name)
                    elif target_id is not None and target_id not in regal_targets[card_id]:
                        msg = "illigal_target_error"
                        re_check = True

                if len(self.hand) <= card_id:
                    re_check = True
                else:
                    if player.hand[card_id].card_category != "Spell":
                        if len(field.card_location[self.player_num]) == field.max_field_num and \
                                self.hand[card_id].active_accelerate_code[0] == False:
                            msg = "full-field error"
                            re_check = True
                        elif self.hand[card_id].have_target != 0:
                            if regal_targets[card_id] != [] and target_id not in regal_targets[card_id]:
                                msg = "illegal spell target error({} not in {})".format(target_id,regal_targets[card_id])
                                re_check = True
                    elif self.hand[card_id].have_target != 0:
                        if regal_targets[card_id] == [] or target_id not in regal_targets[card_id]:
                            msg = "illegal spell target error({} not in {})".format(target_id,regal_targets[card_id])
                            re_check = True
            elif action_num == Action_Code.ATTACK_TO_FOLLOWER.value:
                if target_id not in can_be_attacked:
                    msg = "illegal follower-attacker error({} not in {})".format(target_id,can_be_attacked)
                    re_check = True
                elif card_id not in able_to_creature_attack:
                    msg = "illegal follower-attacker error({} not in {})".format(card_id,able_to_creature_attack)
                    re_check = True
            elif action_num == Action_Code.ATTACK_TO_PLAYER.value:
                if card_id not in able_to_attack:
                    msg = "illegal player-attacker error({} not in {})".format(card_id,able_to_attack)
                    re_check = True

            if not re_check:
                break
            else:
                if not virtual:
                    mylogger.info("msg:{}".format(msg))

        if not virtual:
            mylogger.info("action_num:{} card_id:{} target_id:{}".format(action_num, card_id, target_id))
            if action_num == 1 and len(regal_targets[card_id]) > 0 and target_id is None:
                raise Exception()
        end_flg = self.execute_action(field, opponent, action_code=(action_num, card_id, target_id), virtual=virtual)

        return end_flg

    def execute_action(self, field, opponent, action_code=None, virtual=False):

        field.reset_time_stamp()
        if action_code is None:
            raise Exception("action_code is None!")
        (action_num, card_id, target_id) = action_code

        if action_num == Action_Code.EVOLVE.value:
            self.creature_evolve(field.card_location[self.player_num][card_id], \
                                 field, virtual=virtual, target=target_id)
        elif action_num == Action_Code.TURN_END.value:
            field.turn_end = True
            return True
        elif action_num == Action_Code.PLAY_CARD.value:
            if not virtual:
                if self.hand[card_id].have_enhance == True \
                        and self.hand[card_id].active_enhance_code[0] == True:
                    mylogger.info("play_cost:{}".format(self.hand[card_id].active_enhance_code[1]))
                elif self.hand[card_id].have_accelerate == True \
                        and self.hand[card_id].active_accelerate_code[0] == True:
                    mylogger.info("play_cost:{}".format(self.hand[card_id].active_accelerate_code[1]))
                else:
                    mylogger.info("play_cost:{}".format(self.hand[card_id].cost))

            self.play_card(field, card_id, self, opponent, target=target_id, virtual=virtual)

        elif action_num == Action_Code.ATTACK_TO_FOLLOWER.value:
            self.attack_to_follower(field, card_id, target_id, virtual=virtual)

        elif action_num == Action_Code.ATTACK_TO_PLAYER.value:
            self.attack_to_player(field, card_id, opponent, virtual=virtual)

        field.check_death(player_num=self.player_num, virtual=virtual)
        field.ability_resolution(virtual=virtual, player_num=self.player_num)
        return field.check_game_end()


class HumanPlayer(Player):
    def __init__(self, max_hand_num, first=True, policy=RandomPolicy(), mulligan=Random_mulligan_policy()):
        super(HumanPlayer, self).__init__(max_hand_num, first=True, policy=policy, mulligan=mulligan)

    def mulligan(self, deck, virtual=False):
        self.show_hand()
        tmp = input("input change card id(if you want to change all card,input ↑):")
        if tmp == "":
            return
        elif tmp == "\x1b[A":
            tmp = [i for i in range(len(self.hand))]
        else:
            tmp = tmp.split(",")
        # mylogger.info("tmp:{} type:{}".format(tmp, type(tmp)))
        change_cards_id = []
        if len(tmp) > 0:
            change_cards_id = list(map(int, tmp))
            return_cards = [self.hand.pop(i) for i in change_cards_id[::-1]]
            self.draw(deck, len(return_cards))
            self.show_hand()
            for i in range(len(return_cards)):
                deck.append(return_cards.pop())

            deck.shuffle()

    def decide(self, player, opponent, field, virtual=False):
        field.reset_time_stamp()

        (ward_list, can_be_targeted, can_be_attacked, regal_targets) = field.get_situation(player, opponent)

        (can_play, can_attack, can_evo), (able_to_play, able_to_attack, able_to_creature_attack, able_to_evo) \
            = field.get_flag_and_choices(player, opponent, regal_targets)
        self.show_hand()
        field.show_field()
        print("Your life:{},Oppornent life:{}".format(player.life, opponent.life))
        print("remain_cost:{}".format(field.remain_cost[player.player_num]))
        print("shadows:{}".format(field.graveyard.shadows[player.player_num]))
        choices = [0]
        if can_evo == True:
            print("if you want to evolve creature,input -1")
            choices.append(-1)
        if can_play == True:
            print("if you want to play card, input 1")
            choices.append(1)

        if can_attack == True:
            if len(can_be_attacked) > 0:
                print("if you want to attack to creature, input 2")
                choices.append(2)

            if ward_list == [] and len(able_to_attack) > 0:
                print("if you want to attack to player, input 3")
                choices.append(3)

        print("if you want to call turn end, input 0")
        """
            if len(choices)==1:
                print("Turn end")
                can_play,can_attack=False,False
            """
        tmp = input("you can input {} :".format(choices))
        action_num = 0
        if tmp == "":
            action_num = 0
        else:
            action_num = int(tmp)
        if action_num not in choices:
            print("invalid input!")
            return can_play, can_attack, field.check_game_end()
        if action_num == -1:
            print("you can evolve:{}".format(able_to_evo))
            evo_names = ["id:{} name:{}".format(ele, field.card_location[self.player_num][ele].name) for ele in
                         able_to_evo]
            for cell in evo_names:
                mylogger.info("{}".format(cell))
            card_id = int(input("input creature id :"))
            if card_id not in able_to_evo:
                print("already evolved!")
                return can_play, can_attack, field.check_game_end()
            if field.card_location[self.player_num][card_id].evo_target != None:
                mylogger.info("target-evolve")
                regal = field.get_regal_targets(field.card_location[self.player_num][card_id], target_type=0,
                                                player_num=self.player_num, human=True)
                mylogger.info("targets:{}".format(regal))
                if regal != []:
                    print("you can target:{}".format(regal))
                    target_id = int(input("input target_id :"))
                    if target_id not in regal:
                        print("illigal target!")
                        return can_play, can_attack, field.check_game_end()
                    self.creature_evolve(field.card_location[self.player_num][card_id], field, target=target_id)
                else:
                    mylogger.info("evolve")
                    self.creature_evolve(field.card_location[self.player_num][card_id], field)


            else:
                mylogger.info("evolve")
                self.creature_evolve(field.card_location[self.player_num][card_id], field)

        if action_num == 0:
            print("Turn end")
            return True

        if action_num == 1:
            self.show_hand()
            print("remain cost:", field.remain_cost[self.player_num])
            print("able to play:{}".format(able_to_play))
            hand_names = ["id:{} name:{}".format(ele, self.hand[ele].name) for ele in able_to_play]
            for cell in hand_names:
                mylogger.info("{}".format(cell))
            card_id = int(input("input card id :"))
            if card_id not in able_to_play:
                print("can't play!")
                return can_play, can_attack, field.check_game_end()
            target_id = None
            if self.hand[card_id].have_target != 0:
                print("valid_targets:{}".format(regal_targets[card_id]))
                target_id = input("input target code(splited by space):")
                target_code = None
                if target_id == "":
                    if self.hand[card_id] != "Spell":
                        target_code = [None]
                    else:
                        print("you can't play spell that have target!")
                        return can_play, can_attack, field.check_game_end()


                else:
                    target_code = tuple(map(int, target_id.split(" ")))
                if len(target_code) == 1:
                    target_code = target_code[0]
                if regal_targets[card_id] == []:
                    target_code = None
                elif target_code not in regal_targets[card_id]:
                    print("Invalid target!")
                    return can_play, can_attack, field.check_game_end()
                target_id = target_code
            print("remain cost:", field.remain_cost[self.player_num])
            self.play_card(field, card_id, player, opponent, target=target_id)
            field.show_field()

        if action_num == 2:
            field.show_field()
            print("able_to_attack:{}".format(able_to_creature_attack))
            if len(ward_list) > 0:
                print("can_be_attacked:{}".format(ward_list))
            else:
                print("can_be_attacked:{}".format(can_be_attacked))

            card_id = int(input("input creature id you want to let attack:"))
            if card_id > field.card_num[self.player_num] - 1:
                print("invalid card id!")
                return can_play, can_attack, field.check_game_end()
            if card_id not in able_to_creature_attack:
                print("can't attack!")
                return can_play, can_attack, field.check_game_end()
            tmp = int(input("input target creature id you want to attack:"))
            if tmp not in can_be_attacked:
                mylogger.info("invalid target id!")
                return can_play, can_attack, field.check_game_end()

            if len(ward_list) == 0:
                if tmp > field.card_num[1 - self.player_num] - 1:
                    print("invalid target id!")
                    return can_play, can_attack, field.check_game_end()
            else:
                if tmp not in ward_list:
                    print("invalid target id!")
                    return can_play, can_attack, field.check_game_end()

            self.attack_to_follower(field, card_id, tmp)
            if field.card_num[self.player_num] < field.max_field_num and len(able_to_play) > 0:
                can_attack = True

        if action_num == 3:
            print("able to attack:{}".format(able_to_attack))
            card_id = int(input("input creature id you want to let attack:"))
            if card_id not in able_to_attack:
                print("can't attack!")
                return can_play, can_attack, False
            self.attack_to_player(field, card_id, opponent)

        field.check_death(player_num=self.player_num, virtual=virtual)
        field.solve_field_trigger_ability(virtual=virtual, player_num=self.player_num)
        # if field.stack!=[]:
        #    field.solve_lastword_ability(virtual=virtual,player_num=self.player_num)
        field.ability_resolution(virtual=virtual, player_num=self.player_num)
        return field.check_game_end()
