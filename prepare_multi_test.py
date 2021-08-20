import os
#os.environ["OMP_NUM_THREADS"] = "4"

# -

from emulator_test import *  # importの依存関係により必ず最初にimport
from Field_setting import *
from Player_setting import *
from Policy import *
from Game_setting import Game

from tqdm import tqdm
from Embedd_Network_model import *
import copy
import datetime
# net = New_Dual_Net(100)
import os
from torch.autograd import detect_anomaly
from adabound import AdaBound,AdaBoundW

def preparation(episode_data):
    episode = episode_data[0]
    f = Field(5)
    # p1 = episode_data[0].get_copy(f)
    # p2 = episode_data[1].get_copy(f)
    p1 = episode_data[episode % 2].get_copy(f)
    p2 = episode_data[1 - (episode % 2)].get_copy(f)
    p1.is_first = True
    p2.is_first = False
    p1.player_num = 0
    p2.player_num = 1
    if deck_flg is None:
        deck_type1 = random.randint(0,13)
        deck_type2  = random.randint(0,13)
        #deck_type1 = random.choice(list(key_2_tsv_name.keys()))
        #deck_type2 = random.choice(list(key_2_tsv_name.keys()))
    else:
        deck_type1 = random.choice(deck_flg)#deck_flg
        deck_type2 = random.choice(deck_flg)#deck_flg
    d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
    d1.set_leader_class(key_2_tsv_name[deck_type1][1])
    d1.set_deck_type(deck_id_2_deck_type(deck_type1))
    d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
    d2.set_leader_class(key_2_tsv_name[deck_type2][1])
    d2.set_deck_type(deck_id_2_deck_type(deck_type2))
    d1.shuffle()
    d2.shuffle()
    p1.deck = d1
    p2.deck = d2
    f.players = [p1, p2]
    p1.field = f
    p2.field = f
    x1 = datetime.datetime.now()
    #f.players[0].draw(f.players[0].deck, 3)
    #f.players[1].draw(f.players[1].deck, 3)
    #train_data, reward = G.start(f, virtual_flg=episode!=0)
    train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)


    result_data = []
    sum_of_choices = 0
    sum_code = 0
    for i in range(2):
        for data in train_data[i]:
            # assert False,"{}".format(data[0])
            states = [None,None]
            for j in (0,2):
                states[j//2] = {'hand_ids': data[j].hand_ids, 'hand_card_costs': data[0].hand_card_costs,
                    'follower_card_ids': data[j].follower_card_ids,
                    'amulet_card_ids': data[j].amulet_card_ids,
                    'follower_stats': data[j].follower_stats,
                    'follower_abilities': data[j].follower_abilities,
                    'able_to_evo': data[j].able_to_evo,
                    'life_data': data[j].life_data,
                    'pp_data': data[j].pp_data,
                    'able_to_play': data[j].able_to_play,
                    'able_to_attack': data[j].able_to_attack,
                    'able_to_creature_attack': data[j].able_to_creature_attack,
                    'deck_data': data[j].deck_data}
            before_state,after_state = states
            action_probability = data[1]
            detailed_action_code = data[3]
            sum_of_choices += sum(detailed_action_code['able_to_choice'])
            sum_code += 1
            result_data.append((before_state, action_probability, after_state, detailed_action_code, reward[i]))
            #result_data.append((before_state, action_probability, after_state, detailed_action_code, reward[1-i]))

    x2 = datetime.datetime.now()

    win_name = "Alice" if reward[int(episode%2)] > 0 else "Bob"
    all_len = len(train_data[0])+len(train_data[1])
    tmp_x3 = (x2-x1).total_seconds()/all_len
    x3 = datetime.timedelta(seconds=tmp_x3)
    #print("finished:{:<4} {:<5}(len:{:<3}) time_per_move:{},{}".format(episode + 1,win_name,all_len,x3,x2-x1))
    result_data.append((sum_of_choices,sum_code))
    result_data.append(int(reward[int(episode % 2)] > 0))
    return result_data

def multi_preparation(episode_data):
    #partial_iteration = episode_data[-2]
    p_num = episode_data[-1]
    #print("p_num:",p_num)
    info = f'#{p_num:>2} '
    all_result_data = []
    shared_count = episode_data[-3]
    count_limit  = episode_data[-2]



    battle_data = {"sum_of_choices":0, "sum_code":0, "win_num":0,"end_turn":0}
    #for episode in tqdm(range(partial_iteration),desc=info,position=p_num):
    for _ in tqdm(range(count_limit),desc=info,position=p_num):
        if shared_count.value >= count_limit:
            all_result_data.append(battle_data)
            break
        shared_count.value += 1
        episode = int(shared_count.value)
        f = Field(5)
        p1 = episode_data[episode%2].get_copy(f)
        p2 = episode_data[1-(episode%2)].get_copy(f)
        p1.is_first = True
        p2.is_first = False
        p1.player_num = 0
        p2.player_num = 1
        if deck_flg is None:
            deck_type1 = random.randint(0, 13)
            deck_type2 = random.randint(0, 13)
            #deck_type1 = random.choice(list(key_2_tsv_name.keys()))
            #deck_type2 = random.choice(list(key_2_tsv_name.keys()))
        else:
            deck_type1 = random.choice(deck_flg)#deck_flg
            deck_type2 = random.choice(deck_flg)#deck_flg
        d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
        d1.set_leader_class(key_2_tsv_name[deck_type1][1])
        d1.set_deck_type(deck_id_2_deck_type(deck_type1))
        d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
        d2.set_leader_class(key_2_tsv_name[deck_type2][1])
        d2.set_deck_type(deck_id_2_deck_type(deck_type2))
        d1.shuffle()
        d2.shuffle()
        p1.deck = d1
        p2.deck = d2
        f.players = [p1, p2]
        p1.field = f
        p2.field = f
        x1 = datetime.datetime.now()
        #f.players[0].draw(f.players[0].deck, 3)
        #f.players[1].draw(f.players[1].deck, 3)
        #train_data, reward = G.start(f, virtual_flg=episode!=0)
        train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)


        result_data = []
        sum_of_choices = 0
        sum_code = 0
        end_turn = 0
        for i in range(2):
            end_turn = int(train_data[i][-1][0]["life_data"][0][-1]*100)
            for data in train_data[i]:
                detailed_action_code = data[3]
                after_state = {"hand_ids":data[0]['hand_ids'], 
                         "hand_card_costs":data[0]['hand_card_costs'], 
                         "follower_card_ids":data[0]['follower_card_ids'], 
                         "amulet_card_ids":data[0]['amulet_card_ids'],
                         "follower_stats":data[0]['follower_stats'], 
                         "follower_abilities":data[0]['follower_abilities'], 
                         "able_to_evo":data[0]['able_to_evo'], 
                         "life_data":data[0]['life_data'], 
                         "pp_data":data[0]['pp_data'],
                         "able_to_play":data[0]['able_to_play'], 
                         "able_to_attack":data[0]['able_to_attack'],
                         "able_to_creature_attack":data[0]['able_to_creature_attack'],
                         "deck_data":data[0]['deck_data']}
                before_state = data[2]

                action_probability = data[1]
                sum_of_choices += sum(detailed_action_code['able_to_choice'])
                sum_code += 1
                discounted_reward = reward[i]# * discount_rate
                result_data.append((after_state, action_probability, before_state, detailed_action_code,discounted_reward))

        battle_data["sum_of_choices"] += sum_of_choices
        battle_data["sum_code"] += sum_code
        battle_data["win_num"] += int(reward[int(episode % 2)] > 0)
        battle_data["end_turn"] += end_turn
        all_result_data.append(result_data)

    return all_result_data

import itertools
def multi_battle(episode_data):
    count_limit = episode_data[-3]
    p_id = episode_data[-2]
    deck_id_data = episode_data[-1]
    deck_data_len = len(deck_id_data)
    shared_array = episode_data[-4]
    win_num = 0
    first_num = 0
    info = f'#{str(p_id):>8} '
    for _ in tqdm(range(deck_data_len*count_limit), desc=info, position=p_id):
        if all(shared_array[3*ele] >= count_limit for ele in range(deck_data_len)):
            break
        available_deck_ids = [(index,ele) for index,ele in enumerate(deck_id_data) if shared_array[3*index]< count_limit]

        current_deck_id_data = random.choice(available_deck_ids)
        deck_index,current_deck_ids = current_deck_id_data
        shared_array[3*deck_index] += 1
        episode = shared_array[3*deck_index]
        f = Field(5)
        p1 = episode_data[episode%2].get_copy(f)
        p2 = episode_data[1-(episode%2)].get_copy(f)
        p1.is_first = True
        p2.is_first = False
        p1.player_num = 0
        p2.player_num = 1


        deck_type1 = current_deck_ids[episode%2]#deck_ids[episode%2]
        deck_type2 = current_deck_ids[1-episode%2]#deck_ids[1-episode%2]
        d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
        d1.set_leader_class(key_2_tsv_name[deck_type1][1])
        d1.set_deck_type(deck_id_2_deck_type(deck_type1))
        d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
        d2.set_leader_class(key_2_tsv_name[deck_type2][1])
        d2.set_deck_type(deck_id_2_deck_type(deck_type2))
        d1.shuffle()
        d2.shuffle()
        p1.deck = d1
        p2.deck = d2
        f.players = [p1, p2]
        p1.field = f
        p2.field = f

        f.players[0].draw(f.players[0].deck, 3)
        f.players[1].draw(f.players[1].deck, 3)
        win, lose, _, _ = G.start(f, virtual_flg=True)

        reward = [win,lose]
        shared_array[3*deck_index + 1] += int(reward[int(episode % 2)] > 0)
        shared_array[3*deck_index + 2] += int(episode%2==0)*int(reward[0] > 0)
    return

import itertools

def multi_train(data):
    net, memory, batch_size, iteration_num, train_ids,p_num,current_weight_decay = data
    # optimizer =  optim.AdamW(net.parameters(), weight_decay=current_weight_decay)
    # optimizer = AdaBoundW(net.parameters(),lr=0.001)
    optimizer = optim.SGD(net.parameters(),lr=5e-3)
    all_loss, MSE, CEE = 0, 0, 0

    all_states, all_actions, all_rewards = memory
    states_keys = tuple(all_states.keys())
    normal_states_keys = tuple(set(states_keys) - {'values', 'detailed_action_codes', 'before_states','target'})
    value_keys = tuple(all_states['values'].keys())
    action_code_keys = tuple(all_states['detailed_action_codes'].keys())
    batch_id_list = train_ids
    batch_id_len = len(batch_id_list)
    all_states['target'] = {'actions': all_actions, 'rewards': all_rewards}
    info = f'#{p_num:>2} '
    #print("p_num2:",p_num)
    for i in tqdm(range(iteration_num),desc=info,position=p_num):
        optimizer.zero_grad()
        net.zero_grad()
        key = random.sample(batch_id_list,k=batch_size)
        states = {}
        try:
            states.update({dict_key: torch.clone(all_states[dict_key][key]) for dict_key in normal_states_keys})
        except Exception as e:
            print(normal_states_keys,key)
            raise e
        states['values'] = {sub_key: torch.clone(all_states['values'][sub_key][key]) \
                            for sub_key in value_keys}
        states['detailed_action_codes'] = {sub_key: torch.clone(all_states['detailed_action_codes'][sub_key][key])
                            for sub_key in action_code_keys}
        orig_before_states = all_states["before_states"]
        states['before_states'] = [torch.clone(orig_before_states[i][key]) for i in range(4)]
        actions = all_actions[key]
        rewards = all_rewards[key]

        states['target'] = {'actions': actions, 'rewards': rewards}

        p, v, loss = net(states, target=True)

        loss[0].backward(retain_graph=False)

        all_loss += float(loss[0].item())
        MSE += float(loss[1].item())
        CEE += float(loss[2].item())
        optimizer.step()
        del states
        if i != iteration_num -1:
            del p
            del v
            del loss
    if p_num == 0:
        p_list = [float(cell) for cell in p[0]]
        print("p:")
        for k in range(15):
            first = 3 * k
            second = first + 1
            third = first + 2
            print("{:2d}: {:7.3%} {:2d}: {:7.3%} {:2d}: {:7.3%}".format(first,p_list[first],second,p_list[second],third,p_list[third]))
        print("")
        print("actions:{}\n".format(actions[0]))
        print("v:{}".format(float(v[0])))
        print("rewards:{}".format(rewards[0]))
    


    return all_loss, MSE, CEE


def multi_eval(data):
    net, memory, batch_size, partitial_range, p_num = data
    all_loss, MSE, CEE = 0, 0, 0

    all_states, all_actions, all_rewards = memory
    states_keys = list(all_states.keys())
    value_keys = list(all_states['values'].keys())
    action_code_keys = list(all_states['detailed_action_codes'].keys())
    normal_states_keys = tuple(set(states_keys) - {'values', 'detailed_action_codes', 'before_states','target'})
    memory_len = all_actions.size()[0]
    batch_id_list = partitial_range
    partitial_len =  len(partitial_range)
    separate_num =partitial_len//batch_size
    assert separate_num > 0,"zero sep.{},{},{}".format(partitial_len,batch_size,batch_id_list)
    info = f'#{p_num:>2} '
    for i in tqdm(range(separate_num),desc=info,position=p_num+1):
        first_id =(i*batch_size) % partitial_len
        last_id = ((i+1)*batch_size) % partitial_len
        key = batch_id_list[first_id:last_id] if first_id < last_id else batch_id_list[first_id:]
        states = {}
        states.update({dict_key : torch.clone(all_states[dict_key][key]) for dict_key in normal_states_keys})
        states['values'] = {sub_key: torch.clone(all_states['values'][sub_key][key]) \
                            for sub_key in value_keys}
        states['detailed_action_codes'] = {sub_key: torch.clone(all_states['detailed_action_codes'][sub_key][key])
                            for sub_key in action_code_keys}
        orig_before_states = all_states["before_states"]
        states['before_states'] = [torch.clone(orig_before_states[i][key]) for i in range(4)]
        actions = all_actions[key]
        rewards = all_rewards[key]

        states['target'] = {'actions': actions, 'rewards': rewards}

        _, _, loss = net(states, target=True)
        all_loss += float(loss[0].item())
        MSE += float(loss[1].item())
        CEE += float(loss[2].item())
    return all_loss/separate_num, MSE/separate_num, CEE/separate_num