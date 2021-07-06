# -*- coding: utf-8 -*-
# +
if __name__ == "__main__":
    from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock,freeze_support, Value, Array, Manager,cpu_count
    try:
        set_start_method('spawn')
        print("spawn is run.")
        #set_start_method('fork') GPU使用時CUDA initializationでerror
        #print('fork')
    except RuntimeError:
        pass
import ctypes
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
GAMMA = 0.9
parser = argparse.ArgumentParser(description='デュアルニューラルネットワーク学習コード')

parser.add_argument('--episode_num', help='試行回数', type=int,default=128)
parser.add_argument('--iteration_num', help='イテレーション数', type=int,default=1000)
parser.add_argument('--epoch_num', help='エポック数', type=int,default=64)
parser.add_argument('--batch_size', help='バッチサイズ', type=int,default=256)
parser.add_argument('--mcts', help='サンプリングAIをMCTSにする(オリジナルの場合は[OM])')
parser.add_argument('--deck', help='サンプリングに用いるデッキの選び方')
parser.add_argument('--cuda', help='gpuを使用するかどうか')
parser.add_argument('--multi_train', help="学習時も並列化するかどうか")
parser.add_argument('--save_interval', help="モデルの保存間隔", type=int,default=10)
parser.add_argument('--fixed_deck_ids', help="使用デッキidリストの固定",type=\
    lambda str:list(map(int,str.split(","))))
parser.add_argument('--cpu_num', help="使用CPU数",default=2 if torch.cuda.is_available() else 3,type=int)
parser.add_argument('--batch_num', help='サンプルに対するバッチの数')
parser.add_argument('--fixed_opponent', help='対戦相手を固定')
parser.add_argument('--node_num', help='node_num', default=100,type=int)
parser.add_argument('--weight_decay', help='weight_decay', default=1e-2,type=float)
parser.add_argument('--check', help='check score')
parser.add_argument('--deck_list', help='deck_list',default="0,1,4,5,10,11")
parser.add_argument('--model_name', help='model_name', default=None,type=lambda text:text.replace("\r",""))
parser.add_argument('--opponent_model_name', help='opponent_model_name', default=None)
parser.add_argument('--th', help='threshold',default=1e-3,type=float)
parser.add_argument('--WR_th', help='WR_threshold',default=0.55,type=float)
parser.add_argument('--check_deck_id', help='check_deck_id')
parser.add_argument('--evaluate_num', help='evaluate_num',default=100,type=int)
parser.add_argument('--max_update_interval', help='max_update_interval',default=10,type=int)
parser.add_argument('--limit_OMP',help="limit OMP_NUM_THREADS for quadro",default=False,type=bool)
parser.add_argument('--OMP_NUM', help='num of threads used in OMP',default=0,type=int)
parser.add_argument('--loss_th', help='アーリーストッピングの猶予ステップ',default=10,type=int)
parser.add_argument('--step_iter', help='MCTS_step_iteration',default=100,type=int)
parser.add_argument('--supervised', help='if model use other model')
parser.add_argument('--data_rate', help='the rate of data used by train',default=0.8,type=float)
parser.add_argument('--w_list', help='list of weight decay',default=[0.001,0.002,0.004,0.008],type=lambda txt:list(map(float,txt.split(","))))
parser.add_argument('--rand', help='if model use random initial embedding')
parser.add_argument('--epoch_list', help='list of epoch num as train',default=[100],type=lambda txt:list(map(int,txt.split(","))))
parser.add_argument('--multi_sample_num', help='num of sampling process',default=0,type=int)
parser.add_argument('--hidden_num', help='num of hidden_layer',default=[6,6],type=lambda txt: list(map(int,txt.split(","))))
parser.add_argument('--greedy_mode', help='use self-play greedy model',)
args = parser.parse_args()

deck_flg = args.fixed_deck_ids#list(map(int,args.fixed_deck_ids.split(","))) if args.fixed_deck_ids is not None else None
weight_decay = args.weight_decay
evaluate_num = args.evaluate_num


#Detailed_State_data = namedtuple('Value', ('hand_ids', 'hand_card_costs', 'follower_card_ids',
#                                           'amulet_card_ids', 'follower_stats', 'follower_abilities', 'able_to_evo',
#                                           'life_data', 'pp_data', 'able_to_play', 'able_to_attack',
#                                           'able_to_creature_attack'))
cpu_num = args.cpu_num
batch_num = int(args.batch_num) if args.batch_num is not None else None
G = Game()
fixed_opponent = args.fixed_opponent
cuda_flg = args.cuda is not None
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
            before_state = {'hand_ids': data[0].hand_ids, 'hand_card_costs': data[0].hand_card_costs,
                            'follower_card_ids': data[0].follower_card_ids,
                            'amulet_card_ids': data[0].amulet_card_ids,
                            'follower_stats': data[0].follower_stats,
                            'follower_abilities': data[0].follower_abilities,
                            'able_to_evo': data[0].able_to_evo,
                            'life_data': data[0].life_data,
                            'pp_data': data[0].pp_data,
                            'able_to_play': data[0].able_to_play,
                            'able_to_attack': data[0].able_to_attack,
                            'able_to_creature_attack': data[0].able_to_creature_attack,
                            'deck_data': data[0].deck_data}

            after_state = {'hand_ids': data[2].hand_ids, 'hand_card_costs': data[2].hand_card_costs,
                           'follower_card_ids': data[2].follower_card_ids,
                           'amulet_card_ids': data[2].amulet_card_ids,
                           'follower_stats': data[2].follower_stats,
                           'follower_abilities': data[2].follower_abilities,
                           'able_to_evo': data[2].able_to_evo,
                           'life_data': data[2].life_data,
                           'pp_data': data[2].pp_data,
                           'able_to_play': data[2].able_to_play,
                           'able_to_attack': data[2].able_to_attack,
                           'able_to_creature_attack': data[2].able_to_creature_attack,
                           'deck_data':data[2].deck_data}
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
                
                #print(0,data[0])
                #print(2,data[2])
                # assert False,"{}".format(data[0])
                detailed_action_code = data[3]
                #if sum(detailed_action_code["able_to_choice"]) == 1:
                #    continue
                #print(data[0])
#                 after_state = Detailed_State_data(data[0].hand_ids, data[0].hand_card_costs,
#                                 data[0].follower_card_ids, data[0].amulet_card_ids,
#                                 data[0].follower_stats, data[0].follower_abilities,
#                                 data[0].able_to_evo, data[0].life_data,
#                                 data[0].pp_data, data[0].able_to_play,
#                                 data[0].able_to_attack, data[0].able_to_creature_attack,
#                                                   data[0].deck_data)
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
#                 after_state = Detailed_State_data(data[0]['hand_ids'], data[0]['hand_card_costs'],
#                                data[0]['follower_card_ids'], data[0]['amulet_card_ids'],
#                                data[0]['follower_stats'], data[0]['follower_abilities'],
#                                data[0]['able_to_evo'], data[0]['life_data'],
#                                data[0]['pp_data'], data[0]['able_to_play'],
#                                data[0]['able_to_attack'], data[0]['able_to_creature_attack'],
#                                                  data[0]['deck_data'])
                before_state = data[2]

                action_probability = data[1]
                sum_of_choices += sum(detailed_action_code['able_to_choice'])
                sum_code += 1
                #current_turn = int(data[0].life_data[0][-1] * 100)
                #discount_rate = GAMMA**(end_turn-current_turn)
                discounted_reward = reward[i]# * discount_rate
                result_data.append((after_state, action_probability, before_state, detailed_action_code,discounted_reward))#, reward[i]))
                #print("life_data:{}".format(data[2].life_data))
                #end_turn = data[2].life_data[0][-1]
                #result_data.append((before_state, action_probability, after_state, detailed_action_code, reward[1-i]))

        battle_data["sum_of_choices"] += sum_of_choices
        battle_data["sum_code"] += sum_code
        battle_data["win_num"] += int(reward[int(episode % 2)] > 0)
        battle_data["end_turn"] += end_turn
        all_result_data.append(result_data)

    #all_result_data.append(battle_data)
    #print("\033["+str(p_num)+"A", end="")
    return all_result_data

import itertools
def multi_battle(episode_data):
    count_limit = episode_data[-3]
    #partial_iteration = episode_data[-3]
    p_id = episode_data[-2]
    #deck_ids = episode_data[-1]
    deck_id_data = episode_data[-1]#((deck_id,deck_id),)
    deck_data_len = len(deck_id_data)
    shared_array = episode_data[-4]
    #win_rate_dict = {ele:{"win_num":0,"first_win_num":0}\
    #                 for ele in deck_id_data}
    win_num = 0
    first_num = 0
    info = f'#{str(p_id):>8} '#info = f'#{str(deck_ids):>8} '
    #for episode in tqdm(range(partial_iteration),desc=info,position=p_id):
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
        #x1 = datetime.datetime.now()
        f.players[0].draw(f.players[0].deck, 3)
        f.players[1].draw(f.players[1].deck, 3)
        win, lose, _, _ = G.start(f, virtual_flg=True)

        reward = [win,lose]
        shared_array[3*deck_index + 1] += int(reward[int(episode % 2)] > 0)
        shared_array[3*deck_index + 2] += int(episode%2==0)*int(reward[0] > 0)
        #current_dict = win_rate_dict[current_deck_id]
        #current_dict["battle_num"] += 1
        #current_dict["win_num"] += int(reward[int(episode % 2)] > 0)
        #current_dict["first_win_num"] += int(episode%2==0)*int(reward[0] > 0)

        #train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)

        #win_num += int(reward[int(episode % 2)] > 0)
        #first_num += int(episode%2==0)*int(reward[0] > 0)
    #print(deck_ids,":",win_num/partial_iteration)
    #print("\033[" + str(p_id) + "A", end="")
    #return (deck_ids,win_num/partial_iteration,first_num/(partial_iteration//2))
    return #win_rate_dict

import itertools

def multi_train(data):
    net, memory, batch_size, iteration_num, train_ids,p_num,current_weight_decay = data
    #optimizer =  optim.AdamW(net.parameters(), weight_decay=current_weight_decay)
    #optimizer = AdaBoundW(net.parameters(),lr=0.001)
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
#         first_id = (i*batch_size) % batch_id_len
#         last_id = (i*batch_size+batch_size) % batch_id_len
#         if first_id < last_id:
#             assert last_id - first_id == batch_size,"{},{},{}".format(first_id,last_id,last_id - first_id)
#             key = batch_id_list[first_id:last_id]
#         else:
#             key = batch_id_list[first_id:]+batch_id_list[:last_id]
#         if p_num == 0 and i == 0:print("len:",len(key))
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
        #states['before_states'] = torch.clone(orig_before_states[key])
#         states['before_states'] = {dict_key : torch.clone(orig_before_states[dict_key][key]) for dict_key in normal_states_keys}
#         states['before_states']['values'] = {sub_key: torch.clone(orig_before_states['values'][sub_key][key]) \
#                             for sub_key in value_keys}
        
        actions = all_actions[key]
        rewards = all_rewards[key]

        states['target'] = {'actions': actions, 'rewards': rewards}

        p, v, loss = net(states, target=True)

        #     debug_log = [(v[j],rewards[j]) for j in range(v.size()[0])]
        #     assert False, "all same output!!!\n {}".format(debug_log)
        loss[0].backward(retain_graph=False)

        #         if float(torch.std(v)) < 0.01 and float(torch.std(rewards)) > 0.01 and float(loss[1].item()) > 0.5 and p_num == 0 and i > 10:
        #             for batch_id in range(v.size()[0]):
        #                 print("{}: {} --> {}".format(batch_id,float(v[batch_id]),float(rewards[batch_id])))
        #             print("")
        #             print("{}".format(float(loss[1].item())))
        #             assert False
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
    #states, actions, rewards = memory
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


from emulator_test import *  # importの依存関係により必ず最初にimport
from Field_setting import *
from Player_setting import *
from Policy import *
from Game_setting import Game
if args.limit_OMP:
    half_cpu_num = str(cpu_count()//2)
    os.environ["OMP_NUM_THREADS"] = half_cpu_num
    os.environ["OMP_THREAD_LIMITS"] = half_cpu_num
if args.OMP_NUM > 0:
    os.environ["OMP_NUM_THREADS"] = str(args.OMP_NUM)
def run_main():
    import subprocess
    from torch.utils.tensorboard import SummaryWriter
    print(args)
    p_size = cpu_num
    print("use cpu num:{}".format(p_size))
    print("w_d:{}".format(weight_decay))
    std_th = args.th

    
    loss_history = []

    cuda_flg = args.cuda is not None
    node_num = args.node_num
    net = New_Dual_Net(node_num,rand=args.rand,hidden_num=args.hidden_num[0])
    print(next(net.parameters()).is_cuda)
    
    if args.model_name is not None:
        PATH = 'model/' + args.model_name
        net.load_state_dict(torch.load(PATH))
    if torch.cuda.is_available() and cuda_flg:
        net = net.cuda()
        print(next(net.parameters()).is_cuda)
    net.zero_grad()
    epoch_interval = args.save_interval
    G = Game()
    #Over_all_R = New_Dual_ReplayMemory(100000)
    episode_len = args.episode_num
    batch_size = args.batch_size
    iteration = args.iteration_num
    epoch_num = args.epoch_num
    import datetime
    t1 = datetime.datetime.now()
    print(t1)
    #print(net)
    prev_net = copy.deepcopy(net)
    optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)

    #LOG_PATH = "log_{}_{}_{}_{}_{}_{}/".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
    #                                                         t1.second)
    date = "{}_{}_{}_{}".format(t1.month, t1.day, t1.hour, t1.minute)
    LOG_PATH = "{}episode_{}nodes_deckids{}_{}/".format(episode_len,node_num,args.fixed_deck_ids,date)
    writer = SummaryWriter(log_dir="./logs/" + LOG_PATH)
    TAG="{}_{}_{}".format(episode_len,node_num,args.fixed_deck_ids)
    early_stopper = EarlyStopping(patience=args.loss_th, verbose=True)
    th = args.WR_th
    last_updated = 0
    reset_count = 0
    min_loss = 100
    loss_th = args.loss_th
            
    #pool = Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
        
    #print(torch.cuda.is_available())
    for epoch in range(epoch_num):

        net.cpu()
        prev_net.cpu()
        net.share_memory()

        print("epoch {}".format(epoch + 1))
        t3 = datetime.datetime.now()
        R = New_Dual_ReplayMemory(100000)
        test_R = New_Dual_ReplayMemory(100000)
        episode_len = args.episode_num
        if args.greedy_mode is not None:
            #if args.supervised == "Aggro":
            #    supervise_policy = [AggroPolicy(), AggroPolicy()]
            #else: 
            #    supervise_policy = [New_GreedyPolicy(), New_GreedyPolicy()]
            #p1 = Player(9, True, policy=supervise_policy[0], mulligan=Min_cost_mulligan_policy())
            #p2 = Player(9, False, policy=supervise_policy[1], mulligan=Min_cost_mulligan_policy())
            p1 = Player(9, True, policy=Dual_NN_GreedyPolicy(origin_model=net), mulligan=Min_cost_mulligan_policy())
            p2 = Player(9, False, policy=Dual_NN_GreedyPolicy(origin_model=net), mulligan=Min_cost_mulligan_policy())
        else:

            p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=False,iteration=args.step_iter)
                        ,mulligan=Min_cost_mulligan_policy())
            p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=False,iteration=args.step_iter)
                        ,mulligan=Min_cost_mulligan_policy())

        p1.name = "Alice"
        p2.name = "Bob"
        manager = Manager()
        shared_value = manager.Value("i",0)
        #iter_data = [[p1, p2,shared_value,single_iter,i] for i in range(double_p_size)]
        iter_data = [[p1, p2, shared_value, episode_len, i] for i in range(p_size)]
        freeze_support()
        with Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
            memory = pool.map(multi_preparation, iter_data)
        print("\n" * (p_size+1))
        #pool.terminate()  # add this.
        #pool.close()  # add this.
        # [[result_data,result_data,...,battle_data], ...]
        del p1
        del p2
        del iter_data
        battle_data = [cell.pop(-1) for cell in memory]

        #memories = []
        #[memories.extend(list(itertools.chain.from_iterable(memory[i]))) for i in range(p_size)]
        # [[result_data,result_data,...], [result_data,result_data,...],...]
        sum_of_choice = max(sum([cell["sum_of_choices"] for cell in battle_data]),1)
        sum_of_code = max(sum([cell["sum_code"] for cell in battle_data]),1)
        win_num = sum([cell["win_num"] for cell in battle_data])
        sum_end_turn = sum([cell["end_turn"] for cell in battle_data])
        # [[result_data,result_data,...], [result_data,result_data,...],...]
        #result_data: 一対戦
        origin_memories = list(itertools.chain.from_iterable(memory))
        #origin_memories = random.shuffle(origin_memories)
        # [result_data,result_data,...,result_data,result_data]
        #memories = list(itertools.chain.from_iterable(origin_memories))
        print(type(memory),type(origin_memories),int(episode_len*args.data_rate),len(origin_memories))
        memories = list(itertools.chain.from_iterable(origin_memories[:int(episode_len*args.data_rate)]))
        test_memories = list(itertools.chain.from_iterable(origin_memories[int(episode_len*args.data_rate):]))
        follower_attack_num = 0
        all_able_to_follower_attack = 0

        for data in memories:
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
            hit_flg = int(1 in data[3]['able_to_choice'][10:35])
            all_able_to_follower_attack += hit_flg
            follower_attack_num +=  hit_flg * int(data[1] >= 10 and data[1] <= 34)
            R.push(after_state,data[1], before_state, data[3], data[4])
        for data in test_memories:
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
            hit_flg = int(1 in data[3]['able_to_choice'][10:35])
            all_able_to_follower_attack += hit_flg
            follower_attack_num +=  hit_flg * int(data[1] >= 10 and data[1] <= 34)
            test_R.push(after_state,data[1], before_state, data[3], data[4])


        print("win_rate:{:.3%}".format(win_num/episode_len))
        print("mean_of_num_of_choice:{:.3f}".format(sum_of_choice/sum_of_code))
        print("follower_attack_ratio:{:.3%}".format(follower_attack_num/max(1,all_able_to_follower_attack)))
        print("mean end_turn:{:.3f}".format(sum_end_turn/episode_len))
        print("train_data_size:{}".format(len(R.memory)))
        print("test_data_size:{}".format(len(test_R.memory)))
        net.train()
        prev_net = copy.deepcopy(net)

        p, pai, z, states = None, None, None, None
        batch = len(R.memory) // batch_num if batch_num is not None else batch_size
        print("batch_size:{}".format(batch))
        pass_flg = False
        if args.multi_train is not None:
            if last_updated > args.max_update_interval - 3:
                net = New_Dual_Net(node_num,rand=args.rand,hidden_num=args.hidden_num[0])
                reset_count += 1
                print("reset_num:",reset_count)
            p_size = min(args.cpu_num,3)# if args.cuda is None else 1
            if cuda_flg:
                torch.cuda.empty_cache() 
                net = net.cuda()
            net.share_memory()
            net.train()
            net.zero_grad()
            all_data = R.sample(batch_size,all=True,cuda=cuda_flg,multi=args.multi_sample_num)
            all_states, all_actions, all_rewards = all_data
            memory_len = all_actions.size()[0]
            all_data_ids = list(range(memory_len))
            train_ids = random.sample(all_data_ids, k=memory_len)
            test_data = test_R.sample(batch_size,all=True,cuda=cuda_flg,multi=args.multi_sample_num)
            test_states, test_actions, test_rewards = test_data
            test_memory_len = test_actions.size()[0]
            test_data_range = list(range(test_memory_len))
            test_ids = list(range(test_memory_len))
            min_loss = [0,0.0,100,100,100]
            best_train_data = [100,100,100]
            w_list = args.w_list
            epoch_list = args.epoch_list
            next_net = net#[copy.deepcopy(net) for k in range(len(epoch_list))]
            #[copy.deepcopy(net) for k in range(len(w_list))]
            #iteration_num = int(memory_len//batch)*iteration #(int(memory_len * 0.85) // batch)*iteration
            weight_scale = 0
            freeze_support()
            print("pid:",os.getpid())
            #cmd = "pgrep --parent {} | xargs kill -9".format(int(os.getpid()))
            #proc = subprocess.call( cmd , shell=True)
            with Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
                #for weight_scale in range(len(w_list)):
                for epoch_scale in range(len(epoch_list)):
                    target_net = copy.deepcopy(net)
                    target_net.train()
                    target_net.share_memory()
                    #print("weight_decay:",w_list[weight_scale])
                    print("epoch_num:",epoch_list[epoch_scale])
                    iteration_num = int(memory_len/batch)*epoch_list[epoch_scale]
                    iter_data = [[target_net,all_data,batch,int(iteration_num/p_size),train_ids,i,w_list[weight_scale]]
                                 for i in range(p_size)]
                    torch.cuda.empty_cache()
                    if p_size == 1:
                        loss_data = [multi_train(iter_data[0])]
                    else:
                        freeze_support()
                        #pool = Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
                        loss_data = pool.map(multi_train, iter_data)
                        #pool.terminate()  # add this.
                        #pool.close()  # add this.
                        print("\n" * p_size)
                    #imap = pool.imap(multi_train, iter_data)
                    #loss_data = list(tqdm(imap, total=p_size))
                    #[(1,1,1),(),()]
                    sum_of_loss = sum(map(lambda data: data[0], loss_data))
                    sum_of_MSE = sum(map(lambda data: data[1], loss_data))
                    sum_of_CEE = sum(map(lambda data: data[2], loss_data))
                    train_overall_loss = sum_of_loss / iteration_num
                    train_state_value_loss = sum_of_MSE / iteration_num
                    train_action_value_loss = sum_of_CEE / iteration_num
                    print("AVE | Over_All_Loss(train): {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
                          .format(train_overall_loss, train_state_value_loss, train_action_value_loss))
                #all_states, all_actions, all_rewards = all_data
                    test_ids_len = len(test_ids)
                    #separate_num = test_ids_len
                    separate_num = test_ids_len//batch
                    states_keys = tuple(test_states.keys())#tuple(all_states.keys())
                    value_keys = tuple(test_states['values'].keys())#tuple(all_states['values'].keys())
                    normal_states_keys = tuple(set(states_keys) - {'values', 'detailed_action_codes', 'before_states'})

                    action_code_keys = tuple(test_states['detailed_action_codes'].keys())
                    #tuple(all_states['detailed_action_codes'].keys())

                    target_net.eval()
                    iteration_num = int(memory_len//batch)
                    partition = test_memory_len // p_size
                    iter_data = [[target_net,test_data,batch,
                                  test_data_range[i*partition:min(test_memory_len-1,(i+1)*partition)],i]
                                 for i in range(p_size)]
                    freeze_support()
                    loss_data = pool.map(multi_eval,iter_data)
                    print("\n" * p_size)
                    sum_of_loss = sum(map(lambda data: data[0], loss_data))
                    sum_of_MSE = sum(map(lambda data: data[1], loss_data))
                    sum_of_CEE = sum(map(lambda data: data[2], loss_data))
                    test_overall_loss = sum_of_loss / p_size
                    test_state_value_loss = sum_of_MSE / p_size
                    test_action_value_loss = sum_of_CEE / p_size
                    pass_flg = test_overall_loss > loss_th

                    print("AVE | Over_All_Loss(test ): {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
                          .format(test_overall_loss, test_state_value_loss, test_action_value_loss))
                    
                    target_epoch=epoch_list[epoch_scale]
                    print("debug1:",target_epoch)
                    writer.add_scalars(TAG+"/"+'Over_All_Loss', {'train:'+str(target_epoch): train_overall_loss,
                                                    'test:'+str(target_epoch): test_overall_loss
                                                    }, epoch)
                    writer.add_scalars(TAG+"/"+'state_value_loss', {'train:'+str(target_epoch): train_state_value_loss,
                                                    'test:'+str(target_epoch): test_state_value_loss
                                                    }, epoch)
                    writer.add_scalars(TAG+"/"+'action_value_loss', {'train:'+str(target_epoch): train_action_value_loss,
                                                    'test:'+str(target_epoch): test_action_value_loss
                                                    }, epoch)
                    print("debug2:",target_epoch)
                    if min_loss[2] > test_overall_loss and test_overall_loss > train_overall_loss:
                        next_net = target_net
                        min_loss = [epoch_scale,epoch_list[epoch_scale],test_overall_loss,
                                    test_state_value_loss, test_action_value_loss]
                        print("current best:",min_loss)
                print("finish training")
                pool.terminate()  # add this.
                pool.close()  # add this.
                #print(cmd)
                #proc = subprocess.call( cmd , shell=True)
            print("\n"*p_size +"best_data:",min_loss)
            net = next_net#copy.deepcopy(next_nets[min_loss[0]])
            #del next_net
            loss_history.append(sum_of_loss / iteration)
            p_size = cpu_num


        else:
            prev_optimizer = copy.deepcopy(optimizer)
            if cuda_flg:
                net = net.cuda()
                prev_net = prev_net.cuda()
                optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
                optimizer.load_state_dict(prev_optimizer.state_dict())
                #optimizer = optimizer.cuda()

            current_net = copy.deepcopy(net).cuda() if cuda_flg else copy.deepcopy(net)
            all_data = R.sample(batch_size,all=True,cuda=cuda_flg)
            all_states, all_actions, all_rewards = all_data
            #print("rewards:{}".format(rewards))
            states_keys = list(all_states.keys())
            value_keys = list(all_states['values'].keys())
            normal_states_keys = tuple(set(states_keys) - {'values', 'detailed_action_codes', 'before_states'})
            action_code_keys = list(all_states['detailed_action_codes'].keys())
            memory_len = all_actions.size()[0]
            all_data_ids = list(range(memory_len))
            train_ids = random.sample(all_data_ids,k=int(memory_len*0.8))
            test_ids =list(set(all_data_ids)-set(train_ids))
            #batch_id_list = list(range(memory_len))
            #all_states['target'] = {'actions': all_actions, 'rewards': all_rewards}
            train_num = iteration*len(train_ids)
            nan_count = 0

            for i in tqdm(range(train_num)):
                key = random.sample(train_ids,k=batch)
                states = {}
                states.update({dict_key: torch.clone(all_states[dict_key][key]) for dict_key in normal_states_keys})
                states['values'] = {sub_key: torch.clone(all_states['values'][sub_key][key]) \
                                    for sub_key in value_keys}
                states['detailed_action_codes'] = {
                    sub_key: torch.clone(all_states['detailed_action_codes'][sub_key][key])
                    for sub_key in action_code_keys}
                orig_before_states = all_states["before_states"]
                states['before_states'] = {dict_key: torch.clone(orig_before_states[dict_key][key]) for dict_key in
                                           normal_states_keys}
                states['before_states']['values'] = {sub_key: torch.clone(orig_before_states['values'][sub_key][key]) \
                                                     for sub_key in value_keys}
                
                actions = all_actions[key]
                rewards = all_rewards[key]

                states['target'] = {'actions': actions, 'rewards': rewards}
                net.zero_grad()
                optimizer.zero_grad()
                with detect_anomaly():
                    p, v, loss = net(states, target=True)
                    if True not in torch.isnan(loss[0]):
                        loss[0].backward()
                        optimizer.step()
                        current_net = copy.deepcopy(net)
                        prev_optimizer = copy.deepcopy(optimizer)
                    else:
                        if nan_count < 5:
                            print("loss:{}".format(nan_count))
                            print(loss)
                        net = current_net
                        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
                        optimizer.load_state_dict(prev_optimizer.state_dict())
                        nan_count += 1
            print("nan_count:{}/{}".format(nan_count,train_num))
            train_ids_len = len(train_ids)
            separate_num = train_ids_len
            train_objective_loss = 0
            train_MSE = 0
            train_CEE = 0
            nan_batch_num = 0

            for i in tqdm(range(separate_num)):
                key = [train_ids[i]]
                #train_ids[2*i:2*i+2] if 2*i+2 < train_ids_len else train_ids[train_ids_len-2:train_ids_len]
                states = {}
                states.update({dict_key: torch.clone(all_states[dict_key][key]) for dict_key in normal_states_keys})
                states['values'] = {sub_key: torch.clone(all_states['values'][sub_key][key]) \
                                    for sub_key in value_keys}
                states['detailed_action_codes'] = {
                    sub_key: torch.clone(all_states['detailed_action_codes'][sub_key][key])
                    for sub_key in action_code_keys}
                orig_before_states = all_states["before_states"]
                states['before_states'] = {dict_key: torch.clone(orig_before_states[dict_key][key]) for dict_key in
                                           normal_states_keys}
                states['before_states']['values'] = {sub_key: torch.clone(orig_before_states['values'][sub_key][key]) \
                                                     for sub_key in value_keys}

                actions = all_actions[key]
                rewards = all_rewards[key]
                states['target'] = {'actions': actions, 'rewards': rewards}
                del loss
                torch.cuda.empty_cache()
                _, _, loss = net(states, target=True)
                if True in torch.isnan(loss[0]):
                    if nan_batch_num < 5:
                        print("loss")
                        print(loss)
                    separate_num -= 1
                    nan_batch_num += 1
                    continue
                train_objective_loss += float(loss[0].item())
                train_MSE += float(loss[1].item())
                train_CEE += float(loss[2].item())
            separate_num = max(1,separate_num)
            #writer.add_scalar(LOG_PATH + "WIN_RATE", win_num / episode_len, epoch)
            print("nan_batch_ids:{}/{}".format(nan_batch_num,train_ids_len))
            print(train_MSE,separate_num)
            train_objective_loss /= separate_num
            train_MSE /= separate_num
            train_CEE /= separate_num
            print("AVE(train) | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
                  .format(train_objective_loss,train_MSE,train_CEE))
            test_ids_len = len(test_ids)
            batch_len = 512 if 512 < test_ids_len else 128
            separate_num = test_ids_len // batch_len
            #separate_num = test_ids_len
            test_objective_loss = 0
            test_MSE = 0
            test_CEE = 0
            for i in tqdm(range(separate_num)):
                #key = [test_ids[i]]#test_ids[i*batch_len:min(test_ids_len,(i+1)*batch_len)]
                key = test_ids[i*batch_len:min(test_ids_len,(i*1)*batch_len)]
                states = {}
                states.update({dict_key: torch.clone(all_states[dict_key][key]) for dict_key in normal_states_keys})
                states['values'] = {sub_key: torch.clone(all_states['values'][sub_key][key]) \
                                    for sub_key in value_keys}
                states['detailed_action_codes'] = {
                    sub_key: torch.clone(all_states['detailed_action_codes'][sub_key][key])
                    for sub_key in action_code_keys}
                orig_before_states = all_states["before_states"]
                states['before_states'] = {dict_key: torch.clone(orig_before_states[dict_key][key]) for dict_key in
                                           normal_states_keys}
                states['before_states']['values'] = {sub_key: torch.clone(orig_before_states['values'][sub_key][key]) \
                                                     for sub_key in value_keys}
                actions = all_actions[key]
                rewards = all_rewards[key]
                states['target'] = {'actions': actions, 'rewards': rewards}
                del loss
                torch.cuda.empty_cache()
                p, v, loss = net(states, target=True)
                if True in torch.isnan(loss[0]):
                    separate_num -= 1
                    continue
                test_objective_loss += float(loss[0].item())
                test_MSE += float(loss[1].item())
                test_CEE += float(loss[2].item())
            print("")
            for batch_id in range(1):
                print("states:{}".format(batch_id))
                print("p:{}".format(p[batch_id]))
                print("pi:{}".format(actions[batch_id]))
                print("v:{} z:{}".format(v[batch_id],rewards[batch_id]))
            del p,v
            del actions
            del all_data
            del all_states
            del all_actions
            del all_rewards
            separate_num = max(1, separate_num)
            print(test_MSE,separate_num)
            test_objective_loss /= separate_num
            test_MSE /= separate_num
            test_CEE /= separate_num
            writer.add_scalars(LOG_PATH+'Over_All_Loss', {'train': train_objective_loss,
                                                'test': test_objective_loss
                                                }, epoch)
            writer.add_scalars(LOG_PATH+'MSE', {'train': train_MSE,
                                                'test': test_MSE
                                                }, epoch)
            writer.add_scalars(LOG_PATH+'CEE', {'train': train_CEE,
                                                'test': test_CEE
                                                }, epoch)
            print("AVE | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
                  .format(test_objective_loss, test_MSE, test_CEE))
            
            loss_history.append(test_objective_loss)
            if early_stopper.validate(test_objective_loss): break

        print("evaluate step")
        del R
        del test_R
        net.cpu()
        prev_net.cpu()
        print("evaluate ready")
        if pass_flg:
            min_WR = 0
            WR = 0
            print("evaluation of this epoch is passed.")
        else:
            if args.greedy_mode is not None:
                p1 = Player(9, True, policy=Dual_NN_GreedyPolicy(origin_model=net), mulligan=Min_cost_mulligan_policy())
                p2 = Player(9, False, policy=Dual_NN_GreedyPolicy(origin_model=net), mulligan=Min_cost_mulligan_policy())
            else:
                p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=False,iteration=args.step_iter)
                            ,mulligan=Min_cost_mulligan_policy())

                p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net, cuda=False,iteration=args.step_iter)
                            ,mulligan=Min_cost_mulligan_policy())

            p1.name = "Alice"
            p2.name = "Bob"
            test_deck_list = tuple(100,)  if deck_flg is None else deck_flg# (0,1,4,10,13)
            test_deck_list = tuple(itertools.product(test_deck_list,test_deck_list))
            test_episode_len = evaluate_num#100
            match_num = len(test_deck_list)

            manager = Manager()
            shared_array = manager.Array("i",[0 for _ in range(3*len(test_deck_list))])
            #iter_data = [(p1, p2,test_episode_len, p_id ,cell) for p_id,cell in enumerate(deck_pairs)]
            iter_data = [(p1, p2, shared_array,test_episode_len, p_id, test_deck_list) for p_id in range(p_size)]

            freeze_support()
            with Pool(p_size, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
                _ = pool.map(multi_battle, iter_data)
            print("\n" * (match_num+1))
            del iter_data
            del p1
            del p2
            match_num = len(test_deck_list) #if deck_flg is None else p_size
            min_WR=1.0
            Battle_Result = {(deck_id[0], deck_id[1]): \
                                 tuple(shared_array[3*index+1:3*index+3]) for index, deck_id in enumerate(test_deck_list)}
            #for memory_cell in memory:
            #    #Battle_Result[memory_cell[0]] = memory_cell[1]
            #    #min_WR = min(min_WR,memory_cell[1])
            print(shared_array)
            result_table = {}
            for key in sorted(list((Battle_Result.keys()))):
                cell_WR = Battle_Result[key][0]/test_episode_len
                cell_first_WR = 2*Battle_Result[key][1]/test_episode_len
                print("{}:train_WR:{:.2%},first_WR:{:.2%}"\
                      .format(key,cell_WR,cell_first_WR))
                if key[::-1] not in result_table:
                    result_table[key] = cell_WR
                else:
                    result_table[ key[::-1]] = (result_table[ key[::-1]] + cell_WR)/2
            print(result_table)
            min_WR =  min(result_table.values())
            WR = sum(result_table.values())/len(result_table.values())
            del result_table

        win_flg = False
        #WR=1.0
        writer.add_scalars(TAG+"/"+ 'win_rate', {'mean': WR,
                                              'min': min_WR,
                                              'threthold': th
                                              }, epoch)
        if WR >= th or (len(deck_flg) > 1 and min_WR > 0.5):
            win_flg = True
            print("new_model win! WR:{:.1%} min:{:.1%}".format(WR,min_WR))
        else:
            del net
            net = None
            net = prev_net
            print("new_model lose... WR:{:.1%}".format(WR))
        torch.cuda.empty_cache()
        t4 = datetime.datetime.now()
        print(t4-t3)
        # or (epoch_num > 4 and (epoch+1) % epoch_interval == 0 and epoch+1 < epoch_num)
        if win_flg:
            PATH = "model/{}_{}_{}in{}_{}_nodes.pth".format(t1.month, t1.day, epoch+1,epoch_num,node_num)
            if torch.cuda.is_available() and cuda_flg:
                PATH = "model/{}_{}_{}in{}_{}_nodes_cuda.pth".format(t1.month, t1.day, epoch+1,epoch_num,node_num)
            torch.save(net.state_dict(), PATH)
            print("{} is saved.".format(PATH))
            last_updated = 0
        else:
            last_updated += 1
            print("last_updated:",last_updated)
            if last_updated > args.max_update_interval:
                print("update finished.")
                break
        if len(loss_history) > epoch_interval-1:
            #UB = np.std(loss_history[-epoch_interval:-1])/(np.sqrt(2*epoch) + 1)
            UB = np.std(loss_history) / (np.sqrt(epoch) + 1)
            print("{:<2} std:{}".format(epoch,UB))
            if UB < std_th:
                break

 

    writer.close()
    #pool.terminate()
    #pool.close()
    print('Finished Training')

    PATH = "model/{}_{}_finished_{}_nodes.pth".format(t1.month, t1.day,node_num)
    if torch.cuda.is_available() and cuda_flg:
        PATH = "model/{}_{}_finished_{}_nodes_cuda.pth".format(t1.month, t1.day,node_num)
    torch.save(net.state_dict(), PATH)
    print("{} is saved.".format(PATH))
    t2 = datetime.datetime.now()
    print(t2)
    print(t2-t1)

def check_score():
    print(args)
    p_size = cpu_num
    print("use cpu num:{}".format(p_size))

    loss_history = []
    check_deck_id = int(args.check_deck_id) if args.check_deck_id is not None else None
    cuda_flg = args.cuda == "True"
    #node_num = int(args.node_num)
    #net = New_Dual_Net(node_num)
    model_name = args.model_name
    PATH = 'model/' + model_name
    model_dict=torch.load(PATH)
    n_size=model_dict["final_layer.weight"].size()[1]
    net = New_Dual_Net(n_size,hidden_num=args.hidden_num[0])
    net.load_state_dict(model_dict)
    opponent_net = None
    if args.opponent_model_name is not None:
        #opponent_net = New_Dual_Net(node_num)
        o_model_name = args.opponent_model_name
        PATH = 'model/' + o_model_name
        model_dict=torch.load(PATH)
        n_size=model_dict["final_layer.weight"].size()[1]
        opponent_net = New_Dual_Net(n_size,hidden_num=args.hidden_num[1])
        opponent_net.load_state_dict(model_dict)

    if torch.cuda.is_available() and cuda_flg:
        net = net.cuda()
        opponent_net = opponent_net.cuda() if opponent_net is not None else None
        print("cuda is available.")
    #net.zero_grad()
    deck_sampling_type = False
    if args.deck is not None:
        deck_sampling_type = True
    G = Game()
    net.cpu()
    t3 = datetime.datetime.now()
    if args.greedy_mode is not None:
        p1 = Player(9, True, policy=Dual_NN_GreedyPolicy(origin_model=net))
    else:
        p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net,
                                                                            cuda=cuda_flg,
                                                                            iteration=args.step_iter)
                    , mulligan=Min_cost_mulligan_policy())
    #p1 = Player(9, True, policy=AggroPolicy())
    p1.name = "Alice"
    if fixed_opponent is not None:
        if fixed_opponent == "Aggro":
            p2 = Player(9, False, policy=AggroPolicy(),
                        mulligan=Min_cost_mulligan_policy())
        elif fixed_opponent == "OM":
             p2 = Player(9, False, policy=Opponent_Modeling_ISMCTSPolicy())
        elif fixed_opponent == "NR_OM":
            p2 = Player(9, False, policy=Non_Rollout_OM_ISMCTSPolicy(iteration=200), mulligan=Min_cost_mulligan_policy())
        elif fixed_opponent == "ExItGreedy":
            tmp = opponent_net if opponent_net is not None else net
            p2 = Player(9, False, policy=Dual_NN_GreedyPolicy(origin_model=tmp))
        elif fixed_opponent == "Greedy":
            p2 = Player(9, False, policy=New_GreedyPolicy(), mulligan=Simple_mulligan_policy())
        elif fixed_opponent == "Random":
            p2 = Player(9, False, policy=RandomPolicy(), mulligan=Simple_mulligan_policy())
    else:
        assert opponent_net is not None
        p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=opponent_net, cuda=cuda_flg)
                    , mulligan=Min_cost_mulligan_policy())
    # p2 = Player(9, False, policy=RandomPolicy(), mulligan=Min_cost_mulligan_policy())
    p2.name = "Bob"
    Battle_Result = {}
    deck_list=tuple(map(int,args.deck_list.split(",")))
    print(deck_list)
    test_deck_list = deck_list# (0,1,4,10,13)
    test_deck_list = tuple(itertools.product(test_deck_list,test_deck_list))
    test_episode_len = evaluate_num#100
    episode_num = evaluate_num
    match_num = len(test_deck_list)
    manager = Manager()
    shared_array = manager.Array("i",[0 for _ in range(3*len(test_deck_list))])
    iter_data = [(p1, p2, shared_array,episode_num, p_id, test_deck_list) for p_id in range(p_size)]
    freeze_support()
    print(p1.policy.name)
    print(p2.policy.name)
    pool = Pool(p_size, initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
    memory = pool.map(multi_battle, iter_data)
    pool.close()  # add this.
    pool.terminate()  # add this.
    print("\n" * (match_num + 1))
    memory = list(memory)
    min_WR=1.0
    Battle_Result = {(deck_id[0], deck_id[1]): \
                         tuple(shared_array[3*index+1:3*index+3]) for index, deck_id in enumerate(test_deck_list)}
    print(shared_array)
    txt_dict = {}
    for key in sorted(list((Battle_Result.keys()))):
        cell = "{}:WR:{:.2%},first_WR:{:.2%}"\
              .format(key,Battle_Result[key][0]/test_episode_len,2*Battle_Result[key][1]/test_episode_len)
        print(cell)
        txt_dict[key] = cell
    print(Battle_Result)
    result_name = model_name.split(".")[0] + ":" + args.deck_list
    deck_num = len(deck_list)
    os.makedirs("Battle_Result", exist_ok=True)
    with open("Battle_Result/" + result_name, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        row = ["{} vs {}".format(p1.policy.name, p2.policy.name)]
        deck_names = [deck_id_2_name[deck_list[i]] for i in range(deck_num)]
        row = row + deck_names
        writer.writerow(row)
        for i in deck_list:
            row = [deck_id_2_name[i]]
            for j in deck_list:
                row.append(Battle_Result[(i, j)])
            writer.writerow(row)
        for key in list(txt_dict.keys()):
            writer.writerow([txt_dict[key]])

if __name__ == "__main__":
    if args.check is not None:
        check_score()
    else:
        run_main()


