from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock,freeze_support

try:
    set_start_method('spawn')
    print("spawn is run.")
    #set_start_method('fork') GPU使用時CUDA initializationでerror
    #print('fork')
except RuntimeError:
    pass

from test import *  # importの依存関係により必ず最初にimport
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
GAMMA = 0.9
parser = argparse.ArgumentParser(description='デュアルニューラルネットワーク学習コード')

parser.add_argument('--episode_num', help='試行回数')
parser.add_argument('--iteration_num', help='イテレーション数')
parser.add_argument('--epoch_num', help='エポック数')
parser.add_argument('--batch_size', help='バッチサイズ')
parser.add_argument('--mcts', help='サンプリングAIをMCTSにする(オリジナルの場合は[OM])')
parser.add_argument('--deck', help='サンプリングに用いるデッキの選び方')
parser.add_argument('--cuda', help='gpuを使用するかどうか')
parser.add_argument('--multi_train', help="学習時も並列化するかどうか")
parser.add_argument('--epoch_interval', help="モデルの保存間隔")
parser.add_argument('--fixed_deck_ids', help="使用デッキidリストの固定")
parser.add_argument('--cpu_num', help="使用CPU数",default=2 if torch.cuda.is_available() else 3)
parser.add_argument('--batch_num', help='サンプルに対するバッチの数')
parser.add_argument('--fixed_opponent', help='対戦相手を固定')
parser.add_argument('--node_num', help='node_num', default=100)
parser.add_argument('--weight_decay', help='weight_decay', default=1e-2)
parser.add_argument('--check', help='check score')
parser.add_argument('--deck_list', help='deck_list',default="0,1,4,5,10,11")
parser.add_argument('--model_name', help='model_name', default=None)
parser.add_argument('--opponent_model_name', help='opponent_model_name', default=None)
parser.add_argument('--th', help='threshold',default=1e-3)
parser.add_argument('--WR_th', help='WR_threshold',default=0.55)
args = parser.parse_args()

deck_flg = list(map(int,args.fixed_deck_ids.split(","))) if args.fixed_deck_ids is not None else None
weight_decay = float(args.weight_decay)


#Detailed_State_data = namedtuple('Value', ('hand_ids', 'hand_card_costs', 'follower_card_ids',
#                                           'amulet_card_ids', 'follower_stats', 'follower_abilities', 'able_to_evo',
#                                           'life_data', 'pp_data', 'able_to_play', 'able_to_attack',
#                                           'able_to_creature_attack'))
cpu_num = int(args.cpu_num)
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
    d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
    d2.set_leader_class(key_2_tsv_name[deck_type2][1])
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
    partial_iteration = episode_data[-2]
    p_num = episode_data[-1]
    info = f'#{p_num:>2} '
    all_result_data = []
    battle_data = {"sum_of_choices":0, "sum_code":0, "win_num":0,"end_turn":0}
    for episode in tqdm(range(partial_iteration),desc=info,position=p_num):
        #f = Field(5)
        #p1 = episode_data[0].get_copy(f)
        #p2 = episode_data[1].get_copy(f)
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
        d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
        d2.set_leader_class(key_2_tsv_name[deck_type2][1])
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
            end_turn = int(train_data[i][-1][0].life_data[0][-1]*100)
            for data in train_data[i]:
                # assert False,"{}".format(data[0])
                detailed_action_code = data[3]
                #if sum(detailed_action_code["able_to_choice"]) == 1:
                #    continue
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
                sum_of_choices += sum(detailed_action_code['able_to_choice'])
                sum_code += 1
                #current_turn = int(data[0].life_data[0][-1] * 100)
                #discount_rate = GAMMA**(end_turn-current_turn)
                discounted_reward = reward[i]# * discount_rate
                result_data.append((before_state, action_probability, after_state, detailed_action_code,discounted_reward))#, reward[i]))
                #print("life_data:{}".format(data[2].life_data))
                #end_turn = data[2].life_data[0][-1]
                #result_data.append((before_state, action_probability, after_state, detailed_action_code, reward[1-i]))

        battle_data["sum_of_choices"] += sum_of_choices
        battle_data["sum_code"] += sum_code
        battle_data["win_num"] += int(reward[int(episode % 2)] > 0)
        battle_data["end_turn"] += end_turn
        all_result_data.append(result_data)
    all_result_data.append(battle_data)
    #print("\033["+str(p_num)+"A", end="")
    return all_result_data

import itertools
def multi_battle(episode_data):
    partial_iteration = episode_data[-3]
    p_id = episode_data[-2]
    deck_ids = episode_data[-1]
    win_num = 0
    #first_num = [0,0]
    info = f'#{str(deck_ids):>8} '
    for episode in tqdm(range(partial_iteration),desc=info,position=p_id):
        f = Field(5)
        p1 = episode_data[episode%2].get_copy(f)
        p2 = episode_data[1-(episode%2)].get_copy(f)
        p1.is_first = True
        p2.is_first = False
        p1.player_num = 0
        p2.player_num = 1
        deck_type1 = deck_ids[episode%2]
        deck_type2 = deck_ids[1-episode%2]
        d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
        d1.set_leader_class(key_2_tsv_name[deck_type1][1])
        d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
        d2.set_leader_class(key_2_tsv_name[deck_type2][1])
        d1.shuffle()
        d2.shuffle()
        p1.deck = d1
        p2.deck = d2
        f.players = [p1, p2]
        p1.field = f
        p2.field = f
        x1 = datetime.datetime.now()
        f.players[0].draw(f.players[0].deck, 3)
        f.players[1].draw(f.players[1].deck, 3)
        win, lose, _, _ = G.start(f, virtual_flg=True)
        reward = [win,lose]
        #train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)
        win_num += int(reward[int(episode % 2)] > 0)
        #first_num[episode%2] += int(reward[0] > 0)
    #print(deck_ids,":",win_num/partial_iteration)
    #print("\033[" + str(p_id) + "A", end="")
    return (deck_ids,win_num/partial_iteration)

import itertools

def multi_train(data):
    net, memory, batch_size, iteration_num, train_ids,p_num = data
    optimizer =  optim.Adam(net.parameters(), weight_decay=weight_decay)
    all_loss, MSE, CEE = 0, 0, 0

    all_states, all_actions, all_rewards = memory
    states_keys = list(all_states.keys())
    value_keys = list(all_states['values'].keys())
    action_code_keys = list(all_states['detailed_action_codes'].keys())
    #memory_len = all_actions.size()[0]
    batch_id_list = train_ids#list(range(memory_len))
    all_states['target'] = {'actions': all_actions, 'rewards': all_rewards}
    info = f'#{p_num:>2} '
    for i in tqdm(range(iteration_num),desc=info,position=p_num):
        optimizer.zero_grad()
        states = all_states
        #key = [random.randint(0, memory_len-1) for _ in range(batch_size)]

        key = random.sample(batch_id_list,k=batch_size)
        #key = random.sample(train_ids, k=batch)
        states = {}
        for dict_key in states_keys:
            if dict_key == 'values':
                states['values'] = {}
                for sub_key in value_keys:
                    states['values'][sub_key] = torch.clone(all_states['values'][sub_key][key])
                    # states['values'][sub_key].grad=None
            elif dict_key == 'detailed_action_codes':
                states['detailed_action_codes'] = {}
                for sub_key in action_code_keys:
                    states['detailed_action_codes'][sub_key] = \
                        torch.clone(all_states['detailed_action_codes'][sub_key][key])
                    # states['detailed_action_codes'][sub_key].grad=None
            elif dict_key == 'before_states':
                orig_before_states = all_states["before_states"]
                before_states = {}
                for dict_key in states_keys:
                    if dict_key == 'values':
                        before_states['values'] = {}
                        for sub_key in value_keys:
                            before_states['values'][sub_key] = \
                                torch.clone(orig_before_states['values'][sub_key][key])
                            # states['values'][sub_key].grad=None
                    elif dict_key == 'detailed_action_codes' or dict_key == "before_states":
                        pass
                    else:
                        before_states[dict_key] = torch.clone(orig_before_states[dict_key][key])
                states["before_states"] = before_states
            else:
                states[dict_key] = torch.clone(all_states[dict_key][key])
                # states[dict_key].grad=None

        actions = all_actions[key]
        rewards = all_rewards[key]

        states['target'] = {'actions': actions, 'rewards': rewards}

        actions = all_actions[key]
        rewards = all_rewards[key]
        states['target'] = {'actions': actions, 'rewards': rewards}


        p, v, loss = net(states, target=True)
        z = all_rewards
        pai = all_actions  # 45種類の抽象化した行動
        # loss.backward()
        loss[0].backward()
        all_loss += float(loss[0].item())
        MSE += float(loss[1].item())
        CEE += float(loss[2].item())

        optimizer.step()


    return all_loss, MSE, CEE


def multi_eval(data):
    net, memory, batch_size, iteration_num, p_num = data
    all_loss, MSE, CEE = 0, 0, 0

    all_states, all_actions, all_rewards = memory
    states_keys = list(all_states.keys())
    value_keys = list(all_states['values'].keys())
    action_code_keys = list(all_states['detailed_action_codes'].keys())
    memory_len = all_actions.size()[0]
    batch_id_list = list(range(memory_len))

    #states, actions, rewards = memory
    info = f'#{p_num:>2} '
    for i in tqdm(range(iteration_num),desc=info,position=p_num+1):
        #key = [random.randint(0, memory_len-1) for _ in range(batch_size)]
        key = random.sample(batch_id_list,k=batch_size)
        states = {}
        for dict_key in states_keys:
            if dict_key == 'values':
                states['values'] = {}
                for sub_key in value_keys:
                    states['values'][sub_key] = all_states['values'][sub_key][key]
            elif dict_key == 'detailed_action_codes':
                states['detailed_action_codes'] = {}
                for sub_key in action_code_keys:
                    states['detailed_action_codes'][sub_key] = \
                        all_states['detailed_action_codes'][sub_key][key]
            else:
                states[dict_key] = all_states[dict_key][key]

        actions = all_actions[key]
        rewards = all_rewards[key]
        states['target'] = {'actions': actions, 'rewards': rewards}

        p, v, loss = net(states, target=True)
        z = rewards
        pai = actions  # 45種類の抽象化した行動
        # loss.backward()
        all_loss += float(loss[0].item())
        MSE += float(loss[1].item())
        CEE += float(loss[2].item())
        #if (i+1) % (iteration_num//10) == 0:
        #    #print("action:{}\n{}".format(actions[0],p[0]))
        #    #print("value:{}\n{}".format(z[0],v[0]))
        #    print("{}0% finished.".format((i+1) // (iteration_num//10)))
    return all_loss, MSE, CEE


from test import *  # importの依存関係により必ず最初にimport
from Field_setting import *
from Player_setting import *
from Policy import *
from Game_setting import Game

def run_main():
    from torch.utils.tensorboard import SummaryWriter
    print(args)
    p_size = cpu_num
    print("use cpu num:{}".format(p_size))
    print("w_d:{}".format(weight_decay))
    std_th = float(args.th)

    loss_history = []

    cuda_flg = args.cuda == "True"
    node_num = int(args.node_num)
    net = New_Dual_Net(node_num)
    if torch.cuda.is_available() and cuda_flg:
        net = net.cuda()
        print("cuda is available.")
    net.zero_grad()
    deck_sampling_type = False
    if args.deck is not None:
        deck_sampling_type = True
    epoch_interval = int(args.epoch_interval) if args.epoch_interval is not None else 10
    G = Game()
    #Over_all_R = New_Dual_ReplayMemory(100000)
    episode_len = 100
    if args.episode_num is not None:
        episode_len = int(args.episode_num)
    batch_size = 100
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    iteration = 100
    if args.iteration_num is not None:
        iteration = int(args.iteration_num)
    epoch_num = 2
    if args.epoch_num is not None:
        epoch_num = int(args.epoch_num)
    mcts = False
    if args.mcts is not None:
        mcts = True
    import datetime
    t1 = datetime.datetime.now()
    print(t1)
    #print(net)
    prev_net = copy.deepcopy(net)
    optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)

    LOG_PATH = "log_{}_{}_{}_{}_{}_{}/".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                             t1.second)
    writer = SummaryWriter(log_dir="./logs/" + LOG_PATH)
    th = float(args.WR_th)
    fail_count = 0
    last_updated = 0
    min_loss = 100
    double_p_size = 2*p_size
    #print(torch.cuda.is_available())
    for epoch in range(epoch_num):
        net.cpu()
        #net.class_eye.cpu()
        #et.ability_eye.cpu()
        prev_net.cpu()

        print("epoch {}".format(epoch + 1))
        t3 = datetime.datetime.now()
        R = New_Dual_ReplayMemory(100000)
        #p1 = Player(9, True, policy=Dual_NN_GreedyPolicy(origin_model=net))
        if False:# epoch < 5:
            p1 = Player(9, True, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
            #p1 = Player(9, True, policy=Opponent_Modeling_ISMCTSPolicy())
        #else:
        p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=cuda_flg)
                    ,mulligan=Min_cost_mulligan_policy())
        #p1 = Player(9, True, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
        p1.name = "Alice"
        p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=cuda_flg)
                    ,mulligan=Min_cost_mulligan_policy())
        #p2 = Player(9, False, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
        p2.name = "Bob"

        #import cProfile
        #cProfile.run("memories = multi(episode_len,p1,p2)",sort="tottime")
        #assert False
        #memories = multi(episode_len,p1,p2)
        #iter_data = [(i, p1, p2) for i in range(episode_len)]
        #p_size = 5 if epoch < 5 else cpu_num
        single_iter = episode_len//double_p_size
        iter_data = [(p1, p2,single_iter,i) for i in range(double_p_size)]
        freeze_support()
        pool = Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
        memory = pool.map(multi_preparation, iter_data)
        print("\n" * (double_p_size+1))
        pool.close()  # add this.
        pool.terminate()  # add this.
        battle_data = [cell.pop(-1) for cell in memory]
        memories = []
        [memories.extend(list(itertools.chain.from_iterable(memory[i]))) for i in range(double_p_size)]

        sum_of_choice = max(sum([cell["sum_of_choices"] for cell in battle_data]),1)
        sum_of_code = max(sum([cell["sum_code"] for cell in battle_data]),1)
        win_num = sum([cell["win_num"] for cell in battle_data])
        sum_end_turn = sum([cell["end_turn"] for cell in battle_data])
        memories = list(itertools.chain.from_iterable(memory))
        memories = list(itertools.chain.from_iterable(memories))
        follower_attack_num = 0
        all_able_to_follower_attack = 0

        for data in memories:
            #print(data[0])
            before_state = Detailed_State_data(data[0]['hand_ids'], data[0]['hand_card_costs'],
                            data[0]['follower_card_ids'], data[0]['amulet_card_ids'],
                            data[0]['follower_stats'], data[0]['follower_abilities'],
                            data[0]['able_to_evo'], data[0]['life_data'],
                            data[0]['pp_data'], data[0]['able_to_play'],
                            data[0]['able_to_attack'], data[0]['able_to_creature_attack'],data[0]['deck_data'])
            after_state = Detailed_State_data(data[2]['hand_ids'], data[2]['hand_card_costs'],
                            data[2]['follower_card_ids'], data[2]['amulet_card_ids'],
                            data[2]['follower_stats'], data[2]['follower_abilities'],
                            data[2]['able_to_evo'], data[2]['life_data'],
                            data[2]['pp_data'], data[2]['able_to_play'],
                            data[2]['able_to_attack'], data[2]['able_to_creature_attack'],data[2]['deck_data'])
            hit_flg = int(1 in data[3]['able_to_choice'][10:35])
            all_able_to_follower_attack += hit_flg
            follower_attack_num +=  hit_flg * int(data[1] >= 10 and data[1] <= 34)
            R.push(before_state,data[1], after_state, data[3], data[4])


        print("win_rate:{:.3%}".format(win_num/episode_len))
        print("mean_of_num_of_choice:{:.3f}".format(sum_of_choice/sum_of_code))
        print("follower_attack_ratio:{:.3%}".format(follower_attack_num/max(1,all_able_to_follower_attack)))
        print("mean end_turn:{:.3f}".format(sum_end_turn/episode_len))
        print("sample_size:{}".format(len(R.memory)))
        net.train()
        prev_net = copy.deepcopy(net)

        sum_of_loss = 0
        sum_of_MSE = 0
        sum_of_CEE = 0
        p, pai, z, states = None, None, None, None
        batch = len(R.memory) // batch_num if batch_num is not None else batch_size
        print("batch_size:{}".format(batch))
        if args.multi_train is not None:
            p_size = cpu_num
            if cuda_flg:
                net = net.cuda()
            net.share_memory()

            all_data = R.sample(batch_size,all=True,cuda=cuda_flg)
            all_states, all_actions, all_rewards = all_data
            memory_len = all_actions.size()[0]
            all_data_ids = list(range(memory_len))
            train_ids = random.sample(all_data_ids, k=int(memory_len * 0.8))
            iter_data = [[net,all_data,batch,iteration//p_size,train_ids,i]
                         for i in range(p_size)]
            freeze_support()
            pool = Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
            loss_data = pool.map(multi_train, iter_data)
            print("\n" * p_size)
            #imap = pool.imap(multi_train, iter_data)
            #loss_data = list(tqdm(imap, total=p_size))
            #[(1,1,1),(),()]
            sum_of_loss = sum(map(lambda data: data[0], loss_data))
            sum_of_MSE = sum(map(lambda data: data[1], loss_data))
            sum_of_CEE = sum(map(lambda data: data[2], loss_data))
            train_objective_loss = sum_of_loss / iteration
            train_MSE = sum_of_MSE / iteration
            train_CEE = sum_of_CEE / iteration
            print("AVE | Over_All_Loss(train): {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
                  .format(train_objective_loss, train_MSE, train_CEE))
            all_states, all_actions, all_rewards = all_data
            all_states['target'] = {'actions': all_actions, 'rewards': all_rewards}
            p, v, loss = net(all_states, target=True)

            print("loss:{:.3f} MSE:{:.3f} CEE:{:.3f}".format(loss[0].item(), loss[1].item(), loss[2].item()))
            test_objective_loss = loss[0].item()
            test_MSE = loss[1].item()
            test_CEE = loss[2].item()
            pool.close()  # add this.
            pool.terminate()  # add this.

            #writer.add_scalar(LOG_PATH + "Over_All_Loss", sum_of_loss / iteration, epoch)
            #writer.add_scalar(LOG_PATH + "MSE", sum_of_MSE / iteration, epoch)
            #writer.add_scalar(LOG_PATH + "CEE", sum_of_CEE / iteration, epoch)
            #writer.add_scalar(LOG_PATH + "WIN_RATE", win_num / episode_len, epoch)
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
                  .format(sum_of_loss / iteration, sum_of_MSE / iteration, sum_of_CEE / iteration))
            loss_history.append(sum_of_loss / iteration)

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
                for dict_key in states_keys:
                    if dict_key == 'values':
                        states['values'] = {}
                        for sub_key in value_keys:
                            states['values'][sub_key] = torch.clone(all_states['values'][sub_key][key])
                            #states['values'][sub_key].grad=None
                    elif dict_key == 'detailed_action_codes':
                        states['detailed_action_codes'] = {}
                        for sub_key in action_code_keys:
                            states['detailed_action_codes'][sub_key] = \
                                torch.clone(all_states['detailed_action_codes'][sub_key][key])
                            #states['detailed_action_codes'][sub_key].grad=None
                    elif dict_key == 'before_states':
                        orig_before_states = all_states["before_states"]
                        before_states = {}
                        for dict_key in states_keys:
                            if dict_key == 'values':
                                before_states['values'] = {}
                                for sub_key in value_keys:
                                    before_states['values'][sub_key] = \
                                        torch.clone(orig_before_states['values'][sub_key][key])
                                    # states['values'][sub_key].grad=None
                            elif dict_key == 'detailed_action_codes' or dict_key == "before_states":
                                pass
                            else:
                                before_states[dict_key] = torch.clone(orig_before_states[dict_key][key])
                        states["before_states"] = before_states
                    else:
                        states[dict_key] = torch.clone(all_states[dict_key][key])
                        #states[dict_key].grad=None
                
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
                key = [train_ids[i]]#train_ids[2*i:2*i+2] if 2*i+2 < train_ids_len else train_ids[train_ids_len-2:train_ids_len]#train_ids[i*10:min(train_ids_len,(i+1)*10)] # [batch_id_list[(j+i*batch)%memory_len] for j in range(batch)]
                states = {}
                for dict_key in states_keys:
                    if dict_key == 'values':
                        states['values'] = {}
                        for sub_key in value_keys:
                            states['values'][sub_key] = all_states['values'][sub_key][key]
                    elif dict_key == 'detailed_action_codes':
                        states['detailed_action_codes'] = {}
                        for sub_key in action_code_keys:
                            states['detailed_action_codes'][sub_key] = \
                                all_states['detailed_action_codes'][sub_key][key]
                    elif dict_key == 'before_states':
                        orig_before_states = all_states["before_states"]
                        before_states = {}
                        for dict_key in states_keys:
                            if dict_key == 'values':
                                before_states['values'] = {}
                                for sub_key in value_keys:
                                    before_states['values'][sub_key] = \
                                        torch.clone(orig_before_states['values'][sub_key][key])
                                    # states['values'][sub_key].grad=None
                            elif dict_key == 'detailed_action_codes' or dict_key == "before_states":
                                pass
                            else:
                                before_states[dict_key] = torch.clone(orig_before_states[dict_key][key])
                        states["before_states"] = before_states
                    else:
                        states[dict_key] = all_states[dict_key][key]

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
            batch_len = 100 if 100 < test_ids_len else 10
            separate_num = test_ids_len // batch_len
            separate_num = test_ids_len
            test_objective_loss = 0
            test_MSE = 0
            test_CEE = 0
            for i in tqdm(range(separate_num)):
                key = [test_ids[i]]#test_ids[i*batch_len:min(test_ids_len,(i+1)*batch_len)] # [batch_id_list[(j+i*batch)%memory_len] for j in range(batch)]
                states = {}
                for dict_key in states_keys:
                    if dict_key == 'values':
                        states['values'] = {}
                        for sub_key in value_keys:
                            states['values'][sub_key] = all_states['values'][sub_key][key]
                    elif dict_key == 'detailed_action_codes':
                        states['detailed_action_codes'] = {}
                        for sub_key in action_code_keys:
                            states['detailed_action_codes'][sub_key] = \
                                all_states['detailed_action_codes'][sub_key][key]
                    elif dict_key == 'before_states':
                        orig_before_states = all_states["before_states"]
                        before_states = {}
                        for dict_key in states_keys:
                            if dict_key == 'values':
                                before_states['values'] = {}
                                for sub_key in value_keys:
                                    before_states['values'][sub_key] = \
                                        torch.clone(orig_before_states['values'][sub_key][key])
                                    # states['values'][sub_key].grad=None
                            elif dict_key == 'detailed_action_codes' or dict_key == "before_states":
                                pass
                            else:
                                before_states[dict_key] = torch.clone(orig_before_states[dict_key][key])
                        states["before_states"] = before_states
                    else:
                        states[dict_key] = all_states[dict_key][key]

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

        net.cpu()
        prev_net.cpu()

        p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net, cuda=cuda_flg)
                    ,mulligan=Min_cost_mulligan_policy())
        p1.name = "Alice"
        p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net, cuda=cuda_flg)
                    ,mulligan=Min_cost_mulligan_policy())
        p2.name = "Bob"
        constant_deck_list = (0, 1, 2, 4, 5, 10, 12)  # (0,1,4,10,13)
        test_episode_len = int(400/p_size) if deck_flg is not None else 40#2*episode_len
        
        #iter_data = [(p1, p2,test_episode_len//p_size,i) for i in range(p_size)]
        #freeze_support()
        #pool = Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
        #memory = pool.map(multi_preparation, iter_data)
        #print("\n" * p_size)
        #pool.close()  # add this.
        #pool.terminate()  # add this.

        deck_pairs = ((d,d) for d in constant_deck_list) if deck_flg is None else ((deck_flg,deck_flg) for _ in range(p_size))
        iter_data = [(p1, p2, test_episode_len, p_id ,cell) for p_id,cell in enumerate(deck_pairs)]
        freeze_support()
        pool = Pool(p_size, initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
        memory = pool.map(multi_battle, iter_data)
        print("\n" * (p_size+1))
        # Battle_Results[(j, k)] = [win_lose[0] / iteration, first_num / iteration]

        pool.close()  # add this.
        pool.terminate()  # add this.
        memory = list(memory)
        match_num = len(constant_deck_list) if deck_flg is None else p_size
        min_WR=1.0
        if deck_flg is None:
            Battle_Result = {}
            for memory_cell in memory:
                Battle_Result[memory_cell[0]] = memory_cell[1]
                min_WR = min(min_WR,memory_cell[1])
            print(Battle_Result)
            WR = sum(Battle_Result.values())/match_num
        else:
            WR_list = [cell[1] for cell in memory]
            print(WR_list)
            WR = sum(WR_list)/match_num
            min_WR = WR


        #battle_data = [cell.pop(-1) for cell in memory]
        #win_num = sum([cell["win_num"] for cell in battle_data])

        win_flg = False
        #WR=1.0
        writer.add_scalars(LOG_PATH + 'win_rate', {'mean': WR,
                                              'min': min_WR
                                              }, epoch)
        if WR < max(0.51,th*100/(100+fail_count)):# and min_WR < 0.5:
            net = prev_net
            fail_count += 1
            #th = max(0.5,th*0.95)
            print("new_model lose... WR:{:.1%} fail_count:{}".format(WR,fail_count))
        else:
            fail_count = 0
            #th = 0.55
            win_flg = True
            print("new_model win! WR:{:.1%} min:{:.1%}".format(WR,min_WR))
        #writer.add_scalar(LOG_PATH + 'WR', WR, epoch)






        t4 = datetime.datetime.now()
        print(t4-t3)
        if win_flg or (epoch_num > 4 and (epoch+1) % epoch_interval == 0 and epoch+1 < epoch_num):
            PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_{}|{}_{}nodes.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                 t1.second, epoch+1,epoch_num,node_num)
            if torch.cuda.is_available() and cuda_flg:
                PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_{}|{}_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                        t1.second, epoch + 1 , epoch_num)
            torch.save(net.state_dict(), PATH)
            print("{} is saved.".format(PATH))
        if len(loss_history) > epoch_interval-1:
            #UB = np.std(loss_history[-epoch_interval:-1])/(np.sqrt(2*epoch) + 1)
            UB = np.std(loss_history) / (np.sqrt(epoch) + 1)
            print("{:<2} std:{}".format(epoch,UB))
            if UB < std_th:
                break
        if min_loss > test_objective_loss:
            last_updated = 0
            min_loss = test_objective_loss
        else:
            last_updated += 1
            if last_updated > 20:
                print("update finished.")
                break

    writer.close()
    print('Finished Training')

    PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_all_{}nodes.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                         t1.second,node_num)
    if torch.cuda.is_available() and cuda_flg:
        PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_all_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                             t1.second)
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

    cuda_flg = args.cuda == "True"
    node_num = int(args.node_num)
    net = New_Dual_Net(node_num)
    model_name = args.model_name
    PATH = 'model/' + model_name
    net.load_state_dict(torch.load(PATH))
    opponent_net = None
    if args.opponent_model_name is not None:
        opponent_net = New_Dual_Net(node_num)
        model_name = args.opponent_model_name
        PATH = 'model/' + model_name
        opponent_net.load_state_dict(torch.load(PATH))

    if torch.cuda.is_available() and cuda_flg:
        net = net.cuda()
        opponent_net = opponent_net.cuda() if opponent_net is not None else None
        print("cuda is available.")
    #net.zero_grad()
    deck_sampling_type = False
    if args.deck is not None:
        deck_sampling_type = True
    G = Game()
    episode_len = 100
    if args.episode_num is not None:
        episode_len = int(args.episode_num)
    net.cpu()
    t3 = datetime.datetime.now()
    p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net, cuda=cuda_flg)
                , mulligan=Min_cost_mulligan_policy())
    #p1 = Player(9, True, policy=AggroPolicy())
    p1.name = "Alice"
    if fixed_opponent is not None:
        if fixed_opponent == "Aggro":
            p2 = Player(9, False, policy=AggroPolicy())
        elif fixed_opponent == "OM":
            # p1 = Player(9, True, policy=Opponent_Modeling_MCTSPolicy())
            # p1.name = "Alice"
            p2 = Player(9, False, policy=Opponent_Modeling_ISMCTSPolicy(),
                        mulligan=Min_cost_mulligan_policy())
        else:
            p2 = Player(9, False, policy=Dual_NN_GreedyPolicy(origin_model=prev_net))
    else:
        if opponent_net is not None:  # epoch < 5:
            p2 = Player(9, False, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
            # p2 = Player(9, False, policy=Opponent_Modeling_ISMCTSPolicy())
        else:
            p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=opponent_net, cuda=cuda_flg)
                        , mulligan=Min_cost_mulligan_policy())
    # p2 = Player(9, False, policy=RandomPolicy(), mulligan=Min_cost_mulligan_policy())
    p2.name = "Bob"
    Battle_Result = {}
    deck_list=list(map(int,args.deck_list.split(",")))
    print(deck_list)
    deck_pairs = list(itertools.product(deck_list,deck_list))

    iter_data = [(p1, p2, episode_len, cell_id,cell) for cell_id,cell in enumerate(deck_pairs)]
    freeze_support()
    pool = Pool(p_size, initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
    memory = pool.map(multi_battle, iter_data)
    #Battle_Results[(j, k)] = [win_lose[0] / iteration, first_num / iteration]

    pool.close()  # add this.
    pool.terminate()  # add this.
    print("\n" * (len(deck_pairs) + 1))
    memory = list(memory)
    for memory_cell in memory:
        Battle_Result[memory_cell[0]] = memory_cell[1]
    print(Battle_Result)
    result_name = model_name.split(".")[0] + ":" + args.deck_list
    deck_num = len(deck_list)
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
    #win_rate = sum([cell[0] for cell in memory])/episode_len
    #first_win_rate = [2*sum([cell[1][0] for cell in memory])/episode_len,
    #                  2 * sum([cell[1][1] for cell in memory]) / episode_len]
    #print("win_rate:{:.3%},first_win_rate:{:.3%} {:.3%}".format(win_rate,first_win_rate[0],first_win_rate[1]))


if __name__ == "__main__":
    if args.check is not None:
        check_score()
    else:
        run_main()


