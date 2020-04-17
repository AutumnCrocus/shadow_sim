from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock

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
from torch.utils.tensorboard import SummaryWriter

from Embedd_Network_model import *
import copy
import datetime
# net = New_Dual_Net(100)
import os

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
parser.add_argument('--fixed_deck_id', help="使用デッキidの固定")
parser.add_argument('--cpu_num', help="使用CPU数",default=2 if torch.cuda.is_available() else 3)
parser.add_argument('--batch_num', help='サンプルに対するバッチの数')
parser.add_argument('--fixed_opponent', help='対戦相手を固定')
args = parser.parse_args()

deck_flg = int(args.fixed_deck_id) if args.fixed_deck_id is not None else None

#Detailed_State_data = namedtuple('Value', ('hand_ids', 'hand_card_costs', 'follower_card_ids',
#                                           'amulet_card_ids', 'follower_stats', 'follower_abilities', 'able_to_evo',
#                                           'life_data', 'pp_data', 'able_to_play', 'able_to_attack',
#                                           'able_to_creature_attack'))
cpu_num = int(args.cpu_num)
batch_num = int(args.batch_num) if args.batch_num is not None else None
G = Game()
fixed_opponent = args.fixed_opponent
def preparation(episode_data):
    episode = episode_data[0]
    #print("start:{}".format(episode + 1))
    f = Field(5)
    p1 = episode_data[1].get_copy(f)
    p2 = episode_data[2].get_copy(f)
    #if random.random() < 0.05:
    #    p1 = Player(9,True)
    #if random.random() < 0.05:
    #    p2 = Player(9, False)
    if deck_flg is None:
        deck_type1 = random.choice(list(key_2_tsv_name.keys()))
        deck_type2 = random.choice(list(key_2_tsv_name.keys()))
    else:
        deck_type1 = deck_flg
        deck_type2 = deck_flg
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

            result_data.append((before_state, action_probability, after_state, detailed_action_code, reward[i]))

    x2 = datetime.datetime.now()

    win_name = "Alice" if reward[int(episode%2)] > 0 else "Bob"
    all_len = len(train_data[0])+len(train_data[1])
    tmp_x3 = (x2-x1).total_seconds()/all_len
    x3 = datetime.timedelta(seconds=tmp_x3)
    #print("finished:{:<4} {:<5}(len:{:<3}) time_per_move:{},{}".format(episode + 1,win_name,all_len,x3,x2-x1))
    result_data.append(int(reward[int(episode % 2)] > 0))
    return result_data


import itertools

def multi_train(data):
    net, memory, batch_size, iteration_num, p_num = data
    optimizer =  optim.Adam(net.parameters(), weight_decay=0.01)
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
        optimizer.zero_grad()
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
        loss[0].backward()
        all_loss += float(loss[0].item())
        MSE += float(loss[1].item())
        CEE += float(loss[2].item())
        optimizer.step()
        if i == iteration_num-1:
            values = states["values"]
            action_codes = states['detailed_action_codes']
            for j in range(5):
                for key in states:
                    if key in ["values","detailed_action_codes","target"]:
                        continue
                    print(key)
                    print("{}".format(states[key][j]))
                for key in values.keys():
                    print("{}:{}".format(key,values[key][j]))
                for key in action_codes.keys():
                    print("{}:{}".format(key, action_codes[key][j]))
                print("action_probability")
                print("pai:{}".format(pai[j]))
                print("p:{}".format(p[j][pai[j]]))
                print("state_value_evaluation")
                print("z:{}".format(z[j]))
                print("v:{}".format(v[j]))
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
    print(args)
    p_size = cpu_num
    print("use cpu num:{}".format(p_size))



    cuda_flg = args.cuda == "True"
    net = New_Dual_Net(100)
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
    optimizer = optim.Adam(net.parameters(), weight_decay=0.01)

    LOG_PATH = "log_{}_{}_{}_{}_{}_{}/".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                             t1.second)
    #writer = SummaryWriter(log_dir="./logs" + LOG_PATH)
    writer = SummaryWriter(log_dir="./logs")
    for epoch in range(epoch_num):
        print("epoch {}".format(epoch + 1))
        t3 = datetime.datetime.now()
        R = New_Dual_ReplayMemory(100000)
        p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net,cuda=cuda_flg))
        #p1 = Player(9, True, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
        p1.name = "Alice"
        #if fixed_opponent is not None:
        #    if fixed_opponent == "Aggro":
        #        p2 = Player(9, False, policy=AggroPolicy())
        #    elif fixed_opponent == "OM":
        #        p2 = Player(9, False, policy=Opponent_Modeling_MCTSPolicy())
        #    else:
        #        p2 = Player(9, False, policy=Dual_NN_GreedyPolicy(origin_model=prev_net))
        #else:
        p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net,cuda=cuda_flg))
        #p2 = Player(9, False, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
        p2.name = "Bob"

        #import cProfile
        #cProfile.run("memories = multi(episode_len,p1,p2)",sort="tottime")
        #assert False
        #memories = multi(episode_len,p1,p2)
        iter_data = [(i, p1, p2) for i in range(episode_len)]
        pool = Pool(p_size)  # 最大プロセス数:8
        #memory = pool.map(preparation, iter_data)
        memory = pool.imap(preparation, iter_data)
        memory = list(tqdm(memory,total=episode_len))
        pool.close()  # add this.
        pool.terminate()  # add this.
        #[[state,state,...,reward],[],[],[],...]
        win_num = sum([cell.pop(-1) for cell in memory])

        memories = list(itertools.chain.from_iterable(memory))
        follower_attack_num = 0
        all_able_to_follower_attack = 0
        for data in memories:
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
            #if data[4] > 0 and data[1] >= 10 and data[1] <= 34:
            #    R.push(before_state, data[1], after_state, data[3], data[4])
            #    R.push(before_state, data[1], after_state, data[3], data[4])
            #    R.push(before_state, data[1], after_state, data[3], data[4])


        print("win_rate:{:.3%}".format(win_num/episode_len))
        print("follower_attack_ratio:{:.3%}".format(follower_attack_num/all_able_to_follower_attack))
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
            net.share_memory()
            all_data = R.sample(batch_size,all=True)
            iter_data = [[net,all_data,batch,iteration//p_size,i]
                         for i in range(p_size)]
            pool = Pool(p_size,initializer=tqdm.set_lock, initargs=(RLock(),))  # 最大プロセス数:8
            loss_data = pool.map(multi_train, iter_data)
            print("\n" * p_size)
            #imap = pool.imap(multi_train, iter_data)
            #loss_data = list(tqdm(imap, total=p_size))
            #[(1,1,1),(),()]
            sum_of_loss = sum(map(lambda data: data[0], loss_data))
            sum_of_MSE = sum(map(lambda data: data[1], loss_data))
            sum_of_CEE = sum(map(lambda data: data[2], loss_data))

            pool.close()  # add this.
            pool.terminate()  # add this.
            #print("AVE | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
            #      .format(sum_of_loss / p_size, sum_of_MSE / p_size, sum_of_CEE / p_size))

        else:

            for i in tqdm(range(iteration)):
            #for i in range(iteration):
                #print("\ni:{}\n".format(i))

                states, actions, rewards = R.sample(batch)
                #optimizer.zero_grad()
                states['target'] = {'actions': actions, 'rewards': rewards}
                p, v, loss = net(states, target=True)
                z = rewards
                pai = actions  # 45種類の抽象化した行動
                optimizer.zero_grad()
                loss[0].backward()
                sum_of_loss += float(loss[0].item())
                sum_of_MSE += float(loss[1].item())
                sum_of_CEE += float(loss[2].item())
                optimizer.step()
        writer.add_scalar(LOG_PATH+"Over_All_Loss", sum_of_loss / iteration, epoch)
        writer.add_scalar(LOG_PATH+"MSE", sum_of_MSE / iteration, epoch)
        writer.add_scalar(LOG_PATH+"CEE", sum_of_CEE / iteration, epoch)
        print("AVE | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
              .format(sum_of_loss / iteration, sum_of_MSE / iteration, sum_of_CEE / iteration))


        t4 = datetime.datetime.now()
        print(t4-t3)
        if epoch_num > 4 and (epoch+1) % epoch_interval == 0 and epoch+1 < epoch_num:
            PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_{:.0%}.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                 t1.second, (epoch+1)/epoch_num)
            if torch.cuda.is_available() and cuda_flg:
                PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_{:.0%}_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                        t1.second, (epoch + 1) / epoch_num)
            torch.save(net.state_dict(), PATH)
            print("{} is saved.".format(PATH))

    writer.close()
    print('Finished Training')

    PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_all.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                         t1.second)
    if torch.cuda.is_available() and cuda_flg:
        PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_all_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                             t1.second)
    torch.save(net.state_dict(), PATH)
    print("{} is saved.".format(PATH))
    t2 = datetime.datetime.now()
    print(t2)
    print(t2-t1)


if __name__ == "__main__":
    run_main()


