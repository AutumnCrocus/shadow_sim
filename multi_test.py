from torch.multiprocessing import Pool, Process, set_start_method,cpu_count

try:
    set_start_method('spawn')
    print("spawn is run.")
except RuntimeError:
    pass
from test import *  # importの依存関係により必ず最初にimport
from Field_setting import *
from Player_setting import *
from Policy import *
from Game_setting import Game

from Embedd_Network_model import *
import copy
import datetime
# net = New_Dual_Net(100)
import os

Detailed_State_data = namedtuple('Value', ('hand_ids', 'hand_card_costs', 'follower_card_ids',
                                           'amulet_card_ids', 'follower_stats', 'follower_abilities', 'able_to_evo',
                                           'life_data', 'pp_data', 'able_to_play', 'able_to_attack',
                                           'able_to_creature_attack'))

G = Game()

def preparation(episode_data):
    episode = episode_data[0]
    #print("start:{}".format(episode + 1))
    f = Field(5)
    p1 = episode_data[1].get_copy(f)
    p2 = episode_data[2].get_copy(f)
    deck_type1 = random.choice(list(key_2_tsv_name.keys()))
    deck_type2 = random.choice(list(key_2_tsv_name.keys()))
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
                            'able_to_creature_attack': data[0].able_to_creature_attack}

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
                           'able_to_creature_attack': data[2].able_to_creature_attack}
            action_probability = data[1]
            detailed_action_code = data[3]

            result_data.append((before_state, action_probability, after_state, detailed_action_code, reward[i]))

    x2 = datetime.datetime.now()
    win_name = "Alice" if reward[int(episode%2)] > 0 else "Bob"
    print("finished:{:<4} {:<5},{}".format(episode + 1,win_name,x2-x1))
    return result_data


import itertools

"""
def multi(episode_len,p1,p2):
    p_size = int(torch.cuda.is_available())*4 + 4
    p = Pool(p_size)  # 最大プロセス数:10
    iter_data = [(i,p1,p2) for i in range(episode_len)]
    memory = p.map(preparation, iter_data)
    #memory = p.map(preparation, range(episode_len))
    memory = list(itertools.chain.from_iterable(memory))
    p.close()  # add this.
    p.terminate()  # add this.
    return memory
"""


if __name__ == "__main__":

    p_size = 2#cpu_count() - 1#int(torch.cuda.is_available()) * 4 + 4
    #if torch.cuda.is_available():
    #    p_size = 4
    print("use cpu num:{}".format(p_size))



    from test import *  # importの依存関係により必ず最初にimport
    from Field_setting import *
    from Player_setting import *
    from Policy import *
    from Game_setting import Game
    parser = argparse.ArgumentParser(description='デュアルニューラルネットワーク学習コード')

    parser.add_argument('--episode_num', help='試行回数')
    parser.add_argument('--iteration_num', help='イテレーション数')
    parser.add_argument('--epoch_num', help='エポック数')
    parser.add_argument('--batch_size', help='バッチサイズ')
    parser.add_argument('--mcts', help='サンプリングAIをMCTSにする(オリジナルの場合は[OM])')
    parser.add_argument('--deck', help='サンプリングに用いるデッキの選び方')
    parser.add_argument('--cuda', help='gpuを使用するかどうか')
    args = parser.parse_args()
    cuda_flg = args.cuda == "True"
    net = New_Dual_Net(100)
    if torch.cuda.is_available() and cuda_flg:
        net = net.cuda()
        print("cuda is available.")
    net.zero_grad()
    deck_sampling_type = False
    if args.deck is not None:
        deck_sampling_type = True

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

    for epoch in range(epoch_num):
        print("epoch {}".format(epoch + 1))
        t3 = datetime.datetime.now()
        R = New_Dual_ReplayMemory(100000)
        p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net,cuda=cuda_flg))
        #p1 = Player(9, True, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
        p1.name = "Alice"
        p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=prev_net,cuda=cuda_flg))
        #p2 = Player(9, False, policy=AggroPolicy(), mulligan=Min_cost_mulligan_policy())
        p2.name = "Bob"

        #import cProfile
        #cProfile.run("memories = multi(episode_len,p1,p2)",sort="tottime")
        #assert False
        #memories = multi(episode_len,p1,p2)
        iter_data = [(i, p1, p2) for i in range(episode_len)]
        pool = Pool(p_size)  # 最大プロセス数:8
        memory = pool.map(preparation, iter_data)
        pool.close()  # add this.
        pool.terminate()  # add this.
        #[[],[],[],[],...]
        win_num = sum([int(memory[i][0][-1] > 0) for i in range(len(memory))])
        print("win_rate:{:.3%}".format(win_num/episode_len))
        memories = list(itertools.chain.from_iterable(memory))


        if torch.cuda.is_available() and cuda_flg:
            for data in memories:
                before_state = Detailed_State_data(data[0]['hand_ids'], data[0]['hand_card_costs'],
                                data[0]['follower_card_ids'], data[0]['amulet_card_ids'],
                                data[0]['follower_stats'], data[0]['follower_abilities'],
                                data[0]['able_to_evo'], data[0]['life_data'],
                                data[0]['pp_data'], data[0]['able_to_play'],
                                data[0]['able_to_attack'], data[0]['able_to_creature_attack'])
                after_state = Detailed_State_data(data[2]['hand_ids'], data[2]['hand_card_costs'],
                                data[2]['follower_card_ids'], data[2]['amulet_card_ids'],
                                data[2]['follower_stats'], data[2]['follower_abilities'],
                                data[2]['able_to_evo'], data[2]['life_data'],
                                data[2]['pp_data'], data[2]['able_to_play'],
                                data[2]['able_to_attack'], data[2]['able_to_creature_attack'])
                R.push(before_state, torch.LongTensor([data[1]]).cuda(), after_state, data[3], torch.FloatTensor([data[4]]).cuda())
                #R.push(data[0], torch.LongTensor([data[1]]).cuda(), data[2], data[3], torch.FloatTensor([reward]).cuda())
        else:
            for data in memories:
                before_state = Detailed_State_data(data[0]['hand_ids'], data[0]['hand_card_costs'],
                                data[0]['follower_card_ids'], data[0]['amulet_card_ids'],
                                data[0]['follower_stats'], data[0]['follower_abilities'],
                                data[0]['able_to_evo'], data[0]['life_data'],
                                data[0]['pp_data'], data[0]['able_to_play'],
                                data[0]['able_to_attack'], data[0]['able_to_creature_attack'])
                after_state = Detailed_State_data(data[2]['hand_ids'], data[2]['hand_card_costs'],
                                data[2]['follower_card_ids'], data[2]['amulet_card_ids'],
                                data[2]['follower_stats'], data[2]['follower_abilities'],
                                data[2]['able_to_evo'], data[2]['life_data'],
                                data[2]['pp_data'], data[2]['able_to_play'],
                                data[2]['able_to_attack'], data[2]['able_to_creature_attack'])
                R.push(before_state, torch.LongTensor([data[1]]), after_state, data[3], torch.FloatTensor([data[4]]))

        print("sample_size:{}".format(len(R.memory)))
        prev_net = copy.deepcopy(net)
        optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
        sum_of_loss = 0
        sum_of_MSE = 0
        sum_of_CEE = 0
        p, pai, z, states = None, None, None, None
        for i in tqdm(range(iteration)):
            states, actions, rewards = R.sample(batch_size)
            optimizer.zero_grad()

            states['target'] = {'actions': actions, 'rewards': rewards}
            p, v, loss = net(states, target=True)
            z = rewards
            pai = actions  # 45種類の抽象化した行動
            #loss.backward()
            loss[0].backward()
            sum_of_loss += float(loss[0].item())
            sum_of_MSE += float(loss[1].item())
            sum_of_CEE += float(loss[2].item())
            optimizer.step()
        print("AVE | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}"\
              .format(sum_of_loss/iteration,sum_of_MSE/iteration,sum_of_CEE/iteration))
        t4 = datetime.datetime.now()
        print(t4-t3)
        if epoch_num > 4 and (epoch+1) % (epoch_num//4) == 0 and epoch+1 < epoch_num:
            PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_{:.0%}.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                 t1.second, (epoch+1)/epoch_num)
            if torch.cuda.is_available() and cuda_flg:
                PATH = "model/Multi_Dual_{}_{}_{}_{}_{}_{}_{:.0%}_cuda.pth".format(t1.year, t1.month, t1.day, t1.hour, t1.minute,
                                                                        t1.second, (epoch + 1) / epoch_num)
            torch.save(net.state_dict(), PATH)
            print("{} is saved.".format(PATH))


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


