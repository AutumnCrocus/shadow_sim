from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock
from Embedd_Network_model import *
from multi_test import *
import torch
import argparse
print(args)
net = New_Dual_Net(10)
model_name = 'Multi_Dual_2020_4_23_0_58_12_47%_10nodes.pth'
PATH = 'model/'+ model_name
net.load_state_dict(torch.load(PATH))
p_size = 3
t3 = datetime.datetime.now()
R = New_Dual_ReplayMemory(100000)
p1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net))
p1.name = "Alice"
p2 = Player(9, False, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(origin_model=net))
p2.name = "Bob"
win_num = 0
episode_len = 10
for episode in tqdm(range(episode_len)):
    f = Field(5)
    deck_type1 = deck_flg
    deck_type2 = deck_flg
    if deck_flg is None:
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
    # import cProfile
    # cProfile.run("G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)",sort="tottime")
    # assert False
    train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)
    f.players[0].life = 20
    f.players[0].hand.clear()
    f.players[0].deck = None
    f.players[0].lib_out_flg = False
    f.players[1].life = 20
    f.players[1].hand.clear()
    f.players[1].deck = None
    f.players[1].lib_out_flg = False
    for i in range(2):
        for data in train_data[i]:
            R.push(data[0], data[1], data[2], data[3], reward[i])
    win_num += int(reward[episode % 2] > 0)

print("sample_size:{}".format(len(R.memory)))
print("win_rate:{:.2%}".format(win_num / episode_len))

sum_of_loss = 0
sum_of_MSE = 0
sum_of_CEE = 0
p, pai, z, states, loss = None, None, None, None, None
current_net, prev_optimizer = None, None
states, actions, rewards = R.sample(100,all=True)

states['target'] = {'actions': actions, 'rewards': rewards}
net.eval()
p, v, loss = net(states, target=True)
z = rewards
pai = actions  # 45種類の抽象化した行動
sum_of_loss += float(loss[0].item())
sum_of_MSE += float(loss[1].item())
sum_of_CEE += float(loss[2].item())

print("AVE | Over_All_Loss: {:.3f} | MSE: {:.3f} | CEE:{:.3f}" \
      .format(sum_of_loss, sum_of_MSE, sum_of_CEE))

"""
for key in list(net.state_dict().keys()):
    print(key,net.state_dict()[key].size())
    if "num_batches_tracked" in key:
        continue
    if len(net.state_dict()[key].size()) == 1:
        print(torch.max(net.state_dict()[key],dim=0).indices, "\n", torch.min(net.state_dict()[key],dim=0).indices)
    else:
        print(net.state_dict()[key].size())
        print(torch.max(net.state_dict()[key],1).indices,"\n",torch.min(net.state_dict()[key],1).indices)
    print("")
"""