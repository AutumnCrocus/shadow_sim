from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock

try:
    set_start_method('spawn')
    print("spawn is run.")
    #set_start_method('fork') GPU使用時CUDA initializationでerror
    #print('fork')
except RuntimeError:
    pass

from emulator_test import *  # importの依存関係により必ず最初にimport
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

parser.add_argument('--decklists', help='使用するデッキリスト')
parser.add_argument('--opponent', help='対戦相手のAIタイプ')
parser.add_argument('--model_name', help='使用モデル')
parser.add_argument('--iteration', help='各組み合わせの対戦回数')
args = parser.parse_args()
deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                  6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic",
                  -2: "Sword_Basic",
                  -3: "Rune_Basic",
                  -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic",
                  -8: "Portal_Basic",
                  100: "Test",
                  -9: "Spell-Rune", 11: "PtP-Forest", 12: "Mid-Shadow", 13: "Neutral-Blood"}

key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                  3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"],
                  5: ["Test-Haven.tsv", "HAVEN"],
                  6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                  9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                  11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"],
                  13: ["Neutral_Blood.tsv", "BLOOD"]}
def multi_battle(episode_data):
    episode = episode_data[0]
    print(episode)
    deck_type1,deck_type2 = episode_data[3]
    deck_locations = episode_data[4]
    iteration = episode_data[-1]
    win = 0
    lose = 0
    libout = 0
    G = Game()
    first = 0
    for i in range(iteration):
        f = Field(5)
        p1 = episode_data[1].get_copy(None)
        p2 = episode_data[2].get_copy(None)
        d1 = tsv_to_deck(key_2_tsv_name[deck_type1][0])
        d1.set_leader_class(key_2_tsv_name[deck_type1][1])
        d2 = tsv_to_deck(key_2_tsv_name[deck_type2][0])
        d2.set_leader_class(key_2_tsv_name[deck_type2][1])
        d1.shuffle()
        d2.shuffle()
        p1.deck = d1
        p2.deck = d2

        f.players = [p1, p2] if i%2==0 else [p2,p1]

        p1.field = f
        p2.field = f
        f.players[0].draw(f.players[0].deck, 3)
        f.players[1].draw(f.players[1].deck, 3)
        p1.first = i%2 == 0
        p2.first = i%2 == 1
        p1.player_num = i%2
        p2.player_num = 1- i%2
        x1 = datetime.datetime.now()
        #train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=i % 2)
        w, l, lib_num, turn = G.start(f, virtual_flg= True)
        win = win + w if i%2==0 else win+l
        first += w

    result_data = (deck_locations,[win/100,first/100])

    return result_data
    episode = episode_data[0]
    #print("start:{}".format(episode + 1))

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
    train_data, reward = G.start_for_dual(f, virtual_flg=True, target_player_num=episode % 2)

def run_main():
    deck_lists = list(map(int,args.decklists.split(","))) if args.decklists is not None else None
    if deck_lists is None:
        deck_lists = list(deck_id_2_name.keys())
    else:
        assert all(key in deck_id_2_name for key in deck_lists)
    if deck_lists == [0, 1, 4, 5, 10, 12]:
        deck_lists = [0, 1, 4, 12, 5, 10]
    mylogger.info("deck_lists:{}".format(deck_lists))
    D = [Deck() for i in range(len(deck_lists))]
    deck_index = 0
    # sorted_keys = sorted(list(deck_id_2_name.keys()))
    # for i in sorted_keys:
    #    if i not in deck_lists:
    #        continue
    for i in deck_lists:
        mylogger.info("{}(deck_id:{}):{}".format(deck_index, i, key_2_tsv_name[i]))
        D[deck_index] = tsv_to_deck(key_2_tsv_name[i][0])
        D[deck_index].set_leader_class(key_2_tsv_name[i][1])
        deck_index += 1
    Results = {}
    list_range = range(len(deck_lists))
    #print(list(itertools.product(list_range,list_range)))

    Player1 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(model_name=args.model_name),
                     mulligan=Min_cost_mulligan_policy())

    if args.opponent is not None:
        if args.model_name is not None:
            if args.opponent == "Greedy":
                Player2 = Player(9, True, policy=NN_GreedyPolicy(model_name=args.model_name),
                                 mulligan=Min_cost_mulligan_policy())
            elif args.opponent == "MCTS":
                Player2 = Player(9, True, policy=New_Dual_NN_Non_Rollout_OM_ISMCTSPolicy(model_name=args.model_name),
                                 mulligan=Min_cost_mulligan_policy())
        else:
            Player2 = Player(9, True, policy=Opponent_Modeling_MCTSPolicy(),
                             mulligan=Min_cost_mulligan_policy())
    else:
        Player2 = Player(9, True, policy=AggroPolicy(),
             mulligan=Min_cost_mulligan_policy())

    Player1.name = "Alice"
    Player2.name = "Bob"
    iteration = int(args.iteration) if args.iteration is not None else 10
    deck_list_len = len(deck_lists)
    iter_data = [(deck_list_len*i+j,Player1,
                   Player2,(i,j),(deck_lists[i],deck_lists[j]),iteration) for i,j in itertools.product(list_range,list_range)]
    pool = Pool(3)  # 最大プロセス数:8
    # memory = pool.map(preparation, iter_data)
    result = pool.map(multi_battle, iter_data)
    #result = list(tqdm(result, total=len(list_range)**2))
    pool.close()  # add this.
    pool.terminate()  # add this.
    for data in result:
        Results[data[0]] = data[1]
    print(Results)
if __name__ == "__main__":
    run_main()
