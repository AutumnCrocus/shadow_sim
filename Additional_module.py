# +
import torch
import torch.nn as nn
from preprocess import *
import torch.nn.functional as F
from tqdm import tqdm
from torch.multiprocessing import Pool, Process, set_start_method,cpu_count, RLock,freeze_support, Value, Array, Manager,cpu_count
import sys
from collections import namedtuple
import copy
import random
from my_enum import *
import torch.optim as optim
from Embedd_Network_model import *
class filtered_softmax(nn.Module):
    def __init__(self):
        super(filtered_softmax, self).__init__()

    def forward(self, x, label):
        x = torch.softmax(x, dim=1)
        x = x*label
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        tmp_x = x * torch.tanh(F.softplus(x))
        return tmp_x


class Dual_ResNet(nn.Module):
    def __init__(self, n_in, n_out, activate=Mish()):
        super(Dual_ResNet, self).__init__()
        n_mid = (n_in + n_out)//2
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_out)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.activate = activate
        self.bn1 = nn.BatchNorm1d(n_mid)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.n_in = n_in
        self.n_out = n_out
        self.fc3 = nn.Linear(n_in, n_out)
        # self.mish = Mish()

    def forward(self, x):
        # h1 = self.bn1(self.activate(self.fc1(x)))
        # h2 = self.bn2(self.activate(self.fc2(h1) + x))
        h1 = self.activate(self.bn1(self.fc1(x)))
        if self.n_in == self.n_out:
            h2 = self.activate(self.bn2(self.fc2(h1)) + x)
        else:
            h2 = self.activate(self.bn2(self.fc2(h1)) + self.fc3(x))
        return h2


class Simple_State_Net(nn.Module):
    def __init__(self, n_mid, rand=False, hidden_n=6):
        super(Simple_State_Net, self).__init__()
        self.short_mid = n_mid
        self.vec_size = len(d2v_model.docvecs[0])
        self.life_layer = nn.Linear(5, self.vec_size)
        nn.init.kaiming_normal_(self.life_layer.weight)
        if rand:
            self.emb1 = nn.Embedding(2797, self.vec_size, padding_idx=0)
            nn.init.kaiming_normal_(self.emb1.weight)
        else:
            self.emb1 = nn.Embedding(2797, self.vec_size, padding_idx=0)
            self.emb1.weight = nn.Parameter(d2v_ini_weight)

        self.concat_layer = nn.Linear(self.short_mid, self.short_mid)
        nn.init.kaiming_normal_(self.concat_layer.weight)

        self.transform_size = 10 * (6 + 15 + 1 + self.vec_size)
        self.transform = nn.Linear(self.transform_size, 21*self.vec_size)
        self.class_eye = torch.cat([torch.Tensor([[0] * 8]),
                                    torch.eye(8)], dim=0)

        self.ability_eye = torch.cat([torch.Tensor([[0] * 15]), torch.eye(15)],
                                     dim=0)
        self.deck_type_eye = torch.cat([torch.Tensor([[0] * 4]), torch.eye(4)],
                                       dim=0)
        self.prelu_layer = torch.relu
        # Mish()#torch.tanh
        hidden_layer_num = hidden_n
        # 6 or 3
        origin = 105*self.vec_size
        # 94*self.short_mid
        self.integrate_layer_size = origin
        node_shrink_range = (origin - n_mid) // hidden_layer_num
        self.modify_layer_num = hidden_layer_num
        node_size_list = [origin - i * node_shrink_range
                          for i in range(hidden_layer_num)] + [n_mid]
        modify_layer = [Dual_ResNet(node_size_list[i], node_size_list[i + 1],
                                    activate=nn.PReLU(init=0.01))
                        for i in range(hidden_layer_num)]
        # [nn.Linear(node_size_list[i], node_size_list[i+1])\
        # for i in range(hidden_layer_num)]
        self.modify_layer = nn.ModuleList(modify_layer)
        self.n_mid = n_mid

    def cuda_all(self):
        self.class_eye = self.class_eye.cuda()
        self.ability_eye = self.ability_eye.cuda()
        self.deck_type_eye = self.deck_type_eye.cuda()
        return super(Simple_State_Net, self).cuda()

    def cpu(self):
        self.class_eye = self.class_eye.cpu()
        self.ability_eye = self.ability_eye.cpu()
        self.deck_type_eye = self.deck_type_eye.cpu()
        return super(Simple_State_Net, self).cpu()

    def init_weights(self):
        initrange = 0.1
        self.emb1.weight.data.uniform_(-initrange, initrange)

    def forward(self, states):
        values = states['values']
        hand_ids = states['hand_ids']
        follower_card_ids = states['follower_card_ids']
        amulet_card_ids = states['amulet_card_ids']
        follower_abilities = states['follower_abilities']
        life_datas = values['life_datas']
        class_datas = values['class_datas']
        deck_type_datas = values['deck_type_datas']
        stats = values['follower_stats']
        deck_datas = states["deck_datas"]
        able_to_attack = values["able_to_attack"].view(-1, 10, 1)
        able_to_creature_attack = values["able_to_creature_attack"].view(-1,
                                                                         10, 1)
        able_to_evo = states["able_to_evo"].view(-1, 10, 1)
        class_values = self.class_eye[class_datas].view(-1, 16).unsqueeze(-1)
        # .to(stats.device)
        class_values = class_values.expand(-1, 16, self.vec_size)
        # .expand(-1, 16, self.n_mid)
        deck_type_values = self.deck_type_eye[deck_type_datas].view(-1,
                                                                    8).unsqueeze(-1)
        # .to(stats.device)
        deck_type_values = deck_type_values.expand(-1, 8, self.vec_size)
        # .expand(-1, 8, self.n_mid)
        x1 = self.ability_eye[follower_abilities]
        abilities = torch.sum(x1, dim=2)
        # (-1,10,10)
        src1 = self.emb1(follower_card_ids)
        # (-1,10,20)
        follower_cards = src1
        dummy_tensor = torch.zeros([stats.size()[0], 10, 1]).to(stats.device)
        follower_tensor = [stats, abilities, follower_cards, dummy_tensor]
        # ,able_to_attack,able_to_creature_attack,able_to_evo]
        # [print(cell.size()) for cell in follower_tensor]
        x2 = torch.cat(follower_tensor, dim=2).view(-1, self.transform_size)
        # print(x2.size(),self.modify_layer)
        x2 = self.transform(x2)
        x2 = x2.view(-1, 21, self.vec_size)
        # 1220=10*122
        # (6+15+100+1)
        follower_values = x2
        src2 = self.emb1(amulet_card_ids)
        # (-1,10,20)
        amulet_values = src2
        life_values = self.life_layer(life_datas).unsqueeze(1)
        src3 = self.emb1(hand_ids)
        # (-1,10,20)
        hand_card_values = src3

        src4 = self.emb1(deck_datas)
        # (-1,10,20)
        deck_card_values = src4
        # _deck_card_values
        input_tensor = [follower_values, amulet_values, life_values,
                        class_values, deck_type_values, hand_card_values,
                        deck_card_values]
        # (-1,105,20)
        # print([cell.size() for cell in input_tensor])
        before_x = torch.cat(input_tensor, dim=1)
        x = before_x.view(-1, self.integrate_layer_size)
        # x = self.prelu_layer(self.concat_layer(before_x))
        # .view(-1,94*self.short_mid)#+before_x).view(-1,94*self.short_mid)

        for i in range(self.modify_layer_num):
            x = self.prelu_layer(self.modify_layer[i](x))

        return x


# -
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self._patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verobse:
                    print('early stopping')
                return True
        else:
            self.step = 0
            self.loss = loss

        return False




class Action_Value_Net(nn.Module):
    def __init__(self,parent_net,mid_size = 100):
        super(Action_Value_Net, self).__init__()
        self.n_mid = mid_size
        self.short_mid = mid_size
        #self.emb1 = nn.Embedding(5, mid_size)  # 行動のカテゴリー
        #nn.init.kaiming_normal_(self.emb1.weight)
        self.emb1 = parent_net.emb1
        #self.emb2 = parent_net.emb1#nn.Embedding(3000, mid_size, padding_idx=0)  # 1000枚*3カテゴリー（空白含む）
        #nn.init.kaiming_normal_(self.emb2.weight)
        #self.emb3 = nn.Embedding(1000, mid_size, padding_idx=0)  # フォロワー1000枚
        #self.lin1 = nn.Linear(5*mid_size+4, mid_size)#nn.Linear(7 * mid_size, mid_size)
        self.lin1 = nn.Linear(2*mid_size+parent_net.vec_size+4, mid_size)#nn.Linear(3*mid_size+4, mid_size)
        # #nn.Linear(7 * mid_size, mid_size)

        nn.init.kaiming_normal_(self.lin1.weight)
        #self.lin1 = nn.Linear(5 * mid_size, mid_size)
        self.lin2 = nn.Linear(mid_size, 1)
        nn.init.kaiming_normal_(self.lin2.weight)
        #self.lin3 = nn.Linear(36,mid_size)
        #self.lin3 = nn.Linear(66, mid_size)
        #nn.init.kaiming_normal_(self.lin3.weight)
        self.lin4_len=3
        layer = [Dual_ResNet(45*mid_size, 45*mid_size) for _ in range(self.lin4_len)]
        #[Dual_ResNet(mid_size, mid_size) for _ in range(3)]
        self.lin4 = nn.ModuleList(layer)
        #self.mish = Mish()
        self.action_catgory_eye = torch.cat([torch.Tensor([[0] * 4]), torch.eye(4)], dim=0)
        self.side_emb = nn.Embedding(3,1,padding_idx=2)
        
        self.association_layer = nn.Linear(parent_net.vec_size+5,mid_size)#nn.Linear(10+1,mid_size)#nn.Linear(mid_size+1,mid_size)
        nn.init.kaiming_normal_(self.association_layer.weight)
        self.prelu_1 = nn.PReLU(init=0.01)
        self.prelu_2 = nn.PReLU(init=0.01)

        self.prelu_3 = nn.PReLU(init=0.01)
        self.prelu_4 = nn.PReLU(init=0.01)

    def cuda_all(self):
        self.action_catgory_eye = self.action_catgory_eye.cuda()
        return super(Action_Value_Net, self).cuda()

    def cpu(self):
        self.action_catgory_eye = self.action_catgory_eye.cpu()
        return super(Action_Value_Net, self).cpu()


    def forward(self, states, detailed_action_codes,values,target=False):
        life_datas = values['life_datas']
        pp_datas = values['pp_datas']
        hand_card_costs = values['hand_card_costs']
        stats = values['follower_stats'].view(-1,6*10)
        # action_categories = detailed_action_codes['action_categories']
        # play_card_ids = detailed_action_codes['play_card_ids']
        # field_card_ids = detailed_action_codes['field_card_ids']
        action_categories = detailed_action_codes['action_categories']
        acting_card_ids = detailed_action_codes['acting_card_ids']
        acted_card_ids = detailed_action_codes['acted_card_ids']
        acted_card_sides = detailed_action_codes['acted_card_sides']
        label = detailed_action_codes['able_to_choice']
        #action_choice_len = detailed_action_codes['action_choice_len']

        embed_action_categories = self.action_catgory_eye[action_categories]#.to(stats.device)#self.emb1(action_categories)(-1,45,4)

        embed_acting_card_ids = self.emb1(acting_card_ids)
        #embed_acting_card_ids = acting_card_ids#self.emb2(acting_card_ids)
        embed_acting_card_ids = self.prelu_3(embed_acting_card_ids)

        #embed_acted_card_ids = acted_card_ids#self.emb2(acted_card_ids)#(-1,45,n_mid,?)
        embed_acted_card_ids = self.emb1(acted_card_ids)
        #print(embed_acted_card_ids.size())#(-1,45,20)
        #print("emb_acted:{}".format(embed_acted_card_ids.size()))
        embed_acted_card_sides = self.side_emb(acted_card_sides)  # (-1,45,?,n_mid)
        #print(embed_acted_card_sides.size())
        embed_acted_card_sides = embed_acted_card_sides.view(-1,45,5)
        #print(acted_card_ids,acted_card_sides,embed_acted_card_ids.size(),embed_acted_card_sides.size())
        embed_acted_card_ids = torch.cat([embed_acted_card_ids,embed_acted_card_sides],dim=2)
        embed_acted_card_ids = torch.sigmoid(self.association_layer(embed_acted_card_ids))
        # embed_acted_card_ids = torch.sum(embed_acted_card_ids,dim=2)
        embed_acted_card_ids = embed_acted_card_ids.view(-1,45,self.n_mid)
        #self.emb3(field_card_ids).view(-1,45,3*self.mid_size)
        embed_acted_card_ids = self.prelu_4(embed_acted_card_ids)

        new_states = states#.unsqueeze(1)
        new_states = torch.stack([new_states]*45,dim=1)
        input_tensors = [new_states, embed_action_categories, embed_acting_card_ids, embed_acted_card_ids]
        #print("{}".format([cell.size() for cell in input_tensors]))
        # tmp = torch.cat([new_states,embed_action_categories,embed_play_card_ids,embed_field_card_ids], dim=2)
        tmp = torch.cat(input_tensors, dim=2)
        #print(tmp.size(),label.size())
        label_tensor=torch.stack([label]*tmp.size()[-1],dim=2)
        tmp = tmp * label_tensor
        #(-1,45,2*mid_size+20+4)
        output1 = self.prelu_1(self.lin1(tmp)).view(-1,45*self.n_mid)#(-1,45,n_mid)→(-1,45*n_mid)
        for i in range(len(self.lin4)):
            output1 = self.prelu_3(self.lin4[i](output1))
            
        output1 = output1.view(-1,45,self.n_mid)
        output2 = self.prelu_2(self.lin2(output1)).view(-1,45)
        output = output2 * label#(-1,45)[0,1]

        return output

class Dual_Loss(nn.Module):

    def __init__(self):
        super(Dual_Loss, self).__init__()

    def forward(self, p, v, z, pai,action_choice_len):

        #tmp_MSE = torch.sum(
        #    torch.pow((z - v),2),
        #   dim=1)
        LOSS_EPSILON=1.0e-5
        tmp_MSE = torch.sum(
            -(z+1)*torch.log((v+1)/2+LOSS_EPSILON)/2+(z-1)*torch.log((1-v)/2+LOSS_EPSILON)/2,
           dim=1)
        
        

        MSE = torch.mean(tmp_MSE)
        #print("MSE:{}".format(MSE))
        #print("loss:",loss)
        #print("mean:",MSE)

        tmp_CEE1 = p[range(p.size()[0]),pai]+LOSS_EPSILON
        #choice_len_term = 1/torch.sqrt(action_choice_len)
        #print(choice_len_term)
        tmp_CEE2 = -torch.log(tmp_CEE1)#*choice_len_term
        CEE = torch.mean(tmp_CEE2)
        #pai = pai.t()[0]
        #CEE = self.cross_entropy(p,pai)#softmaxも含まれている
        #loss = CEE
        loss = MSE + CEE
        #L2正則化はoptimizer

        return loss, MSE, CEE

class New_Dual_ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = []
        self.index = 0
        #self.sub_dict = [None,None,None,None,None,None]

    def push(self, state, action, before_state, detailed_action_code, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリが満タンじゃないときには追加
        #self.memory[self.index] = Dual_State_value(state, action, before_state, detailed_action_code, reward)
        self.memory[self.index] = {'state':state, 'action':action, 'before_state':before_state,
                                   'detailed_action_code':detailed_action_code, 'reward':reward}
        self.index = (self.index + 1) % self.capacity

    def sep_sample(self,data):
        tmp,p_num = data
        cuda = self.cuda
        states = [cell['state'] for cell in tmp]  # [cell.state for cell in tmp]
        states = Detailed_State_data_2_Tensor(states, cuda=cuda, normalize=True)
        before_states = [list(cell['before_state']) for cell in tmp]
        # [list(cell.before_state) for cell in tmp]
        # print(before_states)

        tensor_action_categories = torch.LongTensor(
            [cell[0] for cell in before_states])
        tensor_acting_card_ids_in_action = torch.LongTensor(
            [names.index(cell[1])
             if cell[1] in names else 0 for cell in
             before_states])
        tensor_acted_card_ids_in_action = torch.LongTensor(
            [names.index(cell[2])
             if cell[2] in names else 0 for cell in
             before_states])
        tensor_acted_card_sides_in_action = torch.LongTensor(
            [cell[3] for cell in before_states])
        actions = [cell['action'] for cell in tmp]  # [cell.action for cell in tmp]
        actions = torch.LongTensor(actions)  # torch.stack(actions, dim=0)
        rewards = [[cell['reward']] for cell in tmp]  # [[cell.reward] for cell in tmp]
        rewards = torch.FloatTensor(rewards)
        before_states = [tensor_action_categories, tensor_acting_card_ids_in_action,
                         tensor_acted_card_ids_in_action, tensor_acted_card_sides_in_action]
        if cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            before_states = [cell.cuda() for cell in before_states]
            torch.cuda.empty_cache()
        detailed_action_codes = [cell['detailed_action_code'] for cell in
                                 tmp]  # [cell.detailed_action_code for cell in tmp]
        detailed_action_codes = Detailed_action_code_2_Tensor(detailed_action_codes, cuda=cuda)
        states['detailed_action_codes'] = detailed_action_codes
        states['before_states'] = before_states
        #self.sub_dict[p_num] = [states,actions,rewards]
        
        return states, actions, rewards

    
    def integrate_dict(self,key):
        if key in ['values','detailed_action_codes']:
            return key,None
        dict_list = self.dict_list
        #states = self.states
        if key in self.value_key_list:
            dict_data = torch.cat([cell['values'][key] for cell in dict_list],dim=0)
        elif key in self.action_key_list:
            dict_data = torch.cat([cell['detailed_action_codes'][key] for cell in dict_list],dim=0)
        elif key == 'before_states':
            dict_data = [torch.cat([cell[key][i] for cell in dict_list]) for i in range(4)]
        else:
            dict_data = torch.cat([cell[key] for cell in dict_list],dim=0)
        return key,dict_data
            
    def sample(self, batch_size,all=False,cuda=False,multi=0):
        if all:
            #tmp = self.memory
            tmp = random.sample(self.memory, len(self.memory))
        else:
            tmp = random.sample(self.memory, batch_size)

        if multi > 0:
            self.cuda = cuda
            data_len = len(tmp)
            process_num=min(cpu_count()-1,64)
            small_data_len = data_len //process_num
            iter_data = [(tmp[i*small_data_len:min((i+1)*small_data_len,data_len)],i) for i in range(process_num)]
            freeze_support()
            with Pool(process_num, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
                data_list = pool.map(self.sep_sample,iter_data)#(dict,tensor,tensor)
                dict_list = [cell[0] for cell in data_list]
                action_list = [cell[1] for cell in data_list]
                actions = torch.cat(action_list,dim=0)
                reward_list = [cell[2] for cell in data_list]
                rewards = torch.cat(reward_list, dim=0)
                states = {'values':{}, 'detailed_action_codes':{}}
                self.key_list = list(dict_list[0].keys())
                self.value_key_list = list(dict_list[0]['values'].keys())
                self.action_key_list = list(dict_list[0]['detailed_action_codes'].keys())
                self.all_keys_list = self.key_list + self.value_key_list + self.action_key_list
                self.dict_list = dict_list
                print("step2")
                if cuda:
                    for key in self.key_list:
                        if key == 'values' or key == 'detailed_action_codes':
                            for sub_key in list(dict_list[0][key].keys()):
                                states[key][sub_key] = torch.cat([cell[key][sub_key] for cell in dict_list],dim=0)

                        elif key == 'before_states':
                            states[key] = [torch.cat([cell[key][i] for cell in dict_list]) for i in range(4)]
                        else:
                            states[key] = torch.cat([cell[key] for cell in dict_list],dim=0)
                else:
                    iter_data = [key for key in self.all_keys_list]
                    dict_data=pool.map(self.integrate_dict,iter_data)
                    #pool.terminate()
                    #pool.close()
                    for cell in dict_data:
                        key,dict_cell = cell
                        if key in ['values','detailed_action_codes']:
                            pass
                        elif key in self.value_key_list:
                            states['values'][key] = dict_cell
                        elif key in self.action_key_list:
                            states['detailed_action_codes'][key] = dict_cell
                        else:
                            states[key] = dict_cell
                del self.dict_list
                del data_list,action_list,reward_list
                torch.cuda.empty_cache()
 
        else:
            states = [cell['state'] for cell in tmp]#[cell.state for cell in tmp]
            states = Detailed_State_data_2_Tensor(states,cuda=cuda,normalize=True)
            before_states = [list(cell['before_state']) for cell in tmp]
            #[list(cell.before_state) for cell in tmp]
            #print(before_states)

            tensor_action_categories = torch.LongTensor(
                [cell[0]for cell in before_states])
            tensor_acting_card_ids_in_action = torch.LongTensor(
                [names.index(cell[1])
                  if cell[1] in names else 0 for cell in
                 before_states])
            tensor_acted_card_ids_in_action = torch.LongTensor(
                [names.index(cell[2])
                  if cell[2] in names else 0 for cell in
                 before_states])
            tensor_acted_card_sides_in_action = torch.LongTensor(
                [cell[3] for cell in before_states])
            actions = [cell['action'] for cell in tmp]#[cell.action for cell in tmp]
            actions = torch.LongTensor(actions)#torch.stack(actions, dim=0)
            rewards = [[cell['reward']] for cell in tmp]#[[cell.reward] for cell in tmp]
            rewards = torch.FloatTensor(rewards)
            before_states = [tensor_action_categories,tensor_acting_card_ids_in_action,
                             tensor_acted_card_ids_in_action,tensor_acted_card_sides_in_action]
            if cuda:
                actions = actions.cuda()
                rewards = rewards.cuda()
                before_states = [cell.cuda() for cell in before_states]
            detailed_action_codes = [cell['detailed_action_code'] for cell in tmp]#[cell.detailed_action_code for cell in tmp]
            detailed_action_codes = Detailed_action_code_2_Tensor(detailed_action_codes,cuda=cuda)
            states['detailed_action_codes'] = detailed_action_codes
            states['before_states'] = before_states


        return states, actions, rewards

    def __len__(self):
        return len(self.memory)



def Detailed_State_data_2_Tensor(datas,cuda=False, normalize=False):
    data_len = len(datas)
    #print(type(datas))
    #print(type(datas[0]))
    #hand_ids = torch.LongTensor([[0 for _ in range(9)] for _ in range(data_len)])

    hand_ids = torch.LongTensor([[datas[i]["hand_ids"][j] for j in range(9)] for i in range(data_len)])
    #hand_ids = torch.Tensor([[d2v_model.docvecs[datas[i]["hand_ids"][j]] for j in range(9)] for i in range(data_len)])
    # hand_ids = torch.LongTensor([[datas[i]["hand_ids"][j] for j in range(9)] for i in range(data_len)])
    #hand_card_costs = torch.Tensor([[0 for j in range(9)] for i in range(data_len)])
    hand_card_costs = torch.Tensor([[datas[i]["hand_card_costs"][j] for j in range(9)] for i in range(data_len)])


    follower_card_ids = torch.LongTensor(
        [[datas[i]["follower_card_ids"][j] for j in range(10)] for i in range(data_len)])
    #follower_card_ids = torch.Tensor(
    #    [[d2v_model.docvecs[datas[i]["follower_card_ids"][j]] for j in range(10)] for i in range(data_len)])
    # follower_card_ids = torch.LongTensor([[datas[i]["follower_card_ids"][j] for j in range(10)] for i in range(data_len)])
    #follower_card_ids = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])

    amulet_card_ids = torch.LongTensor(
        [[datas[i]["amulet_card_ids"][j] for j in range(10)] for i in range(data_len)])
    #amulet_card_ids = torch.Tensor([[d2v_model.docvecs[datas[i]["amulet_card_ids"][j]] for j in range(10)] for i in range(data_len)])
    # amulet_card_ids = torch.LongTensor([[datas[i]["amulet_card_ids"][j] for j in range(10)] for i in range(data_len)])
    #amulet_card_ids = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    follower_abilities = torch.LongTensor([[[datas[i]["follower_abilities"][j][k] if k < len(
        datas[i]["follower_abilities"][j]) else 0 for k in range(15)] for j in range(10)] for i in range(data_len)])
    follower_stats = torch.Tensor([[datas[i]["follower_stats"][j] for j in range(10)] for i in range(data_len)])
    #follower_stats = torch.Tensor([[(0,0) for _ in range(10)] for _ in range(data_len)])
    #able_to_evo = torch.LongTensor([[0 for _ in range(10)] for _ in range(data_len)])
    
    able_to_evo = torch.Tensor(
        [datas[i]["able_to_evo"] for i in range(data_len)])
    #able_to_evo = torch.LongTensor(
    #    [[datas[i].able_to_evo[j] if j < len(datas[i].able_to_evo) else 0 for j in range(10)] for i in range(data_len)])
    
    able_to_play = torch.Tensor(
        [datas[i]["able_to_play"] for i in
         range(data_len)])
    #able_to_play = torch.LongTensor(
    #    [[datas[i].able_to_play[j] if j < len(datas[i].able_to_play) else 0 for j in range(9)] for i in
    #     range(data_len)])
    able_to_attack = torch.Tensor(
        [datas[i]["able_to_attack"] for i in
         range(data_len)])
    able_to_creature_attack = torch.Tensor(
        [datas[i]["able_to_creature_attack"] for
         i in range(data_len)])
    #able_to_play = torch.LongTensor([[0 for _ in range(9)] for _ in range(data_len)])
    #able_to_attack = torch.LongTensor(
    #    [[1 if j < len(datas[i].able_to_attack) else 0 for j in range(5)] for i in
    #     range(data_len)])
    #able_to_creature_attack = torch.LongTensor(
    #    [[1 if j < len(datas[i].able_to_creature_attack)  else 0 for j in range(5)] for
    #     i in range(data_len)])

    pp_datas = torch.Tensor([datas[i]["pp_data"] for i in range(data_len)])
    life_datas = torch.Tensor([datas[i]["life_data"][0] for i in range(data_len)])
    class_datas = torch.LongTensor([datas[i]["life_data"][1] for i in range(data_len)])
    deck_type_datas = torch.LongTensor([datas[i]["life_data"][2] for i in range(data_len)])
    #d2v_mdoel.docvecs

    deck_datas = [[cell for cell in datas[i]["deck_data"]] for i in range(data_len)]
    #deck_datas = [[d2v_model.docvecs[cell] for cell in datas[i]["deck_data"]] for i in range(data_len)]
    deck_datas = torch.LongTensor(deck_datas)
    #deck_datas = torch.LongTensor([datas[i]["deck_data"] for i in range(data_len)])
    #if normalize:
    #   normalized_tensors =[hand_card_costs,follower_stats,pp_datas,life_datas]
    #    for tensor in normalized_tensors:
    #       mean = torch.mean(tensor)
    #        std = torch.std(tensor)
    #        tensor = (tensor-mean)/std

    ans = {'values': {'life_datas': life_datas,
                      'class_datas': class_datas,
                      'deck_type_datas':deck_type_datas,
                      'hand_card_costs': hand_card_costs,
                      'follower_stats': follower_stats,
                      'pp_datas': pp_datas,
                      'able_to_play': able_to_play,
                      'able_to_attack': able_to_attack,
                      'able_to_creature_attack': able_to_creature_attack,
                      },
           'hand_ids': hand_ids,
           'follower_card_ids': follower_card_ids,
           'amulet_card_ids': amulet_card_ids,
           'follower_abilities': follower_abilities,
           'able_to_evo': able_to_evo,
           'deck_datas':deck_datas}
    if cuda:
        for key in list(ans.keys()):
            if key == "values":
                for sub_key in list(ans["values"].keys()):
                    ans["values"][sub_key] = ans["values"][sub_key].cuda()
            else:
                ans[key] = ans[key].cuda()
        torch.cuda.empty_cache()
    
    return ans


def Detailed_action_code_2_Tensor(action_codes, cuda = False):
    action_code_len = len(action_codes)

    tensor_action_categories = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][0] for j in range(45)] for i in range(action_code_len)])

    # tensor_acting_card_ids_in_action = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][1] for j in range(45)] for i in range(action_code_len)])
    # tensor_acting_card_ids_in_action = torch.Tensor(
    #     [[d2v_model.docvecs[names.index(action_codes[i]['action_codes'][j][1])]
    #       if action_codes[i]['action_codes'][j][1] in names else [0]*20 for j in range(45)] for i in range(action_code_len)])
    tensor_acting_card_ids_in_action = torch.LongTensor(
        [[names.index(action_codes[i]['action_codes'][j][1])
          if action_codes[i]['action_codes'][j][1] in names else 0 for j in range(45)] for i in range(action_code_len)])

    # tensor_acted_card_ids_in_action = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][2] for j in range(45)] for i in range(action_code_len)])
    # tensor_acted_card_ids_in_action = torch.Tensor(
    #     [[d2v_model.docvecs[names.index(action_codes[i]['action_codes'][j][2])]
    #       if action_codes[i]['action_codes'][j][2] in names else [0]*20 for j in range(45)] for i in range(action_code_len)])
    tensor_acted_card_ids_in_action = torch.LongTensor(
        [[names.index(action_codes[i]['action_codes'][j][2])
          if action_codes[i]['action_codes'][j][2] in names else 0 for j in range(45)] for i in range(action_code_len)])
    tensor_acted_card_sides_in_action = torch.LongTensor(
        [[action_codes[i]['action_codes'][j][3] for j in range(45)] for i in range(action_code_len)])

    able_to_choice = torch.Tensor([action_codes[i]['able_to_choice'] for i in range(action_code_len)])
    action_choice_len = torch.Tensor([[int(sum(action_codes[i]['able_to_choice']))] for i in range(action_code_len)])
    action_codes_dict = {'action_categories': tensor_action_categories,
                         'acting_card_ids': tensor_acting_card_ids_in_action,
                         'acted_card_ids': tensor_acted_card_ids_in_action,
                         'acted_card_sides': tensor_acted_card_sides_in_action,
                         'able_to_choice': able_to_choice,
                         'action_choice_len':action_choice_len}
    #
    # tensor_action_categories = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][0] for j in range(45)] for i in range(action_code_len)])
    # tensor_play_card_ids_in_action = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][1] for j in range(45)] for i in range(action_code_len)])
    # tensor_field_card_ids_in_action = torch.LongTensor(
    #     [[action_codes[i]['action_codes'][j][2:5] for j in range(45)] for i in range(action_code_len)])
    # able_to_choice = torch.Tensor([action_codes[i]['able_to_choice'] for i in range(action_code_len)])
    # action_choice_len = torch.Tensor([[int(sum(action_codes[i]['able_to_choice']))] for i in range(action_code_len)])
    # action_codes_dict = {'action_categories': tensor_action_categories,
    #                      'play_card_ids': tensor_play_card_ids_in_action,
    #                      'field_card_ids': tensor_field_card_ids_in_action,
    #                      'able_to_choice': able_to_choice,
    #                      'action_choice_len':action_choice_len}
    if cuda:
        for key in list(action_codes_dict.keys()):
            action_codes_dict[key] = action_codes_dict[key].cuda()
        torch.cuda.empty_cache()

    return action_codes_dict