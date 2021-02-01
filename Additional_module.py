# +
import torch
import torch.nn as nn
from preprocess import *
import torch.nn.functional as F
class Dual_ResNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(Dual_ResNet, self).__init__()
        n_mid = (n_in+n_out)//2
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_out)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.activate = Mish()
        self.bn1 = nn.BatchNorm1d(n_mid)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.n_in = n_in
        self.n_out = n_out
        
        self.fc3 = nn.Linear(n_in, n_out)
        #self.mish = Mish()

    def forward(self, x):
#         h1 = self.bn1(self.activate(self.fc1(x)))
#         h2 = self.bn2(self.activate(self.fc2(h1) + x))
        h1 = self.activate(self.bn1(self.fc1(x)))
        if self.n_in == self.n_out:
            h2 = self.activate(self.bn2(self.fc2(h1)) + x)
        else:
            h2 = self.activate(self.bn2(self.fc2(h1)) + self.fc3(x))
        return h2
    
    
class filtered_softmax(nn.Module):
    def __init__(self):
        super(filtered_softmax, self).__init__()

    def forward(self, x, label):
        x = torch.softmax(x,dim=1)
        x = x*label
        return x

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        tmp_x = x * torch.tanh(F.softplus(x))
        return tmp_x
    
    
class Simple_State_Net(nn.Module):
    def __init__(self, n_mid,rand=False):
        super(Simple_State_Net, self).__init__()
        self.short_mid = n_mid
        self.life_layer = nn.Linear(5, 20)
        nn.init.kaiming_normal_(self.life_layer.weight)
        if rand:
            self.emb1 = nn.Embedding(2797, len(d2v_model.docvecs[0]), padding_idx=0)
            nn.init.kaiming_normal_(self.emb1.weight)
        else:
            self.emb1 = nn.Embedding(2797,len(d2v_model.docvecs[0]),padding_idx=0)
            self.emb1.weight = nn.Parameter(d2v_ini_weight)


        self.concat_layer = nn.Linear(self.short_mid,self.short_mid)
        nn.init.kaiming_normal_(self.concat_layer.weight)
        self.class_eye = torch.cat([torch.Tensor([[0] * 8]), torch.eye(8)], dim=0)

        self.ability_eye = torch.cat([torch.Tensor([[0] * 15]), torch.eye(15)], dim=0)
        self.deck_type_eye = torch.cat([torch.Tensor([[0] * 4]), torch.eye(4)], dim=0)
        self.prelu_layer = torch.relu#Mish()#torch.tanh
        hidden_layer_num = 3
        origin = 105*20#94*self.short_mid
        node_shrink_range = (origin - n_mid) // hidden_layer_num
        self.modify_layer_num = hidden_layer_num
        node_size_list = [origin - i * node_shrink_range for i in range(hidden_layer_num)] + [n_mid]
        modify_layer = [Dual_ResNet(node_size_list[i], node_size_list[i+1]) for i in range(hidden_layer_num)]
        #[nn.Linear(node_size_list[i], node_size_list[i+1]) for i in range(hidden_layer_num)]
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
        able_to_attack = values["able_to_attack"].view(-1,10,1)
        able_to_creature_attack = values["able_to_creature_attack"].view(-1,10,1)
        able_to_evo = states["able_to_evo"].view(-1,10,1)
        class_values = self.class_eye[class_datas].view(-1, 16).unsqueeze(-1)#.to(stats.device)
        class_values = class_values.expand(-1, 16, 20)#.expand(-1, 16, self.n_mid)
        deck_type_values = self.deck_type_eye[deck_type_datas].view(-1, 8).unsqueeze(-1)#.to(stats.device)
        deck_type_values = deck_type_values.expand(-1, 8, 20)#.expand(-1, 8, self.n_mid)
        x1 = self.ability_eye[follower_abilities]
        abilities = torch.sum(x1,dim=2)#(-1,10,10)
        src1 = self.emb1(follower_card_ids)#(-1,10,20)
        follower_cards = src1
        dummy_tensor = torch.zeros([stats.size()[0],10,1]).to(stats.device)
        follower_tensor = [stats, abilities,follower_cards,dummy_tensor]#,able_to_attack,able_to_creature_attack,able_to_evo]
        x2 = torch.cat(follower_tensor,dim=2).view(-1,21,20)
        follower_values = x2
        
        src2 = self.emb1(amulet_card_ids)#(-1,10,20)
        amulet_values = src2
        life_values = self.life_layer(life_datas).unsqueeze(1)
        src3 = self.emb1(hand_ids)#(-1,10,20)
        hand_card_values = src3

        src4 = self.emb1(deck_datas)#(-1,10,20)
        deck_card_values = src4#_deck_card_values
        input_tensor = [follower_values,amulet_values,life_values,\
                       class_values,deck_type_values,hand_card_values,deck_card_values]
        #(-1,105,20)
        #print([cell.size() for cell in input_tensor])
        before_x = torch.cat(input_tensor,dim=1)
        x = before_x.view(-1,2100)
        #x = self.prelu_layer(self.concat_layer(before_x)).view(-1,94*self.short_mid)#+before_x).view(-1,94*self.short_mid)

        for i in range(self.modify_layer_num):
            x = self.prelu_layer(self.modify_layer[i](x))


        return x
# -


