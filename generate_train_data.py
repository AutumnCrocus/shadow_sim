
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from emulator_test import * #importの依存関係により必ず最初にimport
from Field_setting import *
from Player_setting import *

from Game_setting import Game
deck_id_2_name = {0: "Sword_Aggro", 1: "Rune_Earth", 2: "Sword", 3: "Shadow", 4: "Dragon_PDK", 5: "Haven",
                  6: "Blood", 7: "Dragon", 8: "Forest", 9: "Rune", 10: "DS_Rune", -1: "Forest_Basic", -2: "Sword_Basic",
                  -3: "Rune_Basic",
                  -4: "Dragon_Basic", -5: "FOREST_Basic", -6: "Blood_Basic", -7: "Haven_Basic", -8: "Portal_Basic",
                  100: "Test",
                  -9: "Spell-Rune", 11: "PtP-Forest", 12: "Mid-Shadow", 13: "Neutral-Blood"}

key_2_tsv_name = {0: ["Sword_Aggro.tsv", "SWORD"], 1: ["Rune_Earth.tsv", "RUNE"], 2: ["Sword.tsv", "SWORD"],
                  3: ["New-Shadow.tsv", "SHADOW"], 4: ["Dragon_PDK.tsv", "DRAGON"], 5: ["Test-Haven.tsv", "HAVEN"],
                  6: ["Blood.tsv", "BLOOD"], 7: ["Dragon.tsv", "DRAGON"], 8: ["Forest.tsv", "FOREST"],
                  9: ["SpellBoost-Rune.tsv", "RUNE"], 10: ["Dimension_Shift_Rune.tsv", "RUNE"],
                  11: ["PtP_Forest.tsv", "FOREST"], 12: ["Mid_Shadow.tsv", "SHADOW"],
                  13: ["Neutral_Blood.tsv", "BLOOD"]}

f = Field(5)
p1 = Player(9,True)
p1.name = "Alice"
d1 = tsv_to_deck('Sword_Aggro.tsv')
d1.set_leader_class(key_2_tsv_name[0][1])
p2 = Player(9,False)
p2.name = "Bob"
d2 = tsv_to_deck('Sword_Aggro.tsv')
d2.set_leader_class(key_2_tsv_name[0][1])
d1.shuffle()
d2.shuffle()
p1.deck = d1
p2.deck = d2
f.players = [p1,p2]
p1.field = f
p2.field = f
G = Game()
#G.start(f,virtual_flg=True)
#print("Game end")
#f.show_field()
def get_data(f):
    input_field_data = []
    for i in range(2):
        for card in f.card_location[i]:
            if card.card_category == "Creature":
                input_field_data.extend([card.card_id,card.power,card.get_current_toughness()])
            else:
                input_field_data.extend(["Amulet",0,0])

        for k in range(len(f.card_location[i]),5):
            input_field_data.extend([0,0,0])
    input_field_data.extend([f.players[0].life, f.players[1].life])
    return input_field_data

train_datas = G.start_for_train_data(f,virtual_flg=True)
print("train_datas")
for i in range(2):
    print("Player{}".format(i+1))
    for data in train_datas[i]:
        print(data)
    print("")
#print("input_field_data")
#print(input_field_data[:15])
#print(input_field_data[15:30])
#print(input_field_data[30:])