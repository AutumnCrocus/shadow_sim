# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import copy 
import numpy as np
import random
import math
from creature_ability_list import creature_ability_dict
from spell_ability_list import spell_ability_dict
from amulet_ability_list import amulet_ability_dict
from cost_change_ability_list import cost_change_ability_dict
from battle_ability_list import battle_ability_dict
from trigger_ability_list import trigger_ability_dict
from my_moduler import get_module_logger
mylogger = get_module_logger(__name__)
from my_enum import *
import csv
import pandas as pd

def tsv_to_card_list(tsv_name):
    card_list={}
    card_category=list(tsv_name.split("_"))[1]
    with open("Card_List_TSV/"+tsv_name) as f:
        reader = csv.reader(f,delimiter='\t',lineterminator='\n')
        for row in reader:
            card_id=int(row[0])
            #card_cost=int(row[1])
            card_cost=int(row[2])
            #assert card_category in ["Creature","Spell","Amulet"]
            if card_id not in card_list:card_list[card_id]=[]
            
            card_name=row[1]

            
            
            card_class=None
            card_list[card_id].append(card_cost)
            card_traits=None
            has_count=None
            if card_category=="Creature":
                card_class=LeaderClass[row[-2]].value
                card_traits=Trait[row[-1]].value
                power=int(row[3])
                toughness=int(row[4])
                ability=[]
                if row[5]!="":
                    txt=list(row[5].split(","))
                    ability=[int(ele) for ele in txt]
                card_list[card_id].extend([power,toughness,ability])
            elif card_category=="Amulet":
                #mylogger.info("row_contents:{}".format(row))
                card_traits=Trait[row[-2]].value
                card_class=LeaderClass[row[-3]].value
                has_count=False
                if row[-1]!="False":
                    has_count=int(row[-1])
                ability=[]
                if row[3]!="":
                    txt=list(row[3].split(","))
                    ability=[int(ele) for ele in txt]
                card_list[card_id].append(ability)
                    
            elif card_category=="Spell":
                card_traits=Trait[row[-1]].value
                card_class=LeaderClass[row[-2]].value
            else:
                assert False,"{}".format(card_category)
            if card_class==LeaderClass["RUNE"].value:
                spell_boost=list(row[-3-int(card_category=="Amulet")].split(","))
                check_spellboost=[bool(int(spell_boost[i])) for i in range(2)]
                card_list[card_id].append([card_class,check_spellboost,card_traits])
            else:
                card_list[card_id].append([card_class,card_traits])
            if has_count!=None:
                card_list[card_id].append(has_count)
            card_list[card_id].append(card_name)
    return card_list

def tsv_to_dataframe(tsv_name):
    card_category=list(tsv_name.split("_"))[1]
    my_columns=[]
    sample=[]
    assert card_category in ["Creature","Spell","Amulet"]
    if card_category=="Creature":
        my_columns=["Card_id","Card_name","Cost","Power","Toughness","Ability","Class","Trait","Spell_boost"]
        sample=[0,"Sample",0,0,0,[],"NEUTRAL","NONE","None"]
    elif card_category=="Spell":
        my_columns=["Card_id","Card_name","Cost","Class","Trait","Spell_boost"]
        sample=[0,"Sample",0,"NEUTRAL","NONE","None"]
    elif card_category=="Amulet":
        my_columns=["Card_id","Card_name","Cost","Ability","Class","Trait","Spell_boost","Count_down"]
        sample=[0,"Sample",0,[],"NEUTRAL","NONE","None","None"]
        
    df=pd.DataFrame([sample],columns=my_columns)
    with open("Card_List_TSV/"+tsv_name) as f:
    #with open(tsv_name) as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            data=[]
            #[1,"Goblin",1,1,2,[],"NEUTRAL","NONE","NONE"]
            card_id=int(row[0])
            card_name=row[1]
            card_cost=int(row[2])
            data.append(card_id)
            data.append(card_name)
            data.append(card_cost)       
            
            
            card_class=None
            card_trait=None
            has_count=None
            if card_category=="Creature":
                card_class=LeaderClass[row[-2]].name
                card_trait=Trait[row[-1]].name
                power=int(row[3])
                toughness=int(row[4])
                ability=[]
                if row[5]!="":
                    txt=list(row[5].split(","))
                    ability=[KeywordAbility(int(ele)).value for ele in txt]
                data.extend([power,toughness,ability])
            elif card_category=="Amulet":
                card_trait=Trait[row[-2]].name
                card_class=LeaderClass[row[-3]].name
                has_count=False
                if row[-1]!="False":
                    has_count=int(row[-1])
                ability=[]
                if row[3]!="":
                    txt=list(row[3].split(","))
                    ability=[KeywordAbility(int(ele)).value for ele in txt]
                data.append(ability)
                    
            elif card_category=="Spell":
                card_trait=Trait[row[-1]].name
                card_class=LeaderClass[row[-2]].name
            else:
                assert False,"{}".format(card_category)
            if card_class==LeaderClass["RUNE"].name:
                spell_boost=list(row[-3-int(card_category=="Amulet")].split(","))
                check_spellboost=[bool(int(spell_boost[i])) for i in range(2)]
                spell_boost_type="None"
                if check_spellboost[0]==True:
                    if check_spellboost[1]==True:
                        spell_boost_type="Costdown"
                    else:
                        spell_boost_type="Normal"
                    
                data.extend([card_class,card_trait,spell_boost_type])
            else:
                data.extend([card_class,card_trait,"None"])
            if has_count!=None:
                data.append(has_count)
            new_df=pd.DataFrame([data],columns=my_columns)
            df = pd.concat([df, new_df])
            
            
            
    return df

def tsv_2_ability_dict(file_name,name_to_id=None):
    ability_dict={}
    with open("Card_List_TSV/"+file_name) as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            ability_dict[name_to_id[row[0]]]=int(row[1])
        
    return ability_dict

#creature_list=tsv_to_card_list("ALL_Creature_Card_List.tsv")
creature_list=tsv_to_card_list("New-All_Creature_Card_List.tsv")
#creature_df=tsv_to_dataframe("New-All_Creature_Card_List.tsv")
#creature_df.to_csv('DF_TSV/Creature_df.tsv',sep='\t',index=False)
#spell_df=tsv_to_dataframe("New-All_Spell_Card_List.tsv")
#spell_df.to_csv('DF_TSV/Spell_df.tsv',sep='\t',index=False)
#amulet_df=tsv_to_dataframe("New-All_Amulet_Card_List.tsv")
#amulet_df.to_csv('DF_TSV/Amulet_df.tsv',sep='\t',index=False)


creature_name_to_id={}
for key in list(creature_list.keys()):
    creature_name_to_id[creature_list[key][-1]]=key
#creature_fanfare_ability={9:1,10:2,11:3,12:4,13:5,16:6,21:7,22:8,23:9,24:10,25:11,34:16,35:5,37:17,38:18,\
#    39:19,40:20,42:22,43:23,44:24,45:25,46:26,48:27,50:29,51:30,58:38,61:1,62:40,66:43,68:46,69:47,70:48,71:49,\
#    73:33,74:51,75:52,76:53,80:55,81:56,82:57,84:59,85:61,87:64,88:65,89:67,3:68,94:71,98:73,99:74}
creature_fanfare_ability=tsv_2_ability_dict("All_fanfare_list.tsv",name_to_id=creature_name_to_id)
#creature_lastword_ability={11:3,19:3,32:15,49:28,52:3,53:33,55:35,56:36,57:37,59:39,67:45,72:50,85:62,90:69}
creature_lastword_ability=tsv_2_ability_dict("All_lastword_list.tsv",name_to_id=creature_name_to_id)
#creature_end_of_turn_ability={27:13,-12:31,84:60,86:63}
#creature_start_of_turn_ability={28:1}
#creature_has_target={12:1,13:1,35:1,43:3,46:1,48:2,50:2,69:3,70:3,80:1,85:3,88:8}
#creature_evo_effect={26:12,29:14,41:21,52:32,54:34,64:41,65:42,66:44,77:54,83:58,92:70,96:72}
creature_end_of_turn_ability=tsv_2_ability_dict("All_end_of_turn_list.tsv",name_to_id=creature_name_to_id)
creature_start_of_turn_ability=tsv_2_ability_dict("All_start_of_turn_list.tsv",name_to_id=creature_name_to_id)
creature_has_target=tsv_2_ability_dict("All_fanfare_target_list.tsv",name_to_id=creature_name_to_id)
creature_evo_effect=tsv_2_ability_dict("All_evo_effect_list.tsv",name_to_id=creature_name_to_id)
creature_has_evo_effect_target={29:1,41:1,83:2,96:72}
creature_target_regulation={46:lambda x:x.power>=5,48:lambda x:x.origin_cost==1,88:lambda card:card.card_category=="Creature" \
    and card.card_class.name=="NEUTRAL"}
another_target_func=lambda creature,itself:creature!=itself
evo_target_regulation={83:another_target_func}
player_attack_regulation=\
    {16:lambda player:len(player.field.get_creature_location()[1-player.player_num])<2}
creature_in_battle_ability_list={47:1,89:2}
creature_cost_change_ability_list={97:2}
can_only_attack_check=lambda field,player:field.check_word()[1-player.player_num]==True
creature_can_only_attack_list={49:can_only_attack_check}
creature_trigger_ability_dict={60:1,63:4,64:5,79:6,95:7,100:8}
special_evo_stats_id={26:1,27:3,28:1,29:1,41:1,52:1,66:1,77:1}
evo_stats={1:[1,1],2:[0,0],3:[3,1]}
creature_earth_rite_list=[67,68,71,90]
#1:相手のフォロワー,2:自分のフォロワー,3:相手のフォロワーと相手リーダー,
#4:自分と相手のフォロワー,5:自分と相手の全てのカード,6:自分の場のカード,7:自分の場のカードと相手の場のフォロワー,8:自分の他の手札
#9:相手の場の全てのカード
creature_enhance_list={3:[6],10:[6],87:[10],98:[9]}
creature_enhance_target_list={}
creature_enhance_target_regulation_list={}

creature_accelerate_list={90:[1]}
creature_accelerate_card_id_list={90:{1:-2}}
creature_accelerate_target_list={}
creature_accelerate_target_regulation_list={}
#spell_list=tsv_to_card_list("ALL_Spell_Card_List.tsv")
spell_list=tsv_to_card_list("New-All_Spell_Card_List.tsv")
"""
spell_list={1:[2,[3,[False],-1],"Witch Snap"],2:[1,[3,[False],-1],"Insight"],3:[4,[3,[False],-1],"Nova Flare"],4:[3,[2,-1],"Forge Weaponry"],\
    5:[6,[7,-1],"Themis's Decree"],6:[5,[0,-1],"Dance of Death"],7:[6,[2,-1],"Alwida's Command"],8:[2,[4,-1],"Dragon Oracle"],\
    9:[2,[3,[False],-1],"Magic Missile"],10:[2,[3,[False],-1],"Conjure Golem"],11:[2,[3,[True,False],-1],"Wind Blast"],\
    12:[5,[3,[True,True],-1],"Fate's Hand"],13:[8,[3,[True,True],-1],"Fiery Embrace"],14:[4,[3,[True,False],-1],"Fire Chain"],\
    15:[1,[0,-1],"Angelic Snipe"],16:[20,[3,[True,True],-1],"Dimension Shift"],17:[2,[3,[False],-1],"Kaleidoscopic Glow"],\

    18:[2,[6,-1],"Blood Pact"],19:[2,[6,-1],"Razory Claw"],20:[5,[6,-1],"Diabolic Drain"],21:[8,[6,-1],"Revelation"],\

    22:[2,[5,-1],"Undying Resentment"],23:[4,[5,-1],"Phantom Howl"],24:[6,[5,-1],"Death's Breath"],\

    25:[2,[7,-1],"Blackened Scripture"],\
    
    26:[1,[1,-1],"Nature's Guidance"],27:[1,[1,-1],"Fairy Circle"],28:[1,[1,-1],"Airbound Barrage"],\
    29:[2,[1,-1],"Sylvan Justice"],30:[2,[1,-1],"Pixie Mischief"],31:[5,[1,-1],"Will of the Forest"],\

    32:[1,[4,-1],"Blazing Breath"],33:[2,[4,-1],"Breath of the Salamander"],34:[5,[4,-1],"Draconic Fervor"],\
    35:[6,[4,-1],"Lightning Blast"],\
    
    36:[2,[0,-1],"Seraphic Blade"],\

    -1:[2,[3,[False],-1],"Veridic Ritual"],-2:[1,[3,[False],-1],"Orichalcum Golem(Accelerate 1)"]\
    }
"""
spell_name_to_id={}
for key in list(spell_list.keys()):
    spell_name_to_id[spell_list[key][-1]]=key
#spell_has_target={1:1,4:2,6:1,9:3,11:1,13:1,15:3,17:5,19:3,20:1,22:1,25:1,26:6,28:7,29:1,30:6,32:1,33:1,35:9,36:9,\
#    -1:3\
#    }
spell_has_target=tsv_2_ability_dict("All_spell_target_list.tsv",name_to_id=spell_name_to_id)
#1:相手のフォロワー,2:自分のフォロワー,3:相手のフォロワーと相手リーダー,
#4:自分と相手のフォロワー,5:自分と相手の全てのカード,6:自分の場のカード,7:自分の場のカードと相手の場のフォロワー,8:自分の他の手札
#9:相手の場の全てのカード
#spell_triggered_ability={1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,\
#    14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,\
#    28:28,29:29,30:30,31:31,32:32,33:33,34:34,35:35,36:36,\
#    -1:-1,-2:-2} 
spell_triggered_ability=tsv_2_ability_dict("All_spell_effect_list.tsv",name_to_id=spell_name_to_id)
spell_target_regulation={17:lambda x:x.origin_cost<=2,25:lambda x:x.toughness<=3,\
                        36:lambda x:x.origin_cost<=2}
spell_cost_change_ability_list={20:1,21:1}
spell_earth_rite_list=[]
spell_enhance_list={33:[6],35:[10],36:[6]}
spell_enhance_target_list={33:1,36:9}
spell_enhance_target_regulation_list={}

spell_accelerate_list={}
spell_accelerate_card_id_list={}
spell_accelerate_target_list={}
spell_accelerate_target_regulation_list={}
#amulet_list=tsv_to_card_list("ALL_Amulet_Card_List.tsv")
amulet_list=tsv_to_card_list("New-All_Amulet_Card_List.tsv")
"""
amulet_list={1:[2,[],[0,-1],False,"Well of Destiny"],2:[9,[],[4,-1],False,"Polyphonic Roar"],3:[4,[],[0,-1],False,"Path to Purgatory"],4:[1,[],[7,-1],3,"Sacred Plea"],5:[2,[],[7,-1],1,"Heretical Inquiry"],\
    6:[1,[],[7,-1],2,"Pinion Prayer"],7:[2,[],[7,-1],2,"Beastly Vow"],8:[3,[],[7,-1],2,"Divine Birdsong"],9:[5,[],[7,-1],3,"Forgotten Sanctuary"],\
    10:[1,[],[6,-1],4,"Bloodfed Flowerbed"],11:[3,[],[7,-1],False,"Elana's Prayer"],12:[3,[],[7,-1],8,"Whitefang Temple"],\
    13:[2,[],[7,-1],2,"Moriae Encomium"],14:[4,[],[7,-1],3,"Tribunal of Good and Evil"],\
    15:[1,[],[3,[False],-2],False,"Scrap Iron Smelter"],16:[2,[],[3,[False],-2],False,"Silent Laboratory"],17:[1,[],[3,[False],-2],False,"Witch's Cauldron"],\
        
        
    -1:[1,[],[3,[False],-2],False,"Earth Essence"]}
"""
amulet_name_to_id={}
for key in list(amulet_list.keys()):
    amulet_name_to_id[amulet_list[key][-1]]=key
Earth_sigil_list=[-1,15,16]
amulet_start_of_turn_ability={1:1,2:2}
amulet_end_of_turn_ability={3:3,10:10,12:11}
amulet_fanfare_ability={9:9, 13:13, 14:15, 15:16, 16:17}
amulet_lastword_ability={4:4,5:5,6:6,7:7,8:8,9:9,12:12,13:14,14:14,17:13}
amulet_has_target={14:1}
amulet_trigger_ability_dict={11:2,12:3}
#amulet_countdown_list={4:3,5:1,6:2,7:2,8:2,9:3}
amulet_target_regulation={}
amulet_cost_change_ability_list={}
amulet_earth_rite_list=[]
amulet_enhance_list={}
amulet_enhance_target_list={}
amulet_enhance_target_regulation_list={}

amulet_accelerate_list={}
amulet_accelerate_card_id_list={}
amulet_accelerate_target_list={}
amulet_accelerate_target_regulation_list={}
class_card_list={}
for i in range(8):
    class_card_list[i]={"Creature":{},"Spell":{},"Amulet":{}}
for i in list(creature_list):
    class_num=creature_list[i][4][0]
    class_card_list[class_num]["Creature"][i]=creature_list[i]
for i in list(spell_list):
    class_num=spell_list[i][1][0]
    class_card_list[class_num]["Spell"][i]=spell_list[i]
for i in list(amulet_list):
    class_num=amulet_list[i][2][0]
    class_card_list[class_num]["Amulet"][i]=amulet_list[i]


class Card:
    def __init__(self,card_id):
        assert False
    def get_copy(self):
        assert False
    def get_damage(self,amount):
        assert False
    def get_current_toughness(self):
        assert False
    def can_attack_to_follower(self):
        assert False
    def can_attack_to_player(self):
        assert False
    def can_be_attacked(self):
        assert False
    def __str__(self):
        assert False
    def untap(self):
        return
    def down_count(self,num=1,virtual=False):
        return



        
        
class Creature(Card):
    def __init__(self,card_id):
        self.card_id=card_id#カードid
        self.card_category="Creature"
        
        self.cost=creature_list[self.card_id][0]#カードのコスト
        self.origin_cost=creature_list[self.card_id][0]#カードの元々のコスト
        self.power=creature_list[self.card_id][1]#カードの攻撃力
        self.toughness=creature_list[self.card_id][2]#カードの体力
        """
        itself_df=creature_df[creature_df["Card_id"]==card_id]
        self.cost=int(itself_df["Cost"])#カードのコスト
        self.origin_cost=int(itself_df["Cost"])#カードの元々のコスト
        self.power=int(itself_df["Power"])#カードの攻撃力
        self.toughness=int(itself_df["Toughness"])#カードの体力
        """
        self.buff=[0,0]#スタッツ上昇量
        self.until_turn_end_buff=[0,0]#ターン終了時までのスタッツ上昇量
        self.target_regulation=None
        if card_id in creature_target_regulation:
            self.target_regulation=creature_target_regulation[card_id]
        self.player_attack_regulation=None
        if card_id in player_attack_regulation:
            self.player_attack_regulation=player_attack_regulation[card_id]
        self.evo_stat=[2,2]
        if card_id in special_evo_stats_id:
            self.evo_stat=evo_stats[special_evo_stats_id[card_id]]

        #self.ability=copy.copy(creature_list[self.card_id][3])#カードのキーワード能力idリスト
        self.ability=creature_list[self.card_id][3][:]
        #print(itself_df)
        """
        self.ability=[KeywordAbility(int(ability)).value for ability in itself_df["Ability"][0]]
        """
        self.fanfare_ability=None
        if card_id in creature_fanfare_ability:
            self.fanfare_ability=creature_ability_dict[creature_fanfare_ability[card_id]]
        
        self.lastword_ability=[]
        if card_id in creature_lastword_ability:
            self.lastword_ability.append(creature_ability_dict[creature_lastword_ability[card_id]])
        
        self.have_target=0
        if card_id in creature_has_target:
            self.have_target=creature_has_target[card_id]

        self.evo_effect=None
        self.evo_target=None
        if card_id in creature_evo_effect:
            self.evo_effect=creature_ability_dict[creature_evo_effect[card_id]]
            if card_id in creature_has_evo_effect_target:
                self.evo_target=creature_has_evo_effect_target[card_id]
            self.evo_target_regulation=None
            if card_id in evo_target_regulation:
                self.evo_target_regulation=evo_target_regulation[card_id]
                


        self.turn_start_ability=[]
        if card_id in creature_start_of_turn_ability:
            #mylogger.info("ability_id;{}".format(creature_start_of_turn_ability[card_id]))
            self.turn_start_ability.append(creature_ability_dict[creature_start_of_turn_ability[card_id]])

        self.turn_end_ability=[]
        if card_id in creature_end_of_turn_ability:
            self.turn_end_ability.append(creature_ability_dict[creature_end_of_turn_ability[card_id]])
        self.trigger_ability=[]
        if card_id in creature_trigger_ability_dict:
            self.trigger_ability.append(trigger_ability_dict[creature_trigger_ability_dict[card_id]]())

        self.name=creature_list[self.card_id][-1]
        #self.name=str(itself_df["Card_name"][0])
        self.is_in_field=False
        self.is_in_graveyard=False
        self.damage=0
        self.is_tapped=True
        #self.attacked_flg=False
        self.can_attack_num=1
        self.current_attack_num=0
        self.evolved=False
        self.can_only_attack_target=None
        if card_id in creature_can_only_attack_list:
            self.can_only_attack_target=creature_can_only_attack_list[card_id]
        self.in_battle_ability=[]
        if card_id in creature_in_battle_ability_list:
            self.in_battle_ability.append(battle_ability_dict[creature_in_battle_ability_list[card_id]])
        self.cost_change_ability=None
        if card_id in creature_cost_change_ability_list:
            self.cost_change_ability=cost_change_ability_dict[creature_cost_change_ability_list[card_id]]

        self.card_class = LeaderClass(creature_list[card_id][4][0])
        self.trait = Trait(creature_list[card_id][4][-1])    
        if self.card_class.name == "RUNE":
            self.spell_boost=None
            if creature_list[card_id][4][1][0]==True:
                self.spell_boost=0
                self.cost_down=creature_list[card_id][4][1][1]
        
        self.is_earth_rite=card_id in creature_earth_rite_list
        self.have_enhance=card_id in creature_enhance_list
        self.active_enhance_code=[False,0]
        if self.have_enhance==True:
            self.enhance_cost=creature_enhance_list[card_id]
            self.active_enhance_code=[False,0]
            self.enhance_target=0
            self.enhance_target_regulation=None
            if card_id in creature_enhance_target_list:
                self.enhance_target=creature_enhance_target_list[card_id] 
                if card_id in creature_enhance_target_regulation_list:
                    self.enhance_target_regulation=creature_enhance_target_regulation_list[card_id]

        self.have_accelerate=card_id in creature_accelerate_list
        self.active_accelerate_code=[False,0]
        if self.have_accelerate==True:
            self.accelerate_cost=creature_accelerate_list[card_id]
            self.accelerate_card_id=creature_accelerate_card_id_list[card_id]
            self.active_accelerate_code=[False,0]
            self.accelerate_target=0
            self.accelerate_target_regulation=None
            if card_id in creature_accelerate_target_list:
                self.accelerate_target=creature_accelerate_target_list[card_id] 
                if card_id in creature_accelerate_target_regulation_list:
                    self.accelerate_target_regulation=creature_accelerate_target_regulation_list[card_id]

        return 
                
        
    def get_copy(self):
        creature=Creature(self.card_id)
        creature.cost=int(self.cost)#カードのコスト
        creature.power=int(self.power)#カードの攻撃力
        creature.toughness=int(self.toughness)#カードの体力
        creature.buff=self.buff[:]#スタッツ上昇量
        creature.until_turn_end_buff=self.until_turn_end_buff[:]#ターン終了時までのスタッツ上昇量
        creature.ability=self.ability[:]
        creature.lastword_ability=self.lastword_ability[:]
        if len(creature.turn_start_ability)!=len(self.turn_start_ability):
            creature.turn_start_ability=copy.deepcopy(self.turn_start_ability)
        if len(creature.turn_end_ability)!=len(self.turn_end_ability):
            creature.turn_end_ability=copy.deepcopy(self.turn_end_ability)
        if len(creature.trigger_ability)!=len(self.trigger_ability):
            creature.trigger_ability=copy.deepcopy(self.trigger_ability)
        
        creature.is_in_field=self.is_in_field
        creature.is_in_graveyard=self.is_in_graveyard
        creature.damage=int(self.damage)
        creature.is_tapped=self.is_tapped
        #creature.attacked_flg=self.attacked_flg
        creature.current_attack_num=int(self.current_attack_num)
        creature.can_attack_num=int(self.can_attack_num)
        creature.evolved=self.evolved
        if len(creature.in_battle_ability)!=len(self.in_battle_ability):
            creature.in_battle_ability=copy.deepcopy(self.in_battle_ability)
        #if self.in_battle_ability!=[]:
        #    creature.in_battle_ability=copy.deepcopy(self.in_battle_ability)
        if self.card_class.name == "RUNE":
            creature.spell_boost=None
            if creature_list[self.card_id][4][1][0]==True:
                creature.spell_boost=int(self.spell_boost)
                creature.cost_down=self.cost_down
        
        return creature
    def untap(self):
        self.is_tapped=False
        #self.attacked_flg=False
        self.current_attack_num=0

    def evolve(self,field,target,player_num=0,virtual=False,auto=False):
        self.evolved=True
        self.power+=self.evo_stat[0]
        self.toughness+=self.evo_stat[1]
        if auto==True: return
        if self.evo_effect!=None:
            self.evo_effect(field,field.players[player_num],field.players[1-player_num],virtual,target,self)

    def get_damage(self,amount):
        if KeywordAbility.REDUCE_DAMAGE_TO_ZERO.value not in self.ability:
            self.damage+=amount
            if(self.toughness-self.damage<=0):
                self.is_in_field=False
                self.is_in_graveyard=True
            return amount
        else:
            return 0

    def get_current_toughness(self):
        return self.toughness-self.damage
        
    def can_attack_to_follower(self):
        if self.current_attack_num>=self.can_attack_num:
            #raise Exception("{} {}".format(self.current_attack_num,self.can_attack_num))
            return False
        #if self.attacked_flg==True: return False
        if self.evolved==True: return True
        if any((i in self.ability) for i in [KeywordAbility.STORM.value,KeywordAbility.RUSH.value]): return True
        if self.is_tapped==False: return True
        return False

    def can_attack_to_player(self):
        if self.current_attack_num>=self.can_attack_num:
            #raise Exception("{} {}".format(self.current_attack_num,self.can_attack_num))
            return False
        #mylogger.info("{}:type={}".format(KeywordAbility.STORM,type(KeywordAbility.STORM)))
        #if self.attacked_flg==True: return False 
        if KeywordAbility.STORM.value in self.ability: return True
        if self.is_tapped==False: return True
        return False
    def can_be_targeted(self):
        return not any(i in self.ability for i in [KeywordAbility.CANT_BE_TARGETED.value,KeywordAbility.AMBUSH.value])

    def can_be_attacked(self):
        return not any(i in self.ability for i in [KeywordAbility.CANT_BE_ATTACKED.value,KeywordAbility.AMBUSH.value])

    def __str__(self):
        text=""
        default_color="\033[0m"
        if self.is_in_field==True:
            if self.can_attack_to_player():
                text+="\033[36m"
                default_color="\033[36m"
            elif self.can_attack_to_follower():
                text+="\033[33m"
                default_color="\033[33m"
        if self.is_in_field==True:
            text+="name:{:<25} {}/{}/{}".format(self.name,str(self.origin_cost),str(self.power),str(self.toughness-self.damage))
        else:
            text+="name:{:<25} {}/{}/{}".format(self.name,str(self.cost),str(self.power),str(self.toughness))
            if self.have_enhance==True and self.active_enhance_code[0]==True:
                text+=" enhance:{}".format(self.active_enhance_code[1])
            elif self.have_accelerate==True and self.active_accelerate_code[0]==True:
                text+=" accelerate:{}".format(self.active_accelerate_code[1])
        if self.card_class.name == "RUNE" and self.spell_boost!=None and self.is_in_field==False:
            text+=" spell_boost:{:<2}".format(self.spell_boost)
        if self.ability!=[] and self.is_in_field==True:
            text+=" ability={}".format([KeywordAbility(i).name for i in self.ability])
        text+="\033[0m"
        return text


class Spell(Card):
    def __init__(self,card_id):
        self.card_id=card_id#カードid
        self.card_category="Spell"
        self.cost=spell_list[self.card_id][0]#カードのコスト
        self.origin_cost=spell_list[self.card_id][0]#カードの元々のコスト
        self.target_regulation=None
        if card_id in spell_target_regulation:
            self.target_regulation=spell_target_regulation[card_id]
        self.triggered_ability=[spell_ability_dict[spell_triggered_ability[card_id]]]
        self.have_target=0
        if card_id in spell_has_target:
            self.have_target=spell_has_target[card_id]
        self.name=spell_list[self.card_id][-1]
        self.is_in_graveyard=False
        self.cost_change_ability=None
        if card_id in spell_cost_change_ability_list:
            self.cost_change_ability=cost_change_ability_dict[spell_cost_change_ability_list[card_id]]

        self.card_class = LeaderClass(spell_list[card_id][1][0])
        self.trait = Trait(spell_list[card_id][1][-1])    
        if self.card_class.name == "RUNE":
            self.spell_boost=None
            if spell_list[card_id][1][1][0]==True:
                self.spell_boost=0
                self.cost_down=spell_list[card_id][1][1][1]

        self.is_earth_rite=card_id in spell_earth_rite_list
        self.have_enhance=card_id in spell_enhance_list
        self.active_enhance_code=[False,0]
        if self.have_enhance==True:
            self.enhance_cost=spell_enhance_list[card_id]
            self.active_enhance_code=[False,0]
            self.enhance_target=0
            self.enhance_target_regulation=None
            if card_id in spell_enhance_target_list:
                self.enhance_target=spell_enhance_target_list[card_id] 
                if card_id in spell_enhance_target_regulation_list:
                    self.enhance_target_regulation=spell_enhance_target_regulation_list[card_id]

        self.have_accelerate=card_id in spell_accelerate_list
        self.active_accelerate_code=[False,0]
        if self.have_accelerate==True:
            self.accelerate_cost=spell_accelerate_list[card_id]
            self.accelerate_card_id=spell_accelerate_card_id_list[card_id]
            self.active_accelerate_code=[False,0]
            self.accelerate_target=0
            self.accelerate_target_regulation=None
            if card_id in spell_accelerate_target_list:
                self.accelerate_target=spell_accelerate_target_list[card_id] 
                if card_id in spell_accelerate_target_regulation_list:
                    self.accelerate_target_regulation=spell_accelerate_target_regulation_list[card_id]

    def get_copy(self):
        spell=Spell(self.card_id)
        spell.cost=int(self.cost)
        if self.card_class.name == "RUNE":
            spell.spell_boost=None
            if spell_list[self.card_id][1][1][0]==True:
                spell.spell_boost=int(self.spell_boost)
                spell.cost_down=self.cost_down
        return spell
    def __str__(self):
        text="name:"+'{:<25}'.format(self.name)+" cost: "+'{:<2}'.format(str(self.cost))
        if self.card_class.name == "RUNE" and self.spell_boost!=None:
            text+=" spell_boost:{:<2}".format(self.spell_boost)
        if self.have_enhance==True and self.active_enhance_code[0]==True:
            text+=" enhance:{}".format(self.active_enhance_code[1])
        return text

class Amulet(Card):
    def __init__(self,card_id):
        self.card_id=card_id#カードid
        self.card_category="Amulet"
        self.cost=amulet_list[self.card_id][0]#カードのコスト
        self.origin_cost=amulet_list[self.card_id][0]#カードの元々のコスト
        self.can_not_be_targeted=6 in amulet_list[card_id][1]#能力の対象にならないを持つか
        self.ability=amulet_list[card_id][1][:]
        self.trigger_ability=[]
        if card_id in amulet_trigger_ability_dict:
            self.trigger_ability.append(trigger_ability_dict[amulet_trigger_ability_dict[card_id]]())
        self.target_regulation=None
        if card_id in amulet_target_regulation:
            self.target_regulation=amulet_target_regulation[card_id]

        self.fanfare_ability=None
        if card_id in amulet_fanfare_ability:
            self.fanfare_ability=amulet_ability_dict[amulet_fanfare_ability[card_id]]
        self.lastword_ability=[]
        if card_id in amulet_lastword_ability:
            self.lastword_ability.append(amulet_ability_dict[amulet_lastword_ability[card_id]])

        self.have_target=0
        if card_id in amulet_has_target:
            self.have_target=amulet_has_target[card_id]

        self.turn_start_ability=[]
        if card_id in amulet_start_of_turn_ability:
            self.turn_start_ability.append(amulet_ability_dict[amulet_start_of_turn_ability[card_id]])
            #mylogger.info("ability exist")


        self.turn_end_ability=[]
        if card_id in amulet_end_of_turn_ability:
            self.turn_end_ability.append(amulet_ability_dict[amulet_end_of_turn_ability[card_id]])
            #mylogger.info("ability exist")
        self.name=amulet_list[self.card_id][-1]
        self.is_in_graveyard=False
        self.is_in_field=False
        self.countdown=False
        self.ini_count=0
        self.current_count=0
        if amulet_list[card_id][3]!=False:
            self.countdown=True
            self.ini_count=amulet_list[card_id][3]
            self.current_count=amulet_list[card_id][3]
        self.cost_change_ability=None
        if card_id in amulet_cost_change_ability_list:
            self.cost_change_ability=cost_change_ability_dict[amulet_cost_change_ability_list[card_id]]

        self.card_class = LeaderClass(amulet_list[card_id][2][0])
        self.trait = Trait(amulet_list[card_id][2][-1])   
        if self.card_class.name == "RUNE":
            self.spell_boost=None
            if amulet_list[card_id][2][1][0]==True:
                self.spell_boost=0
                self.cost_down=amulet_list[card_id][2][1][1]

        self.is_earth_sigil=self.trait.name == "EARTH_SIGIL" 
        self.is_earth_rite=card_id in amulet_earth_rite_list

        self.have_enhance=card_id in amulet_enhance_list
        self.active_enhance_code=[False,0]
        if self.have_enhance==True:
            self.enhance_cost=amulet_enhance_list[card_id]
            self.active_enhance_code=[False,0]
            self.enhance_target=0
            self.enhance_target_regulation=None
            if card_id in amulet_enhance_target_list:
                self.enhance_target=amulet_enhance_target_list[card_id] 
                if card_id in amulet_enhance_target_regulation_list:
                    self.enhance_target_regulation=amulet_enhance_target_regulation_list[card_id]

        self.have_accelerate=card_id in amulet_accelerate_list
        self.active_accelerate_code=[False,0]
        if self.have_accelerate==True:
            self.accelerate_cost=amulet_accelerate_list[card_id]
            self.accelerate_card_id=amulet_accelerate_card_id_list[card_id]
            self.active_accelerate_code=[False,0]
            self.accelerate_target=0
            self.accelerate_target_regulation=None
            if card_id in amulet_accelerate_target_list:
                self.accelerate_target=amulet_accelerate_target_list[card_id] 
                if card_id in amulet_accelerate_target_regulation_list:
                    self.accelerate_target_regulation=amulet_accelerate_target_regulation_list[card_id]

    def get_copy(self):
        amulet=Amulet(self.card_id)
        amulet.cost=int(self.cost)
        amulet.is_in_field=self.is_in_field
        amulet.current_count=int(self.current_count)
        if self.card_class.name == "RUNE":
            self.spell_boost=None
            if amulet_list[self.card_id][2][1][0]==True:
                amulet.spell_boost=int(self.spell_boost)
                amulet.cost_down=self.cost_down
        return amulet

    def can_be_targeted(self):
        return not any(i in self.ability for i in [KeywordAbility.CANT_BE_TARGETED.value,KeywordAbility.AMBUSH.value])

    def down_count(self,num=1,virtual=False):
        if self.countdown==False: return
        self.current_count-=num
        if virtual==False: mylogger.info("{}'s count down by {}".format(self.name,num))
        if self.current_count<=0:
            self.is_in_graveyard=True
            self.is_in_field=False
            self.current_count=self.ini_count
    def __str__(self):
        if self.is_in_field==False:
            tmp = "name:"+'{:<25}'.format(self.name)+" cost: "+'{:<2}'.format(str(self.cost))
        else:
            tmp = "name:"+'{:<25}'.format(self.name)+" cost: "+'{:<2}'.format(str(self.origin_cost))

        if self.countdown==True:
            tmp=tmp+" count:{:<2}".format(self.current_count)
        if self.have_enhance==True and self.active_enhance_code[0]==True:
            text+=" enhance:{}".format(self.active_enhance_code[1])
        return tmp
            
class Deck:
        def __init__(self):
            self.deck=[]
            self.remain_num=0
            self.mean_cost=0
            self.deck_type=None
                
        def append(self,card,num=1):
            for i in range(num):
                #self.deck.append(copy.deepcopy(card))#各カードは別のカードなので一枚ずつ生成する必要がある
                self.deck.append(card.get_copy())
                #self.deck.append(card)
                self.remain_num+=1

        def show_all(self):
            print("Deck contents")
            print("==============================================")
            for i in range(self.remain_num):
                print(self.deck[self.remain_num-i-1])#引くのはlistの最後尾から
            print("==============================================")
        def draw(self):
            self.remain_num-=1
            return self.deck.pop()

        def shuffle(self):
            random.shuffle(self.deck)

        def get_mean_cost(self):
            sum_of_cost=0
            for card in self.deck:
                sum_of_cost+=card.origin_cost
            return sum_of_cost/len(self.deck)
        
        def set_deck_type(self,type_num):
            self.deck_type=DeckType(type_num)
            mylogger.info("Deck_Type:{}".format(self.deck_type.name))
            #1はAggro,2はMid,3はControl,4はCombo

        def get_cost_histgram(self):
            histgram_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
            for card in self.deck:
                if card.origin_cost<=1:
                    histgram_dict[1]+=1
                elif card.origin_cost>=8:
                    histgram_dict[8]+=1
                else:
                    histgram_dict[card.cost]+=1
            histgram=list(histgram_dict.values())
            height_histgram=["" for i in range(max(histgram))]
            for i in range(max(histgram)):
                for j in range(len(histgram)):
                    if histgram[j]>=i:
                        height_histgram[i]+=" ■ "
                    else:
                        height_histgram[i]+="    "

            for i in range(len(height_histgram)):
                print(height_histgram[len(height_histgram)-1-i])

            print("~1   2   3   4   5   6   7   8+")
