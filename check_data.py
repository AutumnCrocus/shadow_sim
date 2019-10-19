from abc import ABCMeta, abstractmethod
import copy 
from enum import Enum
from card_setting import creature_list,creature_fanfare_ability,creature_lastword_ability,creature_end_of_turn_ability,creature_start_of_turn_ability,creature_has_target,\
    creature_evo_effect,creature_has_evo_effect_target

class LeaderClass(Enum):
    NEUTRAL = 0
    FOREST = 1
    SWORD = 2
    RUNE = 3
    DRAGON = 4
    SHADOW = 5
    BLOOD = 6
    HAVEN = 7
    PORTAL = 8
    
class Trait(Enum):
    EARTH_SIGIL = -2
    NONE = -1
    OFFICER = 0
    COMMANDER = 1

class DeckType(Enum):
    AGGRO = 1
    MID = 2
    CONTROL =3
    COMBO = 4

new_creature_card_list={}
for i in range(9):
    new_creature_card_list[i]={}
    new_card_id={}
    token_id={}
    for key in list(creature_list.keys()):
        
        if creature_list[key][4][0]==i:
            #print("card_data:{}".format(creature_list[key]))
            if key>=0:
                if creature_list[key][0] not in new_creature_card_list[i]:
                    new_creature_card_list[i][creature_list[key][0]]={}
                    new_card_id[creature_list[key][0]]=0
                new_creature_card_list[i][creature_list[key][0]][new_card_id[creature_list[key][0]]]=creature_list[key]
                new_card_id[creature_list[key][0]]+=1
            else:
                if creature_list[key][0] not in token_id:
                    token_id[creature_list[key][0]]=-1
                if creature_list[key][0] not in new_creature_card_list[i]:
                    new_creature_card_list[i][creature_list[key][0]]={}
                    
                new_creature_card_list[i][creature_list[key][0]][token_id[creature_list[key][0]]]=creature_list[key]
                token_id[creature_list[key][0]]-=1

for i in range(9):
    print("card_pool[{}]".format(i))
    print("{}".format(new_creature_card_list[i]))
    print("")

new_creature_fanfare_list={0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{}}
for i in range(8):
    for j in range(21):
        if j in new_creature_card_list[i]:
            new_creature_fanfare_list[i][j]={}
for key in list(creature_list.keys()):
    if key in creature_fanfare_ability:
        card_id=None
        for new_key in list(new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]].keys()):
            if new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]][new_key][-1]==creature_list[key][-1]:
                card_id=new_key
                break
        new_creature_fanfare_list[creature_list[key][4][0]][creature_list[key][0]][card_id]=creature_fanfare_ability[key]

#for i in range(9):
#    print("fanfare[{}]".format(i))
#    print("{}".format(new_creature_fanfare_list[i]))
#    print("")


creature_lastword_ability={11:3,19:3,32:15,49:28,52:3,53:33,55:35,56:36,57:37,59:39,67:45,72:50,85:62}
new_creature_lastword_list={0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{}}
for i in range(8):
    for j in range(21):
        if j in new_creature_card_list[i]:
            new_creature_lastword_list[i][j]={}
for key in list(creature_list.keys()):
    if key in creature_lastword_ability:
        card_id=None
        for new_key in list(new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]].keys()):
            if new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]][new_key][-1]==creature_list[key][-1]:
                card_id=new_key
                break
        new_creature_lastword_list[creature_list[key][4][0]][creature_list[key][0]][card_id]=creature_lastword_ability[key]

#for i in range(9):
#    print("lastwords[{}]".format(i))
#    print("{}".format(new_creature_lastword_list[i]))
#    print("")


new_creature_end_of_turn_ability_list={0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{}}
for i in range(8):
    for j in range(21):
        if j in new_creature_card_list[i]:
            new_creature_end_of_turn_ability_list[i][j]={}
for key in list(creature_list.keys()):
    if key in creature_end_of_turn_ability:
        card_id=None
        for new_key in list(new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]].keys()):
            if new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]][new_key][-1]==creature_list[key][-1]:
                card_id=new_key
                break
        new_creature_end_of_turn_ability_list[creature_list[key][4][0]][creature_list[key][0]][card_id]=creature_end_of_turn_ability[key]

#for i in range(9):
#    print("end_of_turn[{}]".format(i))
#    print("{}".format(new_creature_end_of_turn_ability_list[i]))
#    print("")


new_creature_start_of_turn_ability_list={0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{}}
for i in range(8):
    for j in range(21):
        if j in new_creature_card_list[i]:
            new_creature_start_of_turn_ability_list[i][j]={}
for key in list(creature_list.keys()):
    if key in creature_start_of_turn_ability:
        card_id=None
        for new_key in list(new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]].keys()):
            if new_creature_card_list[creature_list[key][4][0]][creature_list[key][0]][new_key][-1]==creature_list[key][-1]:
                card_id=new_key
                break
        new_creature_start_of_turn_ability_list[creature_list[key][4][0]][creature_list[key][0]][card_id]=creature_start_of_turn_ability[key]
for i in range(9):
    print("start_of_turn[{}]".format(i))
    print("{}".format(new_creature_start_of_turn_ability_list[i]))
    print("")
"""
creature_target_regulation={46:lambda x:x.power>=5,48:lambda x:x.origin_cost<=1,88:lambda card:card.card_category=="Creature" \
    and card.card_class.name=="NEUTRAL"}
another_target_func=lambda creature,itself:creature!=itself
evo_target_regulation={83:another_target_func}
creature_attack_regulation=\
    {16:lambda field,player:len(field.get_creature_location()[1-player.player_num])<2}
creature_in_battle_ability_list={47:1,89:2}
creature_cost_change_ability_list={}
can_only_attack_check=lambda field,player:field.check_word()[1-player.player_num]==True
creature_can_only_attack_list={49:can_only_attack_check}
creature_trigger_ability_dict={60:1,63:4,64:5,79:6}
special_evo_stats_id={26:1,27:3,28:1,29:1,41:1,52:1,66:1,77:1}
evo_stats={1:[1,1],2:[0,0],3:[3,1]}
creature_earth_rite_list=[67,68,71]
#1:相手のフォロワー,2:自分のフォロワー,3:相手のフォロワーと相手リーダー,
#4:自分と相手のフォロワー,5:自分と相手の全てのカード,6:自分の場のカード,7:自分の場のカードと相手の場のフォロワー,8:自分の他の手札
#9:相手の場の全てのカード
creature_enhance_list={10:[6],87:[10]}
creature_enhance_target_list={}
creature_enhance_target_regulation_list={}
"""
class Card:
    __metaclass__ = ABCMeta
    """
    @abstractmethod
    def __init__(self,card_id):
        pass
    """
class New_Creature(Card):
    def __init__(self,card_id,card_class_num,cost):
        if card_id not in new_creature_card_list[card_class_num][cost]:
            raise Exception("Key Error:{}".format(card_id))
        self.card_id=card_id#カードid
        self.card_class_num=card_class_num
        self.card_category="Creature"
        self.cost=cost#カードのコスト
        self.origin_cost=cost#カードの元々のコスト
        self.power=new_creature_card_list[card_class_num][cost][card_id][1]#カードの攻撃力
        self.toughness=new_creature_card_list[card_class_num][cost][card_id][2]#カードの体力
        self.buff=[0,0]#スタッツ上昇量
        self.until_turn_end_buff=[0,0]#ターン終了時までのスタッツ上昇量
        """
        self.target_regulation=None
        if card_id in creature_target_regulation:
            self.target_regulation=creature_target_regulation[card_id]
        self.attack_regulation=None
        if card_id in creature_attack_regulation:
            self.attack_regulation=creature_attack_regulation[card_id]
        self.evo_stat=[2,2]
        if card_id in special_evo_stats_id:
            self.evo_stat=evo_stats[special_evo_stats_id[card_id]]
        """
        #self.ability=copy.copy(creature_list[self.card_id][3])#カードのキーワード能力idリスト
        self.ability=new_creature_card_list[card_class_num][cost][card_id][3][:]
        
        self.fanfare_ability=None
        if card_id in new_creature_fanfare_list[card_class_num][cost]:
            self.fanfare_ability=new_creature_fanfare_list[card_class_num][cost][card_id]
            
        self.lastword_ability=[]
        if card_id in new_creature_lastword_list[card_class_num][cost]:
            self.lastword_ability.append(new_creature_lastword_list[card_class_num][cost][card_id])
        """
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
        self.trigger_ability=None
        if card_id in creature_trigger_ability_dict:
            self.trigger_ability=trigger_ability_dict[creature_trigger_ability_dict[card_id]]()
        """
        self.name=new_creature_card_list[card_class_num][cost][card_id][-1]
        self.is_in_field=False
        self.is_in_graveyard=False
        self.damage=0
        self.is_tapped=True
        self.attacked_flg=False
        self.evolved=False
        self.can_creature_attack=4 in new_creature_card_list[card_class_num][cost][card_id][3]#突進を持つか
        self.can_not_be_attacked=5 in new_creature_card_list[card_class_num][cost][card_id][3]#攻撃されないを持つか
        self.can_not_be_targeted=6 in new_creature_card_list[card_class_num][cost][card_id][3]#能力の対象にならないを持つか
        """
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
        if self.have_enhance==True:
            self.enhance_cost=creature_enhance_list[card_id]
            self.active_enhance_code=[False,0]
            self.enhance_target=0
            self.enhance_target_regulation=None
            if card_id in creature_enhance_target_list:
                self.enhance_target=creature_enhance_target_list[card_id] 
                if card_id in creature_enhance_target_regulation_list:
                    self.enhance_target_regulation=creature_enhance_target_regulation_list[card_id]
        """

        return 
                
        


    def get_damage(self,amount):

        self.damage+=amount

        if(self.toughness-self.damage<=0):
            self.is_in_field=False
            self.is_in_graveyard=True
        return amount

    def get_current_toughness(self):
        return self.toughness-self.damage
    """
    def __str__(self):
        text=""
        if self.is_in_field==True and self.attacked_flg==False:
            if 1 in self.ability or self.is_tapped==False:
                text+="\033[36m"
            elif 4 in self.ability or self.evolved==True:
                text+="\033[33m"
        if self.is_in_field==True:
            text+="name:"+'{:<25}'.format(self.name)+" cost: "+'{:<2}'.format(str(self.origin_cost))+" power: "+'{:<2}'.format(str(self.power))+\
                    " toughness: "+'{:<2}'.format(str(self.toughness-self.damage))
        else:
            text+="name:"+'{:<25}'.format(self.name)+" cost: "+'{:<2}'.format(str(self.cost))+" power: "+'{:<2}'.format(str(self.power))+\
                    " toughness: "+'{:<2}'.format(str(self.toughness-self.damage))
            if self.have_enhance==True and self.active_enhance_code[0]==True:
                text+=" enhance:{}".format(self.active_enhance_code[1])
        if self.card_class.name == "RUNE" and self.spell_boost!=None and self.is_in_field==False:
            text+=" spell_boost:{:<2}".format(self.spell_boost)
        if self.ability!=[] and self.is_in_field==True:
            text+=" ability={}".format(self.ability)
        text+="\033[0m"
        return text
    """


Test_Creature=New_Creature(0,0,1)

print("name:{} card_class_num:{} cost:{} power:{} toughness:{}".format(Test_Creature.name,Test_Creature.card_class_num,Test_Creature.cost,Test_Creature.power,Test_Creature.toughness))