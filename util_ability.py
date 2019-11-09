import random
import card_setting
from my_moduler import get_module_logger
mylogger = get_module_logger(__name__)
from my_enum import *
counter=0
def summon_creature(field,player,virtual,name=None,num=1):
    for i in range(num):
        if len(field.card_location[player.player_num])>=field.max_field_num:
            return
        else:
            if virtual==False:
                mylogger.info("Summon {}".format(name)) 
            creature=card_setting.Creature(card_setting.creature_name_to_id[name])
            #mylogger.info("new creature id:{}".format(id(creature)))
            field.set_card(creature,player.player_num,virtual=virtual)
            if num==1:
                return creature

def set_amulet(field,player,virtual,name=None,num=1):
    for i in range(num):
        if len(field.card_location[player.player_num])>=field.max_field_num:
            return
        else:
            if virtual==False:
                mylogger.info("Set {}".format(name)) 
            field.set_card(card_setting.Amulet(card_setting.amulet_name_to_id[name]),player.player_num,virtual=virtual)

def buff_creature(creature,params=[0,0]):
        if creature.card_category=="Creature":
            creature.power+=params[0]
            creature.toughness+=params[1]
            creature.buff[0]+=params[0]
            creature.buff[1]+=params[1]

def buff_creature_until_end_of_turn(creature,params=[0,0]):
        if creature.card_category=="Creature":
            creature.power+=params[0]
            creature.buff[0]+=params[0]
            creature.until_turn_end_buff[0]+=params[0]


def get_damage_to_creature(field,opponent,virtual,target,num=0):
    if target in field.get_creature_location()[opponent.player_num]:
        damage=field.card_location[opponent.player_num][target].get_damage(num)
        if virtual==False:
            mylogger.info("Player {}'s {} get {} damage".format(opponent.player_num+1,\
                field.card_location[opponent.player_num][target].name,damage))
        
        #if field.card_location[opponent.player_num][target].is_in_graveyard==True:
        #    field.remove_card([opponent.player_num,target],virtual)
    elif target!=None:
        mylogger.info("player_num:{} can_be_targeted:{} field_len:{} target:{}".format(1-opponent.player_num,field.get_can_be_targeted(player_num=opponent.player_num),len(field.get_creature_location()[opponent.player_num]),target))
        field.show_field()
        assert False
def get_damage_to_random_creature(field,opponent,virtual,num=0):
    tmp=field.get_creature_location()[opponent.player_num]
    if tmp!=[]:
        target_id=random.choice(tmp)
        get_damage_to_creature(field,opponent,virtual,target_id,num=num)
        if field.card_location[opponent.player_num][target_id].is_in_graveyard==True:
            field.remove_card([opponent.player_num,target_id],virtual)

def destroy_random_creature(field,opponent,virtual):
    tmp=field.get_creature_location()[opponent.player_num]
    if tmp!=[]:
        target_id=random.choice(tmp)
        destroy_opponent_creature(field,opponent,virtual,target_id)


def get_damage_to_player(player,virtual,num=0):
    damage=player.get_damage(num)
    if virtual==False:
        mylogger.info("Player {} get {} damage".format(player.player_num+1,damage))


def restore_player_life(player,virtual,num=0):
    player.restore_player_life(num=num,virtual=virtual)

def draw_cards(player,virtual,num=1):
    tmp=0
    for i in range(num):
        if len(player.deck.deck)>0:
            player.draw(player.deck,1)
            tmp+=1
        else:
            player.lib_out_flg=True
            break

    if virtual==False:
        if tmp==1:
            mylogger.info("Player {} draw a card".format(player.player_num+1))
        elif tmp>1:
            mylogger.info("Player {} draw {} cards".format(player.player_num+1,tmp))

def destroy_opponent_creature(field,opponent,virtual,target):
    tmp=field.get_can_be_targeted(player_num=1-opponent.player_num)
    if tmp!=[]:
        target_id=target
        if target==None :#or target > num-1:
            return
            mylogger.info("targeted_player_num:{}".format(opponent.player_num))
            field.show_field()
            mylogger.info("target is decided at random")
            raise Exception("Debug")
        if 9 not in field.card_location[opponent.player_num][target_id].ability:
            field.remove_card([opponent.player_num,target_id],virtual=virtual)

def get_damage_to_all_creature(field,virtual,num=0):
    for i,side in enumerate(field.get_creature_location()):
        for j in side:
            creature=field.card_location[i][j]
            creature.get_damage(num)

    while True:
        break_flg=True
        continue_flg=False
        for i,side in enumerate(field.card_location):
            for j,creature in enumerate(side):
                if creature.is_in_graveyard==True:
                    field.remove_card([i,j],virtual=virtual)
                    break_flg=False
                    continue_flg=True
                    break
            if continue_flg==True:
                break

        if break_flg==True:
            break

def return_card_to_hand(field,target_location,virtual):
    field.return_card_to_hand(target_location,virtual=virtual)

def search_cards(player,condition,virtual,num=1):
    option_cards_id=[]
    for j,card in enumerate(player.deck.deck):
        if condition(card)==True:
            option_cards_id.append(j)
    search_cards_id=[]
    if len(option_cards_id) >=num:
        search_cards_id=sorted(random.sample(option_cards_id,num),reverse=True)
    elif len(option_cards_id)>0:
        search_cards_id=sorted(option_cards_id,reverse=True)
    
    if len(search_cards_id)>0:
        cards=[]
        for card_id in search_cards_id:
            cards.append(player.deck.deck.pop(card_id))
        if virtual==False:
            for card in cards:
                mylogger.info("Player{} append {} to hand from deck".format(player.player_num+1,card.name))
        player.append_cards_to_hand(cards)

def necromancy(field,player,num=1,virtual=False):
    if field.graveyard.shadows[player.player_num]>=num:
        field.graveyard.shadows[player.player_num]-=num
        if virtual==False:
            mylogger.info("necromancy({}) is actived".format(num))
        return True
    else:
        return False

def reanimate(field,player,virtual,num=1):
    if len(field.card_location[player.player_num])<field.max_field_num:
        choices=[]
        max_cost=-1
        for card_info in field.graveyard.graveyard[player.player_num]:
            if card_info[0]=="Creature":
                card_cost=card_setting.creature_list[card_info[1]][0]
                if card_cost<=num:
                    if max_cost<card_cost:
                        choices=[]
                        max_cost=card_cost
                    if max_cost==card_cost:
                        choices.append(card_info[1])
        if choices!=[]:
            choice_id=random.choice(choices)
            creature_name=card_setting.creature_list[choice_id][-1]
            summon_creature(field,player,virtual,name=creature_name)

def earth_rite(field,player,virtual):
    for i,thing in enumerate(field.card_location[player.player_num]):
        if thing.trait.value == -2:#thing.trait.name == "EARTH_SIGIL" 
            if virtual==False:
                mylogger.info("Earth Rite Active")
            field.remove_card([player.player_num,i],virtual=virtual)
            return True

    return False

def put_card_in_hand(field,player,virtual,name=None,card_category=None,num=1):
    cards=[]
    for i in range(num):
        card=None
        if card_category=="Creature":
            card=card_setting.Creature(card_setting.creature_name_to_id[name])
        elif card_category=="Spell":
            card=card_setting.Spell(card_setting.spell_name_to_id[name])
        elif card_category=="Amulet":
            card=card_setting.Amulet(card_setting.amulet_name_to_id[name])
        cards.append(card)
    if virtual == False:
        mylogger.info("put {} {} to Player {}'s hand".format(num,name,player.player_num+1))
    player.append_cards_to_hand(cards)
    
    return cards

def transform_card_in_field(field,player,target,virtual,name=None,card_category=None):
    if target in field.card_location[player.player_num]:
        card_id=field.card_location[player.player_num].index(target)
        card=None
        if name==None: raise Exception("name is null!")
        if card_category=="Creature":
            card=card_setting.Creature(card_setting.creature_name_to_id[name])
        elif card_category=="Amulet":
            card=card_setting.Amulet(card_setting.amulet_name_to_id[name])
        field.transform_card([player.player_num,card_id],card=card,virtual=False)
    else:
        raise Exception("{} doesn't exist!".format(target))
    

def add_ability_to_creature(field,player,creature,virtual,add_ability=[]):
    for ability in add_ability:
        if ability not in creature.ability:
            creature.ability.append(ability)
    if virtual==False:
        mylogger.info("{} get {}".format(creature.name,[KeywordAbility(i).name for i in add_ability]))

def add_ability_until_end_of_player_turn(field,player,creature,virtual,add_ability=[]):
    def ability_until_end_of_player_turn(field,player,opponent,virtual,target,self_creature):
        if field.turn_player_num==player.player_num:
            for ability in add_ability:
                if ability not in card_setting.creature_list[self_creature.card_id][3] and ability in self_creature.ability:
                    self_creature.ability.remove(ability)
            #self_creature.turn_end_ability=[]
            while True:
                if ability_until_end_of_player_turn in self_creature.turn_end_ability:
                    self_creature.turn_end_ability.remove(ability_until_end_of_player_turn)
                else:
                    break

            
    for ability in add_ability:
        if ability not in creature.ability:
            creature.ability.append(ability)
    if virtual==False:
        mylogger.info("{} get {} until end of player turn".format(creature.name,[KeywordAbility(i).name for i in add_ability]))
    
    creature.turn_end_ability.append(ability_until_end_of_player_turn)


def add_ability_until_end_of_opponent_turn(field,player,creature,virtual,add_ability=[]):
    def ability_until_end_of_opponent_turn(field,player,opponent,virtual,target,self_creature):
        if field.turn_player_num==opponent.player_num:
            for ability in add_ability:
                if ability not in card_setting.creature_list[self_creature.card_id][3] and ability in self_creature.ability:
                    self_creature.ability.remove(ability)
            #self_creature.turn_end_ability=[]
            while True:
                if ability_until_end_of_opponent_turn in self_creature.turn_end_ability:
                    self_creature.turn_end_ability.remove(ability_until_end_of_opponent_turn)
                else:
                    break

            
    for ability in add_ability:
        if ability not in creature.ability:
            creature.ability.append(ability)
    if virtual==False:
        mylogger.info("{} get {} until end of opponent turn".format(creature.name,[KeywordAbility(i).name for i in add_ability]))
    creature.turn_end_ability.append(ability_until_end_of_opponent_turn)
        
def gain_max_pp(field,player,virtual,num=0):
    field.gain_max_pp(player_num=player.player_num,num=num,virtual=virtual)

def restore_pp(field,player,virtual,num=0):
    field.restore_pp(player_num=player.player_num,num=num,virtual=virtual)

def put_card_from_deck_in_play(field,player,virtual,condition=None):
    assert condition!=None
    pop_id_list=[]
    for i,card in enumerate(player.deck.deck):
        if condition(card)==True:
            pop_id_list.append(i)
    if len(pop_id_list)==0:return
    pop_id=random.choice(pop_id_list)
    card_in_play=player.deck.deck.pop(pop_id)
    if virtual==False:
        mylogger.info("recruit {} from deck".format(card_in_play.name))
    field.set_card(card_in_play,player.player_num,virtual=virtual)
    return card_in_play



