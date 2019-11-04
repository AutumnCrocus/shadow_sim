import random
from my_moduler import get_module_logger
import card_setting
mylogger = get_module_logger(__name__)
from util_ability import * 
def spell_ability_001(field,player,opponent,virtual,target,itself):
    get_damage_to_creature(field,opponent,virtual,target,num=3)
    put_card_in_hand(field,player,virtual,name="Earth Essence",card_category="Amulet")


def spell_ability_002(field,player,opponent,virtual,target,itself):
    draw_cards(player,virtual,num=1)

def spell_ability_003(field,player,opponent,virtual,target,itself):
    if virtual==False:
        mylogger.info("All creatures get 2 damage")
    get_damage_to_all_creature(field,virtual,num=2)

def spell_ability_004(field,player,opponent,virtual,target,itself):
    if virtual==False:
        mylogger.info("Player {}'s {} get +2/+2".format(player.player_num+1,field.card_location[player.player_num][target].name))
    creature=field.card_location[player.player_num][target]
    buff_creature(creature,params=[2,2])


def spell_ability_005(field,player,opponent,virtual,target,itself):

    if virtual==False:
        mylogger.info("All creatures are destroyed!")
    for j in range(2):
        side_num=(player.player_num+j)%2
        #side=field.get_creature_location()[side_num]
        side=field.card_location[side_num]
        for thing in side:
            #thing=field.card_location[side_num][i]
            if thing.card_category=="Creature" and KeywordAbility.CANT_BE_DESTROYED_BY_EFFECTS.value not in thing.ability:
                thing.is_in_field=False
                thing.is_in_graveyard=True
    field.check_death(player_num=player.player_num,virtual=virtual)

    """
    while True:
        break_flg=True
        continue_flg=False
        for i,side in enumerate(field.card_location):
            for j,creature in enumerate(side):
                if creature.card_category=="Creature":
                    field.remove_card([i,j],virtual=virtual)
                    break_flg=False
                    continue_flg=True
                    break
            if continue_flg==True:
                break

        if break_flg==True:
            break
    """

def spell_ability_006(field,player,opponent,virtual,target,itself):
    destroy_opponent_creature(field,opponent,virtual,target)
    get_damage_to_player(opponent,virtual,num=2)


def spell_ability_007(field,player,opponent,virtual,target,itself):
    summon_creature(field,player,virtual,name="Pirate")
    summon_creature(field,player,virtual,name="Viking")


def spell_ability_008(field,player,opponent,virtual,target,itself):
    if field.cost[player.player_num]>=7:
        draw_cards(player,virtual,num=1)
    gain_max_pp(field,player,virtual,num=1)

def spell_ability_009(field,player,opponent,virtual,target,itself):
    if target==-1:
        get_damage_to_player(opponent,virtual,num=1)
    else:
        #mylogger.info("target_id:{}".format(target))
        #field.show_field()
        get_damage_to_creature(field,opponent,virtual,target,num=1)
    draw_cards(player,virtual,num=1)

def spell_ability_010(field,player,opponent,virtual,target,itself):
    summon_creature(field,player,virtual,name="Clay Golem")

def spell_ability_011(field,player,opponent,virtual,target,itself):
    get_damage_to_creature(field,opponent,virtual,target,num=itself.spell_boost+1)

def spell_ability_012(field,player,opponent,virtual,target,itself):
    #mylogger.info("2 draw")
    draw_cards(player,virtual,num=2)
def spell_ability_013(field,player,opponent,virtual,target,itself):
    destroy_opponent_creature(field,opponent,virtual,target)

def spell_ability_014(field,player,opponent,virtual,target,itself):
    for i in range(itself.spell_boost+1):
        tmp=field.get_creature_location()[opponent.player_num]
        if tmp!=[]:
            target_id=random.choice(tmp)
            #mylogger.info("target_id:{}".format(target))
            #field.show_field()
            get_damage_to_creature(field,opponent,virtual,target_id,num=1)
        else:
            break

def spell_ability_015(field,player,opponent,virtual,target,itself):
    if target==-1:
        get_damage_to_player(opponent,virtual,num=1)
    else:
        #mylogger.info("target_id:{}".format(target))
        #field.show_field()
        get_damage_to_creature(field,opponent,virtual,target,num=1)

def spell_ability_016(field,player,opponent,virtual,target,itself):
    if virtual==False:
        mylogger.info("Player {} get extra turn".format(player.player_num+1))
    field.ex_turn_count[player.player_num]+=1

def spell_ability_017(field,player,opponent,virtual,target,itself):
    #mylogger.info("target:{}".format(target))
    return_card_to_hand(field,target,virtual)
    draw_cards(player,virtual,num=1)

def spell_ability_018(field,player,opponent,virtual,target,itself):
    get_damage_to_player(player,virtual,num=2)
    if field.check_game_end()==True:
        return
    draw_cards(player,virtual,num=2)

def spell_ability_019(field,player,opponent,virtual,target,itself):
    get_damage_to_player(player,virtual,num=2)
    if field.check_game_end()==True:
        return
    if target==-1:
        get_damage_to_player(opponent,virtual,num=3)
    else:
        get_damage_to_creature(field,opponent,virtual,target,num=3)

def spell_ability_020(field,player,opponent,virtual,target,itself):
    get_damage_to_creature(field,opponent,virtual,target,num=4)
    restore_player_life(player,virtual,num=2)

def spell_ability_021(field,player,opponent,virtual,target,itself):
    get_damage_to_all_creature(field,virtual,num=8)

def spell_ability_022(field,player,opponent,virtual,target,itself):
    if necromancy(field,player,num=2)==True:
        get_damage_to_creature(field,opponent,virtual,target,num=4)
    else:
        get_damage_to_creature(field,opponent,virtual,target,num=2)

def spell_ability_023(field,player,opponent,virtual,target,itself):
    while len(field.card_location[player.player_num])<field.max_field_num:
        if necromancy(field,player,num=1)==True:
            summon_creature(field,player,virtual,name="Ghost")
        else:
            break

def spell_ability_024(field,player,opponent,virtual,target,itself):
    summon_creature(field,player,virtual,name="Zombie",num=3)
    if necromancy(field,player,num=6)==True:
        if virtual==False:
            mylogger.info("Give +0/+1 and Ward to allied Zombies.")
        for creature_id in field.get_creature_location()[player.player_num]:
            creature=field.card_location[player.player_num][creature_id]
            #mylogger.info("{} {}:{}".format(creature_id,creature.name,creature.ability))
            if creature.name=="Zombie":
                buff_creature(creature,params=[0,1])
                if 3 not in creature.ability:
                    creature.ability.append(3)
                #mylogger.info("ability:{}".format(creature.ability))

def spell_ability_025(field,player,opponent,virtual,target,itself):
    if field.card_location[opponent.player_num][target].get_current_toughness()>3:
        raise Exception("Illegal Target!")

    field.banish_card([opponent.player_num,target],virtual=virtual)

def spell_ability_026(field,player,opponent,virtual,target,itself):
    return_card_to_hand(field,[player.player_num,target],virtual)
    draw_cards(player,virtual)

def spell_ability_027(field,player,opponent,virtual,target,itself):
    put_card_in_hand(field,player,virtual,name="Fairy",card_category="Creature",num=2)

def spell_ability_028(field,player,opponent,virtual,target,itself):
    return_card_to_hand(field,[player.player_num,target[0]],virtual)
    get_damage_to_creature(field,opponent,virtual,target[1],num=3)

def spell_ability_029(field,player,opponent,virtual,target,itself):
    get_damage_to_creature(field,opponent,virtual,target,num=2)
    put_card_in_hand(field,player,virtual,name="Fairy",card_category="Creature")

def spell_ability_030(field,player,opponent,virtual,target,itself):
    return_card_to_hand(field,[player.player_num,target],virtual)
    choices=field.get_creature_location()[opponent.player_num]
    if choices==[]:
        return
    target_id=random.choice(choices)
    return_card_to_hand(field,[opponent.player_num,target_id],virtual)

def spell_ability_031(field,player,opponent,virtual,target,itself):
    for i in range(len(player.hand)):
        get_damage_to_random_creature(field,opponent,virtual,num=1)

def spell_ability_032(field,player,opponent,virtual,target,itself):
    get_damage_to_creature(field,opponent,virtual,target,num=2)

def spell_ability_033(field,player,opponent,virtual,target,itself):
    get_damage_to_creature(field,opponent,virtual,target,num=3)
    if field.card_location[opponent.player_num][target].is_in_field==False:
        field.remove_card([opponent.player_num,target],virtual=virtual)
    if itself.active_enhance_code[0]==True:
        for creature_id in field.get_creature_location()[opponent.player_num]:
            get_damage_to_creature(field,opponent,virtual,creature_id,num=2)

def spell_ability_034(field,player,opponent,virtual,target,itself):
    gain_max_pp(field,player,virtual,num=1)
    draw_cards(player,virtual,num=2)
    restore_player_life(player,virtual,num=3)

def spell_ability_035(field,player,opponent,virtual,target,itself):
    if itself.active_enhance_code[0]==False:
        field.banish_card([opponent.player_num,target],virtual=virtual)
    else:
        index=0
        while index<len(field.card_location[opponent.player_num]):
            before=len(field.card_location[opponent.player_num])
            field.banish_card([opponent.player_num,index],virtual=virtual)
            after=len(field.card_location[opponent.player_num])
            if after==before:
                index+=1
            
    
def spell_ability_036(field,player,opponent,virtual,target,itself):
    if itself.active_enhance_code[0]==False:
        if field.card_location[opponent.player_num][target].origin_cost>2:
            player.show_hand()
            field.show_field()
            raise Exception("Over Cost Error:target_name={}"\
                .format(field.card_location[opponent.player_num][target].name))

    field.remove_card([opponent.player_num,target],virtual=virtual)


def spell_ability_037(field,player,opponent,virtual,target,itself):
    """
    Summon a Club Soldier, a Heart Guardian, and a Spade Raider.
    """
    summon_creature(field,player,virtual,name="Crab Soldier")
    summon_creature(field,player,virtual,name="Heart Guardian")
    summon_creature(field,player,virtual,name="Spade Raider")

def spell_ability_038(field,player,opponent,virtual,target,itself):
    """
    Put 2 random Commanders that cost 3 play points from your deck into play.
    """
    condition=lambda card:card.card_category=="Creature" and card.trait.value==Trait["COMMANDER"].value and card.origin_cost==3
    put_card_from_deck_in_play(field,player,virtual,condition=condition)
    put_card_from_deck_in_play(field,player,virtual,condition=condition)

def spell_ability_039(field,player,opponent,virtual,target,itself):
    """
    Destroy an allied follower.
    Draw 2 cards.
    """
    destroy_opponent_creature(field,player,virtual,target)
    draw_cards(player,virtual,num=2)





def token_spell_ability_001(field,player,opponent,virtual,target,itself):
    if target==-1:
        get_damage_to_player(player,virtual,num=3)
    else:
        get_damage_to_creature(field,opponent,virtual,target,num=3)
    if field.check_game_end()==True:
        return
    if earth_rite(field,player,virtual)==True:
        draw_cards(player,virtual,num=1)

def token_spell_ability_002(field,player,opponent,virtual,target,itself):
    num=0
    for card in field.card_location[player.player_num]:
        if card.trait.value == Trait["EARTH_SIGIL"].value:
            num+=1
    for i in range(num):
        earth_rite(field,player,virtual)
        tmp=random.random()
        if tmp < 1/3:
            summon_creature(field,player,virtual,name="Clay Golem")
        elif tmp < 2/3:
            put_card_in_hand(field,player,virtual,name="Earth Essence",card_category="Amulet")
        else:
            get_damage_to_player(player,virtual,num=2)
        
        if field.check_game_end()==True:
            break




spell_ability_dict={1:spell_ability_001,2:spell_ability_002,3:spell_ability_003,4:spell_ability_004,5:spell_ability_005,\
    6:spell_ability_006,7:spell_ability_007,8:spell_ability_008,9:spell_ability_009,10:spell_ability_010,\
    11:spell_ability_011,12:spell_ability_012,13:spell_ability_013,14:spell_ability_014,15:spell_ability_015,\
    16:spell_ability_016,17:spell_ability_017,18:spell_ability_018,19:spell_ability_019,20:spell_ability_020,\
    21:spell_ability_021,22:spell_ability_022,23:spell_ability_023,24:spell_ability_024,25:spell_ability_025,\
    26:spell_ability_026,27:spell_ability_027,28:spell_ability_028,29:spell_ability_029,30:spell_ability_030,\
    31:spell_ability_031,32:spell_ability_032,33:spell_ability_033,34:spell_ability_034,35:spell_ability_035,\
    36:spell_ability_036,37:spell_ability_037,38:spell_ability_038,39:spell_ability_039,\
    
    
    -1:token_spell_ability_001,-2:token_spell_ability_002}