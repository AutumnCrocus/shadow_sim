
import random
from my_moduler import get_module_logger
import card_setting
mylogger = get_module_logger(__name__)
from util_ability import * 
from my_enum import *

def creature_ability_001(field,player,opponent,virtual,target,itself):
    restore_player_life(player,virtual,num=2)

def creature_ability_002(field,player,opponent,virtual,target,itself):
    if itself.active_enhance_code[0]==True:
        get_damage_to_player(opponent,virtual,num=3)

def creature_ability_003(field,player,opponent,virtual,target,itself):
    draw_cards(player,virtual,num=1)

def creature_ability_004(field,player,opponent,virtual,target,itself):
    if target!=None and field.get_can_be_targeted(player_num=player.player_num)!=[]:
        get_damage_to_creature(field,opponent,virtual,target,num=1)

def creature_ability_005(field,player,opponent,virtual,target,itself):
    if field.get_can_be_targeted(player_num=player.player_num)!=[]:
        if  target==None:
            raise Exception()
        get_damage_to_creature(field,opponent,virtual,target,num=4)

def creature_ability_006(field,player,opponent,virtual,target,itself):

    if virtual==False:
        mylogger.info("All other creatures  are destroyed")

    for j in range(2):
        side_num=(player.player_num+j)%2
        #side=field.get_creature_location()[side_num]
        side=field.card_location[side_num]
        for thing in side:
            #thing=field.card_location[side_num][i]
            if thing!=itself and KeywordAbility.CANT_BE_DESTROYED_BY_EFFECTS.value not in thing.ability:
                thing.is_in_field=False
                thing.is_in_graveyard=True
    field.check_death(player_num=player.player_num,virtual=virtual)


def creature_ability_007(field,player,opponent,virtual,target,itself):

    if virtual==False:
        mylogger.info("Give +1/+1 to all other allied creatures")

    for creature in field.card_location[player.player_num]:
        if itself!=creature and creature.card_category=="Creature":
            buff_creature(creature,params=[1,1])


def creature_ability_008(field,player,opponent,virtual,target,itself):
    """
    Summon Otohime's Bodyguards until your area is full
    """
    summon_creature(field,player,virtual,name="Otohime's Bodyguard",num=4)


def creature_ability_009(field,player,opponent,virtual,target,itself):
    """
    Gain +1/+0 for each enemy follower in play. If there are at least 3,
    gain resistance to targeted enemy spells and effects.
    """

    num=len(field.card_location[opponent.player_num])
    #itself.power+=num
    #itself.buff[0]+=num
    buff_creature(itself,params=[num,0])
    if num>=3:
        itself.can_not_be_targeted=True
        itself.ability.append(KeywordAbility.CANT_BE_TARGETED.value)
        if virtual==False:
            mylogger.info("{} get +{}/0 and can't be targeted".format(itself.name,num))
    else:
        if virtual==False:
            mylogger.info("{} get +{}/0".format(itself.name,num))

def creature_ability_010(field,player,opponent,virtual,target,itself):

    summon_creature(field,player,virtual,name="Steelclad Knight")


def creature_ability_011(field,player,opponent,virtual,target,itself):
    """
    Summon Knight
    """
    summon_creature(field,player,virtual,name="Knight")


def creature_ability_012(field,player,opponent,virtual,target,itself):
    summon_creature(field,player,virtual,name="Steelclad Knight")
    summon_creature(field,player,virtual,name="Knight")

def creature_ability_013(field,player,opponent,virtual,target,itself):
    #Lucifer's end-of-turn ability
    if itself.evolved==False:
        restore_player_life(player,virtual,num=4)
    
    else:
        get_damage_to_player(opponent,virtual,num=4)


def creature_ability_014(field,player,opponent,virtual,target,itself):
    if field.get_can_be_targeted(player_num=player.player_num)!=[]:
        get_damage_to_creature(field,opponent,virtual,target,num=3)


def creature_ability_015(field,player,opponent,virtual,target,itself):
    gain_max_pp(field,player,num=1,virtual=virtual)

def creature_ability_016(field,player,opponent,virtual,target,itself):
    buff_creature(itself,params=[itself.spellboost,itself.spellboost])
    if virtual==False:
        mylogger.info("{[0]} get +{[1]}/{[1]}".format(itself.name,itself.spellboost))

def creature_ability_017(field,player,opponent,virtual,target,itself):
    get_damage_to_player(player,virtual,num=1)
    get_damage_to_player(opponent,virtual,num=1)

def creature_ability_018(field,player,opponent,virtual,target,itself):
    if player.check_vengeance()==True:
        buff_creature(itself,params=[1,1])

def creature_ability_019(field,player,opponent,virtual,target,itself):
    get_damage_to_player(player,virtual,num=2)


def creature_ability_020(field,player,opponent,virtual,target,itself):
    if player.check_vengeance()==True:
        itself.ability.append(KeywordAbility.STORM.value)

def creature_ability_021(field,player,opponent,virtual,target,itself):
    #Wardrobe Raider's evolve ability
    if field.get_can_be_targeted(player_num=player.player_num)!=[] and target!=None:
        get_damage_to_creature(field,opponent,virtual,target,num=2)
    restore_player_life(player,virtual,num=2)

def creature_ability_022(field,player,opponent,virtual,target,itself):
    if player.check_vengeance()==False:
        get_damage_to_player(player,virtual,num=2)

def creature_ability_023(field,player,opponent,virtual,target,itself):
    if target==-1:
        get_damage_to_player(opponent,virtual,num=5)
    elif field.get_can_be_targeted(player_num=player.player_num)!=[]:
        get_damage_to_creature(field,opponent,virtual,target,num=5)
    restore_player_life(player,virtual,num=5)

def creature_ability_024(field,player,opponent,virtual,target,itself):
    if player.check_vengeance()==True:
        add_ability_to_creature(field,player,itself,virtual,\
            add_ability=[KeywordAbility.BANE.value,KeywordAbility.DRAIN.value])

        if virtual==False:
            mylogger.info("{} get bane and drain".format(itself.name))

def creature_ability_025(field,player,opponent,virtual,target,itself):
    condition=lambda card :card.trait.name=="COMMANDER"
    search_cards(player,condition,virtual,num=1)

def creature_ability_026(field,player,opponent,virtual,target,itself):
    if target!=None:
        if itself.target_regulation(field.card_location[opponent.player_num][target]):
            destroy_opponent_creature(field,opponent,virtual,target)

def creature_ability_027(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Give +1/+0 to a 1-play point allied follower.
    """
    if target!=None:
        target_creature=field.card_location[player.player_num][target]
        if  itself.target_regulation(target_creature)==False:
            mylogger.info("target_id:{}".format(target))
            field.show_field()
            assert False
        if virtual==False:
            mylogger.info("{} get +1/0".format(target_creature.name))
        buff_creature(target_creature,params=[1,0])

def creature_ability_028(field,player,opponent,virtual,target,itself):
    destroy_random_creature(field,opponent,virtual)

def creature_ability_029(field,player,opponent,virtual,target,itself):
    if target!=None:
        destroy_opponent_creature(field,player,virtual,target)
        draw_cards(player,virtual,num=2)

def creature_ability_030(field,player,opponent,virtual,target,itself):
    if necromancy(field,player,num=1)==True:
        if virtual==False:
            mylogger.info("{} get +1/+1".format(itself.name))
        buff_creature(itself,params=[1,1])

def creature_ability_031(field,player,opponent,virtual,target,itself):
    #Ghost's end-of-turn  and removed-from-field-ability
    flg = itself in field.card_location[player.player_num]
    if flg==False:
        mylogger.info("Player_num:{}".format(player.player_num))
        field.show_field()
        raise Exception("Error")
    location=[player.player_num,field.card_location[player.player_num].index(itself)]
    field.banish_card(location,virtual=virtual)

def creature_ability_032(field,player,opponent,virtual,target,itself):
    #Andrealphus's evolve ability
    for card in player.hand:
        if card.card_category=="Creature":
            card.lastword_ability.append(creature_ability_003)

def creature_ability_033(field,player,opponent,virtual,target,itself):
    get_damage_to_random_creature(field,opponent,virtual,num=1)

def creature_ability_034(field,player,opponent,virtual,target,itself):
    #Lady Grey's evolve ability
    reanimate(field,player,virtual,num=2)    

def creature_ability_035(field,player,opponent,virtual,target,itself):
    get_damage_to_player(opponent,virtual,num=2) 

def creature_ability_036(field,player,opponent,virtual,target,itself):
    summon_creature(field,player,virtual,name="Mordecai the Duelist") 

def creature_ability_037(field,player,opponent,virtual,target,itself):
    power_list=[]
    max_power=-1
    for creature_id in field.get_creature_location()[opponent.player_num]:
        if field.card_location[opponent.player_num][creature_id].power>max_power:
            max_power=field.card_location[opponent.player_num][creature_id].power
            power_list=[]
        if max_power==field.card_location[opponent.player_num][creature_id].power:
            power_list.append(creature_id)
    if power_list!=[]:
        target_id=random.choice(power_list)
        destroy_opponent_creature(field,opponent,virtual,target_id)
        restore_player_life(player,virtual,num=max_power)

def creature_ability_038(field,player,opponent,virtual,target,itself):
    while len(field.card_location[player.player_num])<field.max_field_num:
        if necromancy(field,player,num=3)==True:
            summon_creature(field,player,virtual,name="Zombie")
        else:
            break
    if virtual==False:
        mylogger.info("Give all other allied followers +2/+0 and Rush until the end of the turn")
    for thing in field.card_location[player.player_num]:
        if thing.card_category=="Creature" and thing !=itself:
            buff_creature_until_end_of_turn(thing,params=[2,0])
            if KeywordAbility.RUSH.value not in thing.ability:
                add_temporary_ability_to_creature(field,player,thing,virtual,add_ability=[KeywordAbility.RUSH.value])
            #    thing.ability.append(4)
            #thing.turn_end_ability.append(ability_until_end_of_turn)
        
def creature_ability_039(field,player,opponent,virtual,target,itself):
    field.graveyard.shadows[player.player_num]+=1
    if virtual==False:
        mylogger.info("Gain 1 shadow")

def creature_ability_040(field,player,opponent,virtual,target,itself):
    restore_player_life(player,virtual,num=3)
    draw_cards(player,virtual,num=1)

def creature_ability_041(field,player,opponent,virtual,target,itself):
    restore_player_life(player,virtual,num=2)

def creature_ability_042(field,player,opponent,virtual,target,itself):
    set_amulet(field,player,virtual,name="Elana's Prayer")

def creature_ability_043(field,player,opponent,virtual,target,itself):
    put_card_in_hand(field,player,virtual,name="Earth Essence",card_category="Amulet")

def creature_ability_044(field,player,opponent,virtual,target,itself):
    condition=lambda card :card.is_earth_rite==True
    search_cards(player,condition,virtual,num=1)

def creature_ability_045(field,player,opponent,virtual,target,itself):
    if earth_rite(field,player,virtual)==True:
        summon_creature(field,player,virtual,name="Magic Illusionist")

def creature_ability_046(field,player,opponent,virtual,target,itself):
    if earth_rite(field,player,virtual)==True:
        buff_creature(itself,params=[1,1])
        add_ability_to_creature(field,player,itself,virtual,add_ability=[7])

def creature_ability_047(field,player,opponent,virtual,target,itself):
    if earth_rite(field,player,virtual)==True:
        if target==-1:
            get_damage_to_player(opponent,virtual,num=3)
        elif target in field.get_can_be_targeted(player_num=player.player_num):
            get_damage_to_creature(field,opponent,virtual,target,num=3)

def creature_ability_048(field,player,opponent,virtual,target,itself):
    if earth_rite(field,player,virtual)==False:
        if target==-1:
            get_damage_to_player(opponent,virtual,num=3)
        elif field.get_can_be_targeted(player_num=player.player_num)!=[]:
            get_damage_to_creature(field,opponent,virtual,target,num=3)
    else:
        get_damage_to_player(opponent,virtual,num=3)
        card_id=0
        while card_id < len(field.card_location[opponent.player_num]):
            if field.card_location[opponent.player_num][card_id].card_category=="Creature":
                before=len(field.card_location[opponent.player_num])
                get_damage_to_creature(field,opponent,virtual,card_id,num=3)
                after=before=len(field.card_location[opponent.player_num])
                card_id+=int(before==after)
            else:
                card_id+=1

def creature_ability_049(field,player,opponent,virtual,target,itself):
    count=0
    while True:
        if earth_rite(field,player,virtual)==True:
            count+=1
        else:
            break
    if count>=1:
        summon_creature(field,player,virtual,name="Clay Golem")
    if count>=2:
        get_damage_to_random_creature(field,opponent,virtual,num=3)
    if count>=3:
        add_ability_to_creature(field,player,itself,virtual,add_ability=[1])
    if count>=4:
        buff_creature(itself,params=[3,3])

def creature_ability_050(field,player,opponent,virtual,target,itself):
    put_card_in_hand(field,player,virtual,name="Fairy",card_category="Creature")

def creature_ability_051(field,player,opponent,virtual,target,itself):
    put_card_in_hand(field,player,virtual,name="Fairy",card_category="Creature",num=2)

def creature_ability_052(field,player,opponent,virtual,target,itself):
    buff_creature_until_end_of_turn(itself,params=[field.players_play_num,0])
def creature_ability_053(field,player,opponent,virtual,target,itself):
    i=0
    count=0
    while True:
        target_object=field.card_location[player.player_num][i]
        if i == len(field.card_location[player.player_num]) or target_object==itself:
            break
        if target_object.card_category=="Creature":
            return_card_to_hand(field,[player.player_num,i],virtual)
            count+=1
        else:
            i+=1

    if count>0:
        if virtual==False:
            mylogger.info("{0} get +{1}/+{1}".format(itself.name,count))
        buff_creature(itself,params=[count,count])

def creature_ability_054(field,player,opponent,virtual,target,itself):
    length=len(player.hand)
    faires=put_card_in_hand(field,player,virtual,name="Fairy",card_category="Creature",num=2)
    faires[0].cost=0
    faires[1].cost=0
    """
    if length<9:
        player.hand[-1].cost=0
        if length<8:
            player.hand[-2].cost=0
    """
        
def creature_ability_055(field,player,opponent,virtual,target,itself):
    if field.players_play_num>=2 and field.get_can_be_targeted(player_num=player.player_num)!=[]:
        get_damage_to_creature(field,opponent,virtual,target,num=2)

def creature_ability_056(field,player,opponent,virtual,target,itself):
    creature=summon_creature(field,player,virtual,name="Crystalia Eve")
    if field.players_play_num>=2 and creature!=None:
        field.auto_evolve(creature,virtual=virtual)
        add_ability_to_creature(field,player,creature,virtual,add_ability=[3])

def creature_ability_057(field,player,opponent,virtual,target,itself):
    if player.check_overflow()==True:
        add_ability_to_creature(field,player,itself,virtual,add_ability=[1])
#       
def creature_ability_058(field,player,opponent,virtual,target,itself):
    if player.check_overflow()==True and target!=None:
        target_creature=field.card_location[player.player_num][target]
        add_ability_to_creature(field,player,target_creature,virtual,add_ability=[1])

def creature_ability_059(field,player,opponent,virtual,target,itself):
    if field.current_turn[player.player_num]>=5:
        gain_max_pp(field,player,num=1,virtual=virtual)
        

def creature_ability_060(field,player,opponent,virtual,target,itself):
    if player.check_overflow()==True:
        restore_player_life(player,virtual,num=3)
        

def creature_ability_061(field,player,opponent,virtual,target,itself):
    if target==-1:
        get_damage_to_player(opponent,virtual,num=3)
    elif field.get_can_be_targeted(player_num=player.player_num!=[]):
        get_damage_to_creature(field,opponent,virtual,target,num=3)

def creature_ability_062(field,player,opponent,virtual,target,itself):
    restore_player_life(player,virtual,num=3)
    put_card_in_hand(field,player,virtual,name="Ouroboros",card_category="Creature",num=1)

def creature_ability_063(field,player,opponent,virtual,target,itself):
    num=field.remain_cost[player.player_num]
    if virtual==False:
        mylogger.info("{} get +{}/0".format(itself.name,num))
    buff_creature(itself,params=[num,0])

def creature_ability_064(field,player,opponent,virtual,target,itself):
    if itself.active_enhance_code[0]==True:
        count=0
        while count<4:
            get_damage_to_player(opponent,virtual,num=1)
            for creature_id in field.get_creature_location()[opponent.player_num]:
                get_damage_to_creature(field,opponent,virtual,creature_id,num=1)
            if field.check_game_end()==True:
                return
            field.check_death(player_num=player.player_num,virtual=virtual)
            count+=1
def creature_ability_065(field,player,opponent,virtual,target,itself):
    if type(target)==int:
        creature=player.hand.pop(target)
        assert creature.card_category=="Creature","target_id:{}".format(target)
        if virtual==False:
            mylogger.info("Player{} put {} to field from hand".format(player.player_num+1,creature.name))
        field.set_card(creature,player.player_num,virtual=virtual)
        add_ability_to_creature(field,player,creature,virtual,add_ability=[4])
        creature.turn_end_ability.append(creature_ability_066)
    
def creature_ability_066(field,player,opponent,virtual,target,itself):
    target_location=[player.player_num,field.card_location[player.player_num].index(itself)]
    return_card_to_hand(field,target_location,virtual)

def creature_ability_067(field,player,opponent,virtual,target,itself):
    restore_player_life(player,virtual,num=4)

def creature_ability_068(field,player,opponent,virtual,target,itself):
    if itself.active_enhance_code[0]==True:
        buff_creature(itself,params=[3,3])

def creature_ability_069(field,player,opponent,virtual,target,itself):
    set_amulet(field,player,virtual,name="Earth Essence",num=2)
    new_cards=put_card_in_hand(field,player,virtual,name="Orichalcum Golem",card_category="Creature",num=1)
    new_cards[0].cost=7
    if virtual == False:
        mylogger.info("{}'s cost is changed to 7".format(itself.name))

def creature_ability_070(field,player,opponent,virtual,target,itself):
    """
    Evolve: Gain +1/+0 and Ward if there are at least 4 allied Swordcraft followers in play.
    """
    count=0
    for card in field.card_location[player.player_num]:
        if card.card_class.value==LeaderClass.SWORD.value:
            count+=1
    if count>=4:
        buff_creature(itself,params=[1,0])
        add_ability_to_creature(field,player,itself,virtual,add_ability=[KeywordAbility.WARD.value])

def creature_ability_071(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Gain Storm if an allied Commander card is in play.
    """
    for card in field.card_location[player.player_num]:
        if card.trait.value==Trait.COMMANDER.value:
            add_ability_to_creature(field,player,itself,virtual,add_ability=[KeywordAbility.STORM.value])
            return
            

def creature_ability_072(field,player,opponent,virtual,target,itself):
    """
    Evolve: Destroy an enemy follower if an allied Neutral follower is in play.
    """
    if target!=None:return
    for card in field.card_location[player.player_num]:
        if card.card_class.value==LeaderClass.NEUTRAL.value:
            #mylogger.info("hit")
            destroy_opponent_creature(field,opponent,virtual,target)
            return


def creature_ability_073(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Enhance (9) - Can attack 2 times per turn. Reduce damage to 0 until the end of the turn.
    """
    if itself.active_enhance_code[0]==True:
        itself.can_attack_num=2
        if virtual==False:
            mylogger.info("{} get 'can attack 2 times per turn' ".format(itself.name))
        add_temporary_ability_to_creature(field,player,itself,virtual,add_ability=[KeywordAbility.REDUCE_DAMAGE_TO_ZERO.value])


def creature_ability_074(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Summon a Kunoichi Trainee.
    """
    summon_creature(field,player,virtual,name="Kunoichi Trainee",num=1)

def creature_ability_075(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Draw a card if Overflow is active for you.
    """
    if player.check_overflow()==True:
        draw_cards(player,virtual,num=1)
        
    

def creature_ability_076(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Put a random Dragoncraft follower from your deck into your hand.
    """
    condition=lambda card :card.card_class.name == "DRAGON" and card.card_category=="Creature"
    search_cards(player,condition,virtual,num=1)


def creature_ability_077(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Put a Fire Lizard into your hand.
    """
    put_card_in_hand(field,player,virtual,name="Fire Lizard",card_category="Creature")

def creature_ability_078(field,player,opponent,virtual,target,itself):
    """
    Fanfare: Destroy a follower, and then put a copy of that follower into play.
    """
    if target==None: return
    card_index=target
    target_creature=field.card_location[card_index[0]][card_index[1]]
    card_name=target_creature.name
    destroy_opponent_creature(field,field.players[card_index[0]],virtual,card_index[1])
    #ability_resolution(self,virtual=False,player_num=0)
    summon_creature(field,field.players[card_index[0]],virtual,name=card_name)

    



creature_ability_dict={0:None,1:creature_ability_001,2:creature_ability_002,3:creature_ability_003,\
    4:creature_ability_004,5:creature_ability_005,6:creature_ability_006,7:creature_ability_007,8:creature_ability_008,
    9:creature_ability_009,10:creature_ability_010,11:creature_ability_011,12:creature_ability_012,13:creature_ability_013,
    14:creature_ability_014,15:creature_ability_015,16:creature_ability_016,17:creature_ability_017,18:creature_ability_018,
    19:creature_ability_019,20:creature_ability_020,21:creature_ability_021,22:creature_ability_022,23:creature_ability_023,
    24:creature_ability_024,25:creature_ability_025,26:creature_ability_026,27:creature_ability_027,28:creature_ability_028,
    29:creature_ability_029,30:creature_ability_030,31:creature_ability_031,32:creature_ability_032,33:creature_ability_033,
    34:creature_ability_034,35:creature_ability_035,36:creature_ability_036,37:creature_ability_037,38:creature_ability_038,
    39:creature_ability_039,40:creature_ability_040,41:creature_ability_041,42:creature_ability_042,43:creature_ability_043,
    44:creature_ability_044,45:creature_ability_045,46:creature_ability_046,47:creature_ability_047,48:creature_ability_048,
    49:creature_ability_049,50:creature_ability_050,51:creature_ability_051,52:creature_ability_052,53:creature_ability_053,
    54:creature_ability_054,55:creature_ability_055,56:creature_ability_056,57:creature_ability_057,58:creature_ability_058,
    59:creature_ability_059,60:creature_ability_060,61:creature_ability_061,62:creature_ability_062,63:creature_ability_063,
    64:creature_ability_064,65:creature_ability_065,66:creature_ability_066,67:creature_ability_067,68:creature_ability_068,\
    69:creature_ability_069,70:creature_ability_070,71:creature_ability_071,72:creature_ability_072,73:creature_ability_073,\
    74:creature_ability_074,75:creature_ability_075,76:creature_ability_076,77:creature_ability_077,78:creature_ability_078}

