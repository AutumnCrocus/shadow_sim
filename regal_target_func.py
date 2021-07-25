from my_enum import *
always_true_func = lambda target:True
TURN_PLAYER_ID = 1
NON_TURN_PLAYER_ID = 0
def decide_target(field,card,player_num,evo=False,target_category=None,target_regulation=None,with_ids=False):
    player = field.players[player_num]
    player_side = field.card_location[player_num]
    can_be_targeted = field.get_can_be_targeted(player_num=player_num)
    opponent_side = field.card_location[1 - player_num]
    player_side_followers = field.get_creature_location()[player_num]
    if target_regulation == None:
        target_regulation = always_true_func
    if target_category == Target_Type.ENEMY_FOLLOWER.value:
        if target_category == Target_Type.ENEMY_FOLLOWER.value:
            regal_targets = [card_id for card_id in can_be_targeted if
                             target_regulation(opponent_side[card_id], card)]
            if with_ids:
                target_card_ids = [(player_side[location_id].name, TURN_PLAYER_ID) for location_id in regal_targets]
    elif target_category == Target_Type.ALLIED_FOLLOWER.value:
        regal_targets = [card_id for card_id in player_side_followers if
                         target_regulation(player_side[card_id],card)]
        if with_ids:
            target_card_ids = [(player_side[location_id].name, NON_TURN_PLAYER_ID) for location_id in regal_targets]
    elif target_category == Target_Type.ENEMY.value:
        regal_targets = [-1] + [card_id for card_id in can_be_targeted if
                                target_regulation(opponent_side[card_id], card)]
        if with_ids:
            target_card_ids = [(0, TURN_PLAYER_ID)]
            if len(regal_targets) > 1:
                target_card_ids += [(opponent_side[location_id].name, TURN_PLAYER_ID) for location_id in
                                    regal_targets[1:]]
    if with_ids:
        return regal_targets,target_card_ids
    return regal_targets