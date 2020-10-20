# -*- coding: utf-8 -*-
from enum import Enum
class __LeaderClass(Enum):
    NEUTRAL = 0
    FOREST = 1
    SWORD = 2
    RUNE = 3
    DRAGON = 4
    SHADOW = 5
    BLOOD = 6
    HAVEN = 7
    PORTAL = 8

LeaderClass=__LeaderClass

class __Trait(Enum):
    EARTH_SIGIL = -2
    NONE = -1
    OFFICER = 0
    COMMANDER = 1
    ARTIFACT = 2

Trait=__Trait

class __DeckType(Enum):
    AGGRO = 1
    MID = 2
    CONTROL = 3
    COMBO = 4

DeckType=__DeckType

class __KeywordAbility(Enum):
    STORM = 1
    BANE = 2
    WARD = 3
    RUSH = 4
    CANT_BE_ATTACKED = 5
    CANT_BE_TARGETED = 6
    AMBUSH = 7
    DRAIN = 8
    CANT_BE_DESTROYED_BY_EFFECTS = 9
    REDUCE_DAMAGE_TO_ZERO = 10
    REDUCE_DAMAGE_TO_ZERO_BY_EFFECTS = 11
    CANT_ATTACK_TO_FOLLOWER = 12
    CANT_ATTACK_TO_PLAYER = 13
    CANT_ATTACK = 14
    BANISH_WHEN_LEAVES = 15
#1は速攻(出たターンでも攻撃できる),2は必殺(交戦したクリーチャーを必ず破壊する)
#3は守護(相手の場に守護を持つフォロワーがいる限り、原則このフォロワー以外には攻撃できない)
#4は突進(出たターンでもフォロワーに攻撃できる)
#5は攻撃されない
#6は能力の対象にならない
#7は潜伏(攻撃されず、能力の対象にならない,攻撃すると解除される)
#8はドレイン(攻撃によって与えたダメージ分プレイヤーのライフを回復)
#９は効果で破壊不能
#10はダメージを受けない
#11は能力によるダメージを受けない

KeywordAbility=__KeywordAbility

class __State_Code(Enum):
    PLAY = 1
    SET = 2
    DESTROYED = 3
    EVOLVE = 4
    ATTACK_TO_FOLLOWER = 5
    ATTACK_TO_PLAYER = 6
    RESTORE_PLAYER_LIFE = 7
    RESTORE_FOLLOWER_TOUGHNESS = 8
    START_OF_TURN = 9
    END_OF_TURN = 10
    GET_DAMAGE = 11

State_Code=__State_Code

class __Action_Code(Enum):
    ERROR = -10
    TURN_END = 0
    PLAY_CARD = 1
    ATTACK_TO_FOLLOWER = 2
    ATTACK_TO_PLAYER = 3
    EVOLVE = 4

Action_Code=__Action_Code

class __Active_Ability_Check_Code(Enum):
    OVERFLOW = 0
    VENGEANCE = 1
    RESONANCE = 2

    BAHAMUT = 3

Active_Ability_Check_Code=__Active_Ability_Check_Code

class __Target_Type(Enum):
    ENEMY_FOLLOWER = 1
    ALLIED_FOLLOWER = 2
    ENEMY = 3
    FOLLOWER = 4
    CARD = 5
    ALLIED_CARD = 6
    ALLIED_CARD_AND_ENEMY_FOLLOWER = 7
    CARD_IN_HAND = 8
    ENEMY_CARD = 9
    ALLY = 10
    ALLIED_AMULET = 11
    ENEMY_AMULET = 12

Target_Type = __Target_Type

class __Card_Category(Enum):
    NONE = 0
    Creature = 1
    Spell = 2
    Amulet = 3

Card_Category = __Card_Category
