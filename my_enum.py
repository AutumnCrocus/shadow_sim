from enum import Enum
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
    CONTROL = 3
    COMBO = 4

class KeywordAbility(Enum):
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

class State_Code(Enum):
    PLAY = 1
    SET = 2
    DESTROYED = 3
    EVOLVE = 4
    ATTACK_TO_FOLLOWER = 5
    ATTACK_TO_PLAYER = 6
    RESTORE_PLAYER_LIFE = 7