import random
import card_setting
from my_moduler import get_module_logger
mylogger = get_module_logger(__name__)
from my_enum import *
PRIORITY_MAX_LIMIT=10
class reduce_damage_to_zero():
    def __init__(self):
        self.priority=PRIORITY_MAX_LIMIT
    def __call__(argument=None,state_code=None):
        if state_code[0]==State_Code.GET_DAMAGE.value and and type(argument)==int:
            return 0
        else:
            return argument

class can_not_take_more_than_x_damage():
    def __init__(self,num=0):
        self.priority=PRIORITY_MAX_LIMIT-1
        self.damage_upper_bound=num
    def __call__(self,argument=None,state_code=None):
        if state_code[0]==State_Code.GET_DAMAGE.value and type(argument)==int:
            return (argument,self.damage_upper_bound)[argument>self.damage_upper_bound]
        else:
            return argument



