# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:39:19 2019

@author: NickovicD
"""
from enum import Enum

class StlComparisonOperator(Enum):
    LESS = 0
    LEQ = 1
    EQ = 2
    NEQ = 3
    GREATER = 4
    GEQ = 5