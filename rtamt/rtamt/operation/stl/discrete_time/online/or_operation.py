from rtamt.operation.abstract_operation import AbstractOperation
import numpy as np

class OrOperation(AbstractOperation):
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, left, right):
        #out = max(left, right)
        #out = left*right
        print("inside or :"+str(left)+" or "+str(right))
        #return out
        return [left,right]
