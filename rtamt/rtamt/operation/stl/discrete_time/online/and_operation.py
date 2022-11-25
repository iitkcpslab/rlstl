from rtamt.operation.abstract_operation import AbstractOperation


class AndOperation(AbstractOperation):
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, left, right):
        #out = min(left, right)
        #out = (left+right)
        #print("inside and :"+str(left)+" and "+str(right))
        #return out
        return [left,right]
