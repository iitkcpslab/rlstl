from rtamt.operation.abstract_operation import AbstractOperation


class ImpliesOperation(AbstractOperation):
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, left, right):
        out = max(-left, right)
        return out
