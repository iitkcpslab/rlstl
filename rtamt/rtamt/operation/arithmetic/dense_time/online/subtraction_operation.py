from rtamt.operation.abstract_operation import AbstractOperation
import rtamt.operation.stl.dense_time.online.intersection as intersect

class SubtractionOperation(AbstractOperation):
    def __init__(self):
        self.left = []
        self.right = []

    def update(self, left_list, right_list):
        out = []
        self.left = self.left + left_list
        self.right = self.right + right_list

        out, last, left, right = intersect.intersection(self.left, self.right, intersect.subtraction)

        self.left = left
        self.right = right
        if out:
            self.last = last

        return out

    def update_final(self, *args, **kargs):
        return self.update(args[0], args[1]) + [self.last]
    #
    # def offline(self, left_list, right_list):
    #     out = []
    #     self.left = self.left + left_list
    #     self.right = self.right + right_list
    #
    #     out, last, left, right = intersect.intersection(self.left, self.right, intersect.subtraction)
    #     out.append(last)
    #
    #     return out
