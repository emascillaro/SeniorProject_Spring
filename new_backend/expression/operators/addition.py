from expression.operators.operator import Operator

class Addition(Operator):

    def evaluate(self, left_operand, right_operand):
        return left_operand + right_operand

    @property
    def symbol(self):
        return "+"
