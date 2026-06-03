from expression.operators.operator import Operator

class Multiplication(Operator):

    def evaluate(self, a, b):
        return a * b
