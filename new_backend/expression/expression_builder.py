from expression.expression import Expression

from expression.operators.addition import Addition
from expression.operators.subtraction import Subtraction
from expression.operators.multiplication import Multiplication
from expression.operators.division import Division
from expression.operators.exponent import Exponent

class ExpressionBuilder:

    OPERATORS = {
        '+': Addition(),
        '-': Subtraction(),
        '*': Multiplication(),
        '/': Division(),
        '**': Exponent()
    }

    def build(self, equation_string):

        for symbol, operator_class in self.OPERATORS.items():

            if symbol in equation_string:

                location = equation_string.index(symbol)

                left_operand = int(equation_string[:location])
                right_operand = int(equation_string[location + 1:])

                operator = operator_class()

                return Expression(left_operand, operator, right_operand)

        raise ValueError(f"Invalid expression: {equation_string}")
