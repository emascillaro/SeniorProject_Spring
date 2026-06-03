class ExpressionEvaluator:

    def evaluate(self, expression):

        return expression.operator.evaluate(
            expression.left_operand,
            expression.right_operand
        )
