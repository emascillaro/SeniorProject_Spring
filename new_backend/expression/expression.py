class Expression:
    def __init__(self, left_operand, operator, right_operand):
        self.left_operand = left_operand
        self.operator = operator
        self.right_operand = right_operand

    def __str__(self):
        return (
            f"{self.left_operand} "
            f"{self.operator.symbol} "
            f"{self.right_operand}"
        )
