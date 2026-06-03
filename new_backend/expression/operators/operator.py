from abc import ABC, abstractmethod

class Operator(ABC):

    @abstractmethod
    def evaluate(self, left_operand, right_operand):
        pass

    @property
    @abstractmethod
    def symbol(self):
        pass
