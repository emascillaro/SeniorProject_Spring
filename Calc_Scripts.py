import numpy

#from otherCode import equationString

def calculation(equationString):

    if '+' in equationString:
        location = equationString.index('+')
        symbol = "+"

    elif '-' in equationString:
        location = equationString.index('-')
        symbol = "-"

    elif '*' in equationString:
        location = equationString.index('*')
        symbol = "*"

    elif '/' in equationString:
        location = equationString.index('/')
        symbol = "/"

    elif '^' in equationString:
        location = equationString.index('^')
        symbol = "^"

    #First Number
    string_numA = equationString[0:location]
    int_numA = int(string_numA)

    #Second Number
    string_numB = equationString[location + 1:]
    int_numB = int(string_numB)

    # Calculator Functions
    def add (numA, numB):
        return numA + numB

    def subtract (numA, numB):
        return numA - numB

    def multiply (numA, numB):
        return numA * numB

    def divide (numA, numB):
        return numA / numB

    def exponent (numA, numB):
        return numA ** numB

    if symbol == '+':
        return(add(int_numA, int_numB))

    if symbol == '-':
        return(subtract(int_numA, int_numB))

    if symbol == '*':
        return(multiply(int_numA, int_numB))

    if symbol == '/':
        return(divide(int_numA, int_numB))

    if symbol == '^':
        return(exponent(int_numA, int_numB))