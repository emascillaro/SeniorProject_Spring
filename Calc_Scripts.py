import numpy

#from otherCode import equationString

equationString = "253^386"

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
    print(add(int_numA, int_numB))

if symbol == '-':
    print(subtract(int_numA, int_numB))

if symbol == '*':
    print(multiply(int_numA, int_numB))

if symbol == '/':
    print(divide(int_numA, int_numB))

if symbol == '^':
    print(exponent(int_numA, int_numB))
