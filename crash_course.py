# Functions

def double(x):
    """
    This is where you put an optional docstring that explains what the
    function does. E.g. this function multiplies its input by 2.
    """
    return x * 2

def apply_to_one(f):
    """Calls the function f with 1 as its argument"""
    return f(1)

my_double = double              # refers to defined function double
x = apply_to_one(my_double)     # equals 2

# Lambda functions
y = apply_to_one(lambda x: x + 4)       # equals 5

def my_print(message = "my default message"):
    print(message)

my_print()              # prints "my default message"
my_print("hello")       # prints "hello"

# Strings
first_name = "Pooh"
last_name = "Bear"

# f-strings
print(f"{first_name} {last_name}")

# Exceptions
def divide(x, y):
    try:
        print(x/y)
    except ZeroDivisionError:
        print("cannot divide by zero")

# Lists

integer_list = [1, 2, 3]
hetero_list = ["string", True, 0.1]
list_of_lists = [integer_list, hetero_list, []]

list_length = len(integer_list)     # equals 3
list_sum = sum(integer_list)        # equals 5

# Different ways to write arrays
import numpy as np
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]                                  # method 1, hard code
y = np.arange(start = 0, stop = 10, step = 1)                       # method 2, numpy arange function
z = np.linspace(start = 0, stop = 9, num = 10, endpoint = True)     # method 3, numpy linspace


zero = x[0]         # equals 0
one = x[1]          # equals 1
nine = x[-1]        # equals 9, last element
eight = x[-2]       # equals 8, 2nd next to last element
x[0] = -1           # x now equals [-1, ..., 9]

# Slicing
first_three = x[:3]                 # [-1, 1, 2]
three_to_end = x[3:]                # [3, 4, ..., 9]
one_to_four = x[1:5]                # [1, 2, 3, 4]
last_three = x[-3:]                 # [7, 8, 9]
without_first_and_last = x[1:-1]    # [1, ..., 8]
copy_of_x = x[:]                    # [-1, ..., 9]
every_third = x[::3]                # [-1, 3, 6, 9]
five_to_three = x[5:2:-1]           # [5, 4, 3]

# List membership
1 in [1, 2, 3]    # True
0 in [1, 2, 3]    # False

# Concatenate
x = [1, 2, 3]
x.extend([4, 5, 6]) # [1, 2, 3, 4, 5, 6]
x = [1, 2, 3]
x.append([4, 5, 6]) # [1, 2, 3, [4, 5, 6]]

# Unpack
x, y = [1, 2]       # x is 1, y is 2
x, y = 1, 2         # x is 1, y is 2
x, y = y, x         # Pythonic method to swap variable values, now x is 2, y is 1
_, y = [1, 2]       # underscore indicates value to be discarded, y == 2 

# Tuples
my_tuple = (1, 2)   # Tuples immutable

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")

def sum_and_product(x, y):
    return (x + y), (x * y) # Tuples useful returning multiple values from functions

# Dictionaries
grades = {"Bill": 80, "Bob": 60}
bills_grade = grades["Bill"]            # equals 80
bills_grade = grades.get("Bill", 0)     # equals 80
bos_grade = grades["Bo"]                # KeyError
bos_grade = grades.get("Bo", 0)         # equals 0

grade_keys = grades.keys()              # iterable for the keys
grade_values = grades.values()          # iterable for the values
grade_items = grades.items()            # iterable for the (key, value) tuples

# defaultdict
word_counts = {}
document = ["apple", "banana", "apple", "orange", "apple"]

# Method 1
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# Method 2
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

# Method 3
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

# Method 4 (preferred method)
from collections import defaultdict
word_counts = defaultdict(int)          # int() produces 0
for word in document:
    word_counts[word] += 1

# Counters

from collections import Counter
c = Counter([0, 1, 2, 0])           # c (basically) is {0:2, 1:1, 2:1}

word_counts = Counter(document)     # also works

# Prints the 10 most common words and their count in document
for word, count in word_counts.most_common(10):
    print(word, count)

# Sets
primes_below_ten = {2, 3, 5, 7}         # these are a collection of distinct/unique elements 

# Sets are useful in a membership test

stopwords_list = ["a", "an", "at"]
"zip" in stopwords_list                 # method checks every element (including repeating elements)

stopwords_set = set(stopwords_list)     # convert list to set
"zip" in stopwords_set                  # method checks only distinct elements, it's faster

# Control Flow
if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for else if"
else:
    message = "when all else fails use else"

ternary = "even" if x % 2 == 0 else "odd"       # if-then-else on one line

x = 0
while x < 10:       # for and in loops utilised more often
    print(x)
    x += 1

for x in range(10):
    if x == 3:
        continue    # go to next iteration
    elif x == 5:
        break       # end loop
    print(x)

# Truthiness
one_less_than_two = 1 < 2               # returns True
true_equals_false = True == False       # returns False

x = "hello"
#if condition returns False, AssertionError is raised:
assert x == "goodbye", "x should be 'hello'"

x = None
assert x == None            # not Pythonic
assert x is None            # is Pythonic checking if value is 'None'

# Sorting
x = [4, 2, 1, 3]
y = sorted(x)           # becomes [1, 2, 3, 4]
x.sort()                # x is sorted

# List comprehensions
even_numbers = [x for x in range(5) if x % 2 == 0]      # [0, 2, 4]
squares = [x * x for x in range (5)]                    # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]            # [0, 4, 16]

# Automated Testing and Assert
def smallest_item(xs):
    return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1

# Less common method
def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)

# Object-Oriented Programming OOP

# Encapsulation
class BankAccount:
    """
    In this code, we have a BankAccount class with private attribute __balance. 
    The class has methods to deposit, withdraw, and get the balance of the account. 
    We create an instance of the BankAccount class with an initial balance of 1000 
    and then perform a deposit of 500. We then attempt to withdraw 2000 (which is 
    not possible due to insufficient balance) and print the current balance of the 
    account (which should be 1500).
    """
    def __init__(self, balance):
        self.__balance = balance             # initialize private attribute __balance

    def deposit(self, amount):
        self.__balance += amount            # add amount to private attribute __balance

    def withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance -= amount        # subtract amount from private attribute __balance (encapsulating balance)
        else:
            print("Insufficient balance")

    def get_balance(self):
        return self.__balance               # return private attribute __balance

account = BankAccount(1000)     # create an instance of BankAccount with initial balance of 1000
account.deposit(500)            # deposit 500 into the account
account.withdraw(2000)          # prints "Insufficient balance"
print(account.get_balance())    # prints 1500


# Inheritance
class Animal:
    """
    In this example, we have a Animal class with a name attribute and an abstract 
    method make_sound. We then create two subclasses of Animal, Dog and Cat, which 
    inherit from Animal. The Dog and Cat classes override the make_sound method with 
    their own implementation.

    We then create instances of Dog and Cat with names "Buddy" and "Mittens", 
    respectively, and call the make_sound method on them. This demonstrates how the 
    Dog and Cat classes inherit the name attribute and the abstract method make_sound 
    from the Animal class and implement their own behavior for the make_sound method.
    """
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        pass                    # abstract method Note: The pass statement is used as a placeholder for future code.

class Dog(Animal):
    species = "Canis lupus familiaris"

    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    species = "Felis silvestris catus"

    def make_sound(self):
        return "Meow!"

dog = Dog("Buddy")
cat = Cat("Mittens")
print(dog.name + " says " + dog.make_sound())  # prints "Buddy says Woof!"
print(cat.name + " says " + cat.make_sound())  # prints "Mittens says Meow!"

# Abstraction
from abc import ABC, abstractmethod

class Animal(ABC):
    """
    In this example, we have an Animal class that is an abstract class and contains an 
    abstract method make_sound. We then create two subclasses of Animal, Dog and Cat, 
    which inherit from Animal. The Dog and Cat classes override the make_sound method with 
    their own implementation.

    We then create instances of Dog and Cat with names "Buddy" and "Mittens", respectively, 
    and add them to a list called animals. We then iterate through the list and call the name 
    and make_sound methods on each object.

    This demonstrates how the Animal class acts as an abstraction, where the abstract make_sound 
    method is defined in the abstract base class and the implementation is provided in the subclasses. 
    The Animal class itself cannot be instantiated, but we can create instances of the subclasses 
    and treat them as Animal objects. This allows us to use the common name attribute and make_sound 
    method defined in the Animal class without knowing the specific implementation in the subclasses.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def make_sound(self):
        pass  # abstract method

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

dog = Dog("Buddy")
cat = Cat("Mittens")
animals = [dog, cat]

for animal in animals:
    print(animal.name + " says " + animal.make_sound())

# Polymorphism
"""
In this example, we have a Shape class with an abstract method draw. We then create 
two subclasses of Shape, Rectangle and Circle, which inherit from Shape. The Rectangle 
and Circle classes override the draw method with their own implementation.

We then create a list of Shape objects containing one Rectangle and one Circle. We iterate 
through the list and call the draw method on each object. This demonstrates how the Rectangle 
and Circle objects can be treated as Shape objects and their respective draw methods are 
called based on their specific implementation.

This is an example of polymorphism, where objects of different classes can be treated as if 
they are of the same class and their methods can be called interchangeably. In this case, the 
Rectangle and Circle objects are both treated as Shape objects and their respective draw methods 
are called based on their specific implementation.
"""
class Shape:
    def draw(self):
        pass  # abstract method

class Rectangle(Shape):
    def draw(self):
        return "Drawing a rectangle"

class Circle(Shape):
    def draw(self):
        return "Drawing a circle"

shapes = [Rectangle(), Circle()]

for shape in shapes:
    print(shape.draw())


# Iterables and Generators
names = ["Alice", "Bob", "Charlie", "Debbie"]
for idx, name in enumerate(names):
    print(f"name {idx} is {name}")

# Randomness
import random           # module is psuedorandom ie deterministic
random.seed(10)         # this ensures we get the same result every time 
random_choice = random.choice(names)        # select random name

four_with_replacement = [random.choice(range(10) for _ in range(4))]

# Regular Expressions
import re
re_examples = [
    not re.match("a", "cat"),                   # 'cat' doesnt start with 'a'
    re.search("a", "cat"),                      # 'cat' has an 'a' in it
    not re.match("c", "dog"),                   # 'dog' doesnt have a 'c'
    3 == len(re.split("[ab]", "carbs")),        # split on 'a' or 'b' to ['c', 'b', 's']
    "R-D-" == re.sub("[0-9]", "-", "R2D2")      # replace digits with dashes
]

# zip and Argument Unpacking
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
[pair for pair in zip(list1, list2)]        # pair is [('a', 1), ('b', 2), ('c', 3)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)              # '*' denotes argument unpacking

# args and kwargs
def example_function(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

example_function(1, 2, 3, a="apple", b="banana")

# args: (1, 2, 3)
# kwargs: {'a': 'apple', 'b': 'banana'}

# Type Annotations
"""
In this example, we are defining a function called add_numbers that takes two arguments x and y, 
both of which are annotated with the type int. The -> int annotation at the end indicates that 
the function returns an integer.

Then, we call the function with two integer arguments 5 and 3, and store the result in a variable 
called result. Finally, we print out the value of result, which should be 8.

Note that type annotations are optional in Python, and are mainly used to provide additional information 
about the types of values that functions or variables are expected to work with. This can be helpful for 
catching type-related errors early on, and for making code easier to understand and maintain. However, 
Python itself does not enforce these annotations in any way, and they do not affect the runtime behavior 
of the program.
"""
def add_numbers(x: int, y: int) -> int:     # arguments 'x' and 'y' are integers and function returns as an integer
    return x + y

result = add_numbers(5, 3)
print(result)

from typing import List
def total(xs: List(float)) -> float:
    return sum(total)

from typing import Optional
values: List(int) = []
best_so_far: Optional[float] = None