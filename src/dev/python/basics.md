## Basics of Python

Python is a high-level, dynamically typed multiparadigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable.

We recommend to read [PEP 8](https://www.python.org/dev/peps/pep-0008/).

### Python versions and Zen of Python

There are currently supported versions of Python 3.X. Support for Python 2.7 ended in 2020. For this class all code will use Python 3.7.

You can check your Python version at the command line by running python --version.


```python
!python --version
```

    Python 3.7.4



```python
import this
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


### Basic data types

#### Numbers

Integers and floats work as you would expect from other languages:


```python
x = 3
print(x, type(x))
```

    3 <class 'int'>



```python
print(x + 1)   # Addition;
print(x - 1)   # Subtraction;
print(x * 2)   # Multiplication;
print(x ** 2)  # Exponentiation;
```

    4
    2
    6
    9



```python
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
```

    4
    8



```python
y = 2.5
print(type(y)) # Prints "<type 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

    <class 'float'>
    2.5 3.5 5.0 6.25


Note that unlike many languages, Python does not have unary increment (x++) or decrement (x--) operators.

Python also has built-in types for long integers and complex numbers; you can find all of the details in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#numeric-types-int-float-long-complex).

#### Booleans

Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (`&&`, `||`, etc.):


```python
T, F = True, False
print(type(T)) # Prints "<type 'bool'>"
```

    <class 'bool'>


Now we let's look at the operations:


```python
print(T and F) # Logical AND;
print(T or F)  # Logical OR;
print(not T)   # Logical NOT;
print(T != F)  # Logical XOR;
```

    False
    True
    False
    True


#### Strings


```python
hello = 'hello'   # String literals can use single quotes
world = "world"   # or double quotes; it does not matter.
print(hello, len(hello))
```

    hello 5



```python
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
```

    hello world



```python
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

    hello world 12


String objects have a bunch of useful methods; for example:


```python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                               # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

    Hello
    HELLO
      hello
     hello 
    he(ell)(ell)o
    world


You can find a list of all string methods in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#string-methods).

### Containers

Python includes several built-in container types: lists, dictionaries, sets, and tuples.

#### Lists

A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:


```python
xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
```

    [3, 1, 2] 2
    2



```python
xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)
```

    [3, 1, 'foo']



```python
xs.append('bar') # Add a new element to the end of the list
print(xs)  
```

    [3, 1, 'foo', 'bar']



```python
x = xs.pop()     # Remove and return the last element of the list
print(x, xs) 
```

    bar [3, 1, 'foo']


As usual, you can find all the gory details about lists in the [documentation](https://docs.python.org/3.7/tutorial/datastructures.html#more-on-lists).

#### Slicing

In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:


```python
nums = range(5)    # range is a built-in function that creates a list of integers
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
```

    range(0, 5)
    range(2, 4)
    range(2, 5)
    range(0, 2)
    range(0, 5)
    range(0, 4)


#### Loops

You can loop over the elements of a list like this:


```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
```

    cat
    dog
    monkey


If you want access to the index of each element within the body of a loop, use the built-in `enumerate` function:


```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
```

    #1: cat
    #2: dog
    #3: monkey


#### List comprehensions:

When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:


```python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)
```

    [0, 1, 4, 9, 16]


You can make this code simpler using a list comprehension:


```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)
```

    [0, 1, 4, 9, 16]


List comprehensions can also contain conditions:


```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)
```

    [0, 4, 16]


#### Dictionaries

A dictionary stores (key, value) pairs, similar to a `Map` in Java or an object in Javascript. You can use it like this:


```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
```

    cute
    True



```python
d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
```

    wet



```python
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))   # Get an element with a default; prints "wet"
```

    N/A
    wet



```python
del d['fish']        # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

    N/A


You can find all you need to know about dictionaries in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#dict).

It is easy to iterate over the keys in a dictionary:


```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
```

    A person has 2 legs
    A cat has 4 legs
    A spider has 8 legs


Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:


```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)
```

    {0: 0, 2: 4, 4: 16}


#### Sets

A set is an unordered collection of distinct elements. As a simple example, consider the following:


```python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
```

    True
    False



```python
animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;
```

    True
    3



```python
animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       
animals.remove('cat')    # Remove an element from a set
print(len(animals))       
```

    3
    2


_Loops_: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:


```python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"
```

    #1: fish
    #2: dog
    #3: cat


Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:


```python
from math import sqrt
print({int(sqrt(x)) for x in range(30)})
```

    {0, 1, 2, 3, 4, 5}


#### Tuples

A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:


```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d[t])       
print(d[(1, 2)])
```

    <class 'tuple'>
    5
    1


### Functions

Python functions are defined using the `def` keyword. For example:


```python
def sign(x: float) -> str:
    '''Function sign''' # document line
    
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
```

    negative
    zero
    positive



```python
help(sign)
```

    Help on function sign in module __main__:
    
    sign(x: float) -> str
        Function sign
    


We will often define functions to take optional keyword arguments, like this:


```python
def hello(name: str, loud: bool=False) -> None:
    '''Function hello
    
    If loud is True, 
    then the name is printed in capital letters.
    '''
    
    if loud:
        print('HELLO, %s' % name.upper())
    else:
        print('Hello, %s!' % name)

hello('Bob')
hello('Fred', loud=True)
```

    Hello, Bob!
    HELLO, FRED



```python
help(hello)
```

    Help on function hello in module __main__:
    
    hello(name: str, loud: bool = False) -> None
        Function hello
        
        If loud is True, 
        then the name is printed in capital letters.
    


### Classes

The syntax for defining classes in Python is straightforward:


```python
class Greeter:
    '''Class Greeter
    
    method greet:
    If loud is True, 
    then the name is printed in capital letters.
    '''

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud: bool=False) ->None:
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

    Hello, Fred
    HELLO, FRED!



```python
help(Greeter)
```

    Help on class Greeter in module __main__:
    
    class Greeter(builtins.object)
     |  Greeter(name)
     |  
     |  Class Greeter
     |  
     |  method greet:
     |  If loud is True, 
     |  then the name is printed in capital letters.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, name)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  greet(self, loud: bool = False) -> None
     |      # Instance method
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
