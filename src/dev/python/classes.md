# Классы в Python. Основы

## Введение в ООП

Прежде чем погружаться в синтаксис классов Python, давайте вспомним, что такое **объектно-ориентированное программирование (ООП)**.

ООП — это парадигма программирования, где код организуется в виде объектов, содержащих данные (атрибуты) и поведение (методы). Основные концепции ООП:

- **Инкапсуляция** — объединение данных и методов в одном объекте
- **Наследование** — создание новых классов на основе существующих
- **Полиморфизм** — возможность использовать объекты разных классов одинаковым образом

Python реализует все эти концепции через классы!

## Базовый синтаксис классов

```python
class Counter:
    """Я считаю. И это всё."""
    
    def __init__(self, initial=0):  # конструктор
        self.value = initial
    
    def increment(self):
        self.value += 1
    
    def get(self):
        return self.value

# Использование
c = Counter(42)
c.increment()
print(c.get())  # 43
```

### Зачем нужен self?

В отличие от Java и C++, в Python нет ключевого слова `this`. Первый аргумент методов — это сам объект, который принято называть `self`:

```python
class Noop:
    def __init__(self):
        self.data = "hello"

# Технически можно назвать иначе, но не стоит!
class StrangeNoop:
    def __init__(ego):  # пожалуйста, не делайте так
        ego.data = "world"
```

## Атрибуты: экземпляра vs класса

```python
class Counter:
    # Атрибут класса - общий для всех экземпляров
    all_counters = []
    
    def __init__(self, initial=0):
        # Атрибут экземпляра - уникальный для каждого объекта
        self.value = initial
        Counter.all_counters.append(self)

# Добавить атрибут можно и после объявления класса
Counter.some_other_attribute = 42
```

## Инкапсуляция и соглашения об именовании

Python не имеет строгих модификаторов доступа, но использует соглашения:

```python
class Noop:
    public_attribute = 42      # публичный атрибут
    _internal_attribute = []   # внутренний (для разработчиков)
    __private_attribute = []   # "приватный" (name mangling)

noop = Noop()
print(noop.public_attribute)        # 42
print(noop._internal_attribute)     # [] (но не стоит!)
# print(noop.__private_attribute)   # Ошибка!
print(noop._Noop__private_attribute) # Работает (но не делайте так!)
```

## Свойства (Properties)

Свойства позволяют контролировать доступ к атрибутам:

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Температура ниже абсолютного нуля!")
        self._celsius = value

temp = Temperature(25)
print(f"{temp.celsius}°C = {temp.fahrenheit}°F")
temp.celsius = 30  # работает setter
```

## Наследование

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Подклассы должны реализовать этот метод")

class Dog(Animal):
    def speak(self):
        return f"{self.name} говорит: Гав!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} говорит: Мяу!"

animals = [Dog("Бобик"), Cat("Мурка")]
for animal in animals:
    print(animal.speak())
```

### Функция super()

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # вызов родительского конструктора
        self.student_id = student_id
```

## Множественное наследование

Python поддерживает множественное наследование, но используйте его осторожно!

```python
class Flyer:
    def fly(self):
        return "Лечу!"

class Swimmer:
    def swim(self):
        return "Плыву!"

class Duck(Flyer, Swimmer):
    pass

donald = Duck()
print(donald.fly())   # Лечу!
print(donald.swim())  # Плыву!
```

### Проблема ромбовидного наследования

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

print(D().method())  # Каков результат?
print(D.mro())       # Показывает порядок разрешения методов
```

## Декораторы классов

Декораторы могут преобразовывать классы:

```python
import functools

def singleton(cls):
    """Декоратор, превращающий класс в singleton"""
    instance = None
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance
    
    return wrapper

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Создано соединение с БД")

db1 = DatabaseConnection()  # Создано соединение с БД
db2 = DatabaseConnection()  # (ничего не выводится)
print(db1 is db2)  # True
```

## Магические методы

Магические методы (dunder methods) позволяют определить поведение объектов в различных ситуациях.

### Строковое представление

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
    
    def __str__(self):
        return f"'{self.title}' by {self.author}"
    
    def __repr__(self):
        return f"Book('{self.title}', '{self.author}')"

book = Book("Война и мир", "Л. Толстой")
print(str(book))    # 'Война и мир' by Л. Толстой
print(repr(book))   # Book('Война и мир', 'Л. Толстой')
```

### Операторы сравнения

```python
import functools

@functools.total_ordering
class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
    
    def __eq__(self, other):
        return (self.amount == other.amount and 
                self.currency == other.currency)
    
    def __lt__(self, other):
        if self.currency != other.currency:
            raise ValueError("Нельзя сравнивать разные валюты")
        return self.amount < other.amount

m1 = Money(100, "USD")
m2 = Money(150, "USD")
print(m1 < m2)  # True
```

### Вызываемые объекты

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
```

## Практический пример: Умный словарь

```python
from collections import deque

class MemorizingDict(dict):
    """Словарь, который запоминает историю изменений"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = deque(maxlen=10)
    
    def __setitem__(self, key, value):
        self.history.append(key)
        super().__setitem__(key, value)
    
    def get_history(self):
        return list(self.history)

# Использование
md = MemorizingDict()
md["a"] = 1
md["b"] = 2
md["c"] = 3
print(md.get_history())  # ['a', 'b', 'c']
```

## Заключение

Классы в Python предоставляют мощный и гибкий инструмент для организации кода. Они сочетают в себе:

- **Простоту** — интуитивно понятный синтаксис
- **Гибкость** — динамическое добавление атрибутов, множественное наследование
- **Мощь** — магические методы для кастомизации поведения
- **Элегантность** — декораторы и свойства для читаемого кода

Помните: классы — это инструмент, а не серебряная пуля. Используйте их там, где они действительно упрощают код и делают его более понятным!

> "Простота — залог надежности." — Эдсгер Дейкстра
