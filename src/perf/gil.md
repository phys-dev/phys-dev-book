# Multithreading and GIL

## || minimum

### Process

* Process - running program.
* Each process has a state isolated from other processes:
* * virtual address space,
* * pointer to executable instruction,
* * call stack,
* * system resources, such as open file
  descriptors.
* Processes are convenient for simultaneously performing multiple tasks.
* Alternative way: delegate each task to a thread.

### Thread

* A thread is similar to a process in that its execution occurs independently of other threads (and processes).
* Unlike a process, a thread executes within a process and shares the address space and system resources with it.
* Threads are convenient for simultaneously performing several tasks that require access to a shared state.
* The joint execution of several processes and threads is controlled by the operating system, sequentially allowing each process or thread to use some processor cycles.

## `threading` module

* A thread in Python is a system thread, that is, it is not the interpreter that controls its execution, but the operating system.
* You can create a thread using the `Thread` class from the module of the standard `threading` library.


```python
import time
from threading import Thread

def countdown(n):
    for i in range(n):
        print(n - i - 1, "left")
        time.sleep(1)
```


```python
t = Thread(target=countdown, args=(3,))
```


```python
t.start()
```

    2 left


An alternative way to create a stream is inheritance:


```python
class CountdownThread(Thread):
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    def run(self): # start method.
        for i in range(self.n):
            print(self.n - i - 1, "left")
            time.sleep(1)
```


```python
t = CountdownThread(3)
```


```python
t.start()
```

    2 left


The disadvantage of this approach is that it limits the reuse of code: the functionality of the `CountdownThread` class can be used only in a separate thread.

* When creating a stream, you can specify a name. By default it is **'Thread-N'**:


```python
Thread().name
```




    'Thread-6'




```python
Thread(name="NumberCruncher").name
```




    'NumberCruncher'



* Each active thread has an id - a non-negative number unique to all active threads.


```python
t = Thread()
t.start()
t.ident
```




    123145519448064



* The `join` method allows you to wait until the stream finishes.
* * The execution of the calling thread will pause until thread t ends.
* * Repeated calls to the `join` method have no effect.


```python
t = Thread(target=time.sleep, args=(5, )) 
t.start()
t.join() # locks for 5 seconds
t.join() # performed instantly
```

    1 left
    1 left
    0 left
    0 left


* You can check if the thread is running using the `is_alive` method:


```python
t = Thread(target=time.sleep, args=(5, )) 
t.start()
```


```python
t.is_alive() # False after 5 seconds
```




    True



* A daemon is a thread created with the `daemon=True` argument:

* The difference between a daemon thread and a regular thread is that daemon threads are **automatically** destroyed when exiting the interpreter.


```python
t = Thread(target=time.sleep, args=(5,), daemon=True)
t.start()
```


```python
t.daemon
```




    True



* In Python, there is no built-in mechanism for terminating threads - this is not an accident, but an informed decision of the language developers.
* Correct termination of the flow is often associated with the release of resources, for example:
* * a stream can work with a file whose descriptor needs to be closed,
* * or capture a synchronization primitive.
* To end a stream, a flag is usually used:


```python
class Task:
    def __init__(self):
        self._running = True
    
    def terminate(self):
        self._running = False
    
    def run(self, n):
        while self._running:
            ...
```

The set of synchronization primitives in the threading module is standard:
* `Lock` - a regular mutex, used to provide exclusive access to a shared state.
* `RLock` - a recursive mutex that allows a thread that owns a mutex to capture the mutex more than once.
* `Semaphore` - a variation of the mutex that allows you to capture yourself no more than a fixed number of times.
* `BoundedSemaphore` - a semaphore that makes sure that it is captured and released the same number of times.

All synchronization primitives implement a single interface:
* the acquire method captures the synchronization primitive,
* and the release method releases it.


```python
class SharedCounter:
    def __init__(self, value):
        self._value = value
        self._lock = Lock()
    
    def increment(self, delta=1):
        self._lock.acquire()
        self._value += delta
        self._lock.release()
    
    def get(self):
        return self._value
```

## `queue` module

The `queue` module implements several thread-safe queues:
* `Queue` - FIFO queue,
* `LifoQueue` — LIFO queue of stacks,
* `PriorityQueue` — a queue of elements — a parse
(priority, item).
* There are no special frills in the implementation of queues: all state-changing methods work “inside” the mutex.
* The `Queue` class uses `deque` as the container, and the `LifoQueue` and `PriorityQueue` classes use the list.


```python
def worker(q):
    while True:
        item = q.get() # blocking expects next
        do_something(item) # element
        q.task_done() # notifies the execution queue
        
def master(q):
    for item in source():
        q.put(item)
        
        # blocking waits until all elements of the queue
        # will not be processed
        q.join()
```

## `futures` module

* The `concurrent.futures` module contains the abstract class `Executor` and its implementation as a thread pool - `ThreadPoolExecutor`.
* The executor’s interface consists of only three methods:


```python
from concurrent.futures import *
```


```python
executor = ThreadPoolExecutor(max_workers=4)
```


```python
executor.submit(print, "Hello, world!")
```

    Hello, world!




    <Future at 0x1043b1d90 state=running>



    



```python
list(executor.map(print, ["Knock?", "Knock!"]))
```

    Knock?
    Knock!





    [None, None]




```python
executor.shutdown()
```

* Contractors support the context manager protocol:


```python
with ThreadPoolExecutor(max_workers=4) as executor:
    ...
```

* The `Executor.submit` method returns an instance of the `Future` class that encapsulates asynchronous calculations.

What can be done with `Future`?


```python
with ThreadPoolExecutor(max_workers=4) as executor:
    f = executor.submit(sorted, [4, 3, 1, 2])
```

* Ask about the calculation status:


```python
f.running(), f.done(), f.cancelled()
```




    (False, True, False)



* Wait for the blocking result of the calculation:


```python
print(f.result())
```

    [1, 2, 3, 4]



```python
print(f.exception())
```

    None


* Add a function that will be called after the calculation is completed:


```python
f.add_done_callback(print)
```

    <Future at 0x1043b7f10 state=finished returned list>


## `futures` module example: `integrate`


```python
import math

def integrate(f, a, b, *, n_iter=1000):
    acc = 0
    step = (b - a) / n_iter
    for i in range(n_iter):
        acc += f(a + i * step) * step
    return acc
```


```python
integrate(math.cos, 0, math.pi / 2)
```




    1.0007851925466296




```python
from functools import partial

def integrate_async(f, a, b, *, n_jobs, n_iter=1000):
    executor = ThreadPoolExecutor(max_workers=n_jobs)
    spawn = partial(executor.submit, integrate, f,
                    n_iter=n_iter // n_jobs)
    step = (b-a)/n_jobs
    fs=[spawn(a+i*step,a+(i+1)*step)
        for i in range(n_jobs)]
    return sum(f.result() for f in as_completed(fs))
```


```python
integrate_async(math.cos, 0, math.pi / 2, n_jobs=2)
```




    1.0007851925466305



## Concurrency and Competition

Compare the performance of the serial and parallel versions of the integrate function using the “magic” `timeit` command:


```python
%%timeit -n100
integrate(math.cos, 0, math.pi / 2, n_iter=10**6)
```

    154 ms ± 2.28 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%%timeit -n100
integrate_async(math.cos, 0, math.pi / 2, n_iter=10**6, n_jobs=2)
```

    142 ms ± 2.11 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


### GIL

* GIL (global interpreter lock) is a mutex that ensures that only one thread at a time has access to the internal state of the interpreter.
* The Python C API allows you to release the GIL, but it is only safe when working with objects that are not dependent on the Python interpreter.

### Is GIL Bad?
* The answer depends on the task.
* The presence of GIL makes it impossible to use threads in Python for parallelism: several threads do not speed up, and sometimes even slow down the program.
* GIL does not interfere with the use of threads for competition when working with I/O.

### C and Cython - GIL Remedy


```python
%load_ext Cython
```


```cython
%%cython

from libc.math cimport cos

def integrate(f, double a, double b, long n_iter):
    cdef double acc = 0
    cdef double step=(b-a)/n_iter
    cdef long i
    with nogil:
        for i in range(n_iter):
            acc += cos(a + i * step) * step
    return acc
```


```python
%%timeit -n100
integrate_async(math.cos, 0, math.pi / 2, n_iter=10**6, n_jobs=2)
```

    5.88 ms ± 126 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


## `multiprocessing` module
### Processes Another GIL Remedy

* You can use processes instead of threads.
* Each process will have its own GIL, but it will not prevent them from working in parallel.
* The `multiprocessing` module is responsible for working with processes in Python:


```python
import multiprocessing as mp
```


```python
p = mp.Process(target=countdown, args=(5, ))
```


```python
p.start()
```

    4 left
    3 left
    2 left
    1 left
    0 left


* The module implements basic synchronization primitives: mutexes, semaphores, conditional variables.
* To organize the interaction between processes, you can use `Pipe` - a socket-based connection between two processes:


```python
def ponger(conn):
    conn.send("pong")
```


```python
parent_conn, child_conn = mp.Pipe()
p = mp.Process(target=ponger, args=(child_conn, ))
```


```python
p.start()
```


```python
parent_conn.recv()
```




    'pong'




```python
p.join()
```

## Process and performance

The implementation of the `integrate_async` function based on the thread pool worked for a long time, let's try to use the process pool:


```python
from concurrent.futures import ProcessPoolExecutor
```


```python
def integrate_async(f, a, b, *, n_jobs, n_iter=1000):
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    spawn = partial(executor.submit, integrate, f,
                    n_iter=n_iter // n_jobs)

    step = (b - a) / n_jobs
    fs=[spawn(a + i * step, a + (i + 1) * step)
        for i in range(n_jobs)]
    
    return sum(f.result() for f in as_completed(fs))
```


```python
%%timeit -n100
integrate_async(math.cos, 0, math.pi / 2, n_iter=10**6, n_jobs=2)
```

    16.6 ms ± 144 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


## `joblib` package

The `joblib` package implements a parallel analogue of the for loop, which is convenient for parallel execution of independent tasks.


```python
from joblib import Parallel, delayed

def integrate_async(f, a, b, *, n_jobs, n_iter=1000, backend=None):
    step = (b - a) / n_jobs
    with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
        fs = (delayed(integrate)(f, a + i * step,
                                 a + (i + 1) * step, 
                                 n_iter=n_iter // n_jobs)
              for i in range(n_jobs))
    return sum(parallel(fs))
```


```python
%%timeit -n100
integrate_async(math.cos, 0, math.pi / 2, n_iter=10**6, n_jobs=2, backend="threading")
```

    104 ms ± 280 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%%timeit -n100
integrate_async(math.cos, 0, math.pi / 2, n_iter=10**6, n_jobs=2, backend="multiprocessing")
```

    290 ms ± 1.13 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


## Summary

* GIL is a global mutex that limits the use of threads for parallelism in programs in Python.
* For programs that use mainly I / O, GIL is not scary: in CPython, these operations release the GIL.
* For programs that need parallelism, there are options for increasing productivity:
* * write critical functionality in C or Cython; 
* * or use the multiprocessing module.

## `numba` JIT compiler


```python
import math
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def integrate(a, b, *, n_iter=1000):
    acc = 0
    step = (b - a) / n_iter
    for i in prange(n_iter):
        acc += math.cos(a + i * step) * step
    return acc
```


```python
%%timeit -n100
integrate(0, math.pi / 2, n_iter=10**6)
```

    5.11 ms ± 983 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Profit.
