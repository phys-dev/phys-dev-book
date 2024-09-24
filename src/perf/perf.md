# Performance

## Class `Matrix`


```python
import random

class Matrix(list):
    @classmethod
    def zeros(cls, shape):
        n_rows, n_cols = shape
        return cls([[0] * n_cols for i in range(n_rows)])

    @classmethod
    def random(cls, shape):
        M, (n_rows, n_cols) = cls(), shape
        for i in range(n_rows):
            M.append([random.randint(-255, 255)
                      for j in range(n_cols)])
        return M

    def transpose(self):
        n_rows, n_cols = self.shape
        return self.__class__(zip(*self))

    @property
    def shape(self):
        return ((0, 0) if not self else
                (len(self), len(self[0])))
```


```python
def matrix_product(X, Y):
    """Computes the matrix product of X and Y.

    >>> X = Matrix([[1], [2], [3]])
    >>> Y = Matrix([[4, 5, 6]])
    >>> matrix_product(X, Y)
    [[4, 5, 6], [8, 10, 12], [12, 15, 18]]
    >>> matrix_product(Y, X)
    [[32]]
    """
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    # верим, что с размерностями всё хорошо
    Z = Matrix.zeros((n_xrows, n_ycols))
    for i in range(n_xrows):
        for j in range(n_xcols):
            for k in range(n_ycols):
                Z[i][k] += X[i][j] * Y[j][k]
    return Z
```


```python
%doctest_mode
```

    Exception reporting mode: Plain
    Doctest mode is: ON



```python
>>> X = Matrix([[1], [2], [3]])
>>> Y = Matrix([[4, 5, 6]])
>>> matrix_product(X, Y)
[[4, 5, 6], [8, 10, 12], [12, 15, 18]]
>>> matrix_product(Y, X)

[[32]]
```




    [[32]]




```python
%doctest_mode
```

    Exception reporting mode: Context
    Doctest mode is: OFF


# Runtime measurement

Everything seems to work, but how fast? Use the magic `timeit` command to check.


```python
%%timeit shape = 64, 64; X = Matrix.random(shape); Y = Matrix.random(shape)
matrix_product(X, Y)
```

    86.6 ms ± 1.52 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


Total: Multiplying two 64x64 matrices takes near 0.1 seconds. Y U SO SLOW?

We define an auxiliary function `bench`, which generates random matrices of the specified size, and then` n_iter` times multiplies them in a loop.


```python
def bench(shape=(64, 64), n_iter=16):
    X = Matrix.random(shape)
    Y = Matrix.random(shape)
    for iter in range(n_iter):
        matrix_product(X, Y)    
```

Let's try to look at it more closely with the help of the `line_profiler` module.


```python
#!pip install line_profiler
```


```python
%load_ext line_profiler
%lprun -f matrix_product bench()
```

Note that the operation `list .__ getitem__` is not free. Swap the `for` loops so that the code does less index lookups.


```python
def matrix_product(X, Y):
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    Z = Matrix.zeros((n_xrows, n_ycols))
    for i in range(n_xrows):
        Xi = X[i]
        for k in range(n_ycols):
            acc = 0
            for j in range(n_xcols):
                acc += Xi[j] * Y[j][k]
            Z[i][k] = acc
    return Z
```


```python
%lprun -f matrix_product bench()
```

2 seconds faster, but still too slow:> 30% of the time goes exclusively to iteration! Fix it.


```python
def matrix_product(X, Y):
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    Z = Matrix.zeros((n_xrows, n_ycols))
    for i in range(n_xrows):
        Xi, Zi = X[i], Z[i]
        for k in range(n_ycols):
            Zi[k] = sum(Xi[j] * Y[j][k] for j in range(n_xcols))
    return Z
```


```python
%lprun -f matrix_product bench()
```

The matrix_product functions are pretty prettier. But, it seems, this is not the limit. Let’s try again to remove unnecessary index calls from the innermost cycle.


```python
def matrix_product(X, Y):
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    Z = Matrix.zeros((n_xrows, n_ycols))
    Yt = Y.transpose()  # <--
    for i, (Xi, Zi) in enumerate(zip(X, Z)):
        for k, Ytk in enumerate(Yt):
            Zi[k] = sum(Xi[j] * Ytk[j] for j in range(n_xcols))
    return Z
```

# Numba

Numba does not work with inline lists. Rewrite the `matrix_product` function using ndarray.


```python
import numba
import numpy as np


@numba.jit
def jit_matrix_product(X, Y):
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    Z = np.zeros((n_xrows, n_ycols), dtype=X.dtype)
    for i in range(n_xrows):
        for k in range(n_ycols):
            for j in range(n_xcols):
                Z[i, k] += X[i, j] * Y[j, k]
    return Z
```

Let's see what happened.


```python
shape = 64, 64
X = np.random.randint(-255, 255, shape)
Y = np.random.randint(-255, 255, shape)

%timeit -n100 jit_matrix_product(X, Y)
```

    The slowest run took 21.46 times longer than the fastest. This could mean that an intermediate result is being cached.
    495 µs ± 900 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# Cython


```python
%load_ext cython
```


```python
%%capture
%%cython -a
import random

class Matrix(list):
    @classmethod
    def zeros(cls, shape):
        n_rows, n_cols = shape
        return cls([[0] * n_cols for i in range(n_rows)])

    @classmethod
    def random(cls, shape):
        M, (n_rows, n_cols) = cls(), shape
        for i in range(n_rows):
            M.append([random.randint(-255, 255)
                      for j in range(n_cols)])
        return M

    def transpose(self):
        n_rows, n_cols = self.shape
        return self.__class__(zip(*self))

    @property
    def shape(self):
        return ((0, 0) if not self else
                (int(len(self)), int(len(self[0]))))

    
def cy_matrix_product(X, Y):
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    Z = Matrix.zeros((n_xrows, n_ycols))
    Yt = Y.transpose()
    for i, Xi in enumerate(X):
        for k, Ytk in enumerate(Yt):
            Z[i][k] = sum(Xi[j] * Ytk[j] for j in range(n_xcols))
    return Z
```


```python
X = Matrix.random(shape)
Y = Matrix.random(shape)
```


```python
%timeit -n100 cy_matrix_product(X, Y)
```

    21.4 ms ± 1.36 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


The problem is that Cython cannot efficiently optimize work with lists that can contain elements of various types, so we rewrite `matrix_product` using *ndarray*.


```python
X = np.random.randint(-255, 255, size=shape)
Y = np.random.randint(-255, 255, size=shape)
```


```python
%%capture
%%cython -a
import numpy as np

def cy_matrix_product(X, Y):
    n_xrows, n_xcols = X.shape
    n_yrows, n_ycols = Y.shape
    Z = np.zeros((n_xrows, n_ycols), dtype=X.dtype)
    for i in range(n_xrows):
        for k in range(n_ycols):
            for j in range(n_xcols):
                Z[i, k] += X[i, j] * Y[j, k]
    return Z
```


```python
%timeit -n100 cy_matrix_product(X, Y)
```

    176 ms ± 4.65 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


How so! It just got worse, with most of the code still using Python calls. Let's get rid of them by annotating the code with types.


```python
%%capture
%%cython -a
import numpy as np
cimport numpy as np

def cy_matrix_product(np.ndarray X, np.ndarray Y):
    cdef int n_xrows = X.shape[0]
    cdef int n_xcols = X.shape[1]
    cdef int n_yrows = Y.shape[0]
    cdef int n_ycols = Y.shape[1]
    cdef np.ndarray Z
    Z = np.zeros((n_xrows, n_ycols), dtype=X.dtype)
    for i in range(n_xrows):
        for k in range(n_ycols):
            for j in range(n_xcols):
                Z[i, k] += X[i, j] * Y[j, k]
    return Z
```


```python
%timeit -n100 cy_matrix_product(X, Y)
```

    173 ms ± 4 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


Unfortunately, typical annotations did not change the run time, because the body of the nested Cython itself could not optimize. Fatality-time: indicate the type of elements in *ndarray*.


```python
%%capture
%%cython -a
import numpy as np
cimport numpy as np

def cy_matrix_product(np.ndarray[np.int64_t, ndim=2] X,
                      np.ndarray[np.int64_t, ndim=2] Y):
    cdef int n_xrows = X.shape[0]
    cdef int n_xcols = X.shape[1]
    cdef int n_yrows = Y.shape[0]
    cdef int n_ycols = Y.shape[1]
    cdef np.ndarray[np.int64_t, ndim=2] Z = \
        np.zeros((n_xrows, n_ycols), dtype=np.int64)
    for i in range(n_xrows):
        for k in range(n_ycols):
            for j in range(n_xcols):
                Z[i, k] += X[i, j] * Y[j, k]
    return Z
```


```python
%timeit -n100 cy_matrix_product(X, Y)
```

    541 µs ± 5.14 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Let's try to go further and disable checks for going beyond the boundaries of the array and overflow of integer types.


```python
%%capture
%%cython -a
import numpy as np

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.overflowcheck(False)
def cy_matrix_product(np.ndarray[np.int64_t, ndim=2] X, 
                      np.ndarray[np.int64_t, ndim=2] Y):
    cdef int n_xrows = X.shape[0]
    cdef int n_xcols = X.shape[1]
    cdef int n_yrows = Y.shape[0]
    cdef int n_ycols = Y.shape[1]
    cdef np.ndarray[np.int64_t, ndim=2] Z = \
        np.zeros((n_xrows, n_ycols), dtype=np.int64)
    for i in range(n_xrows):        
        for k in range(n_ycols):
            for j in range(n_xcols):
                Z[i, k] += X[i, j] * Y[j, k]
    return Z
```


```python
%timeit -n100 cy_matrix_product(X, Y)
```

    226 µs ± 2.84 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# Numpy


```python
import numpy as np

X = np.random.randint(-255, 255, shape)
Y = np.random.randint(-255, 255, shape)
```


```python
%timeit -n100 X.dot(Y)
```

    151 µs ± 4.01 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%timeit -n100 X*Y
```

    6.02 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Profit.
