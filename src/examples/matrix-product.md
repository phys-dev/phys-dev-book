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
        return ((0, 0) if not self else (len(self), len(self[0])))
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

    52.5 ms ± 1.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


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
!pip install line_profiler
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: line_profiler in /home/fedorov/.local/lib/python3.12/site-packages (5.0.0)



```python
%load_ext line_profiler
%lprun -f matrix_product bench()
```


    Timer unit: 1e-09 s
    
    Total time: 5.31022 s
    File: /tmp/ipykernel_7428/3720342099.py
    Function: matrix_product at line 1
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         1                                           def matrix_product(X, Y):
         2                                               """Computes the matrix product of X and Y.
         3                                           
         4                                               >>> X = Matrix([[1], [2], [3]])
         5                                               >>> Y = Matrix([[4, 5, 6]])
         6                                               >>> matrix_product(X, Y)
         7                                               [[4, 5, 6], [8, 10, 12], [12, 15, 18]]
         8                                               >>> matrix_product(Y, X)
         9                                               [[32]]
        10                                               """
        11        16      87526.0   5470.4      0.0      n_xrows, n_xcols = X.shape
        12        16      31645.0   1977.8      0.0      n_yrows, n_ycols = Y.shape
        13                                               # верим, что с размерностями всё хорошо
        14        16     869157.0  54322.3      0.0      Z = Matrix.zeros((n_xrows, n_ycols))
        15      1040     582911.0    560.5      0.0      for i in range(n_xrows):
        16     66560   34411875.0    517.0      0.6          for j in range(n_xcols):
        17   4259840 2224753558.0    522.3     41.9              for k in range(n_ycols):
        18   4194304 3049412607.0    727.0     57.4                  Z[i][k] += X[i][j] * Y[j][k]
        19        16      69070.0   4316.9      0.0      return Z


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


    Timer unit: 1e-09 s
    
    Total time: 4.99064 s
    File: /tmp/ipykernel_7428/481365942.py
    Function: matrix_product at line 1
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         1                                           def matrix_product(X, Y):
         2        16      94661.0   5916.3      0.0      n_xrows, n_xcols = X.shape
         3        16      29810.0   1863.1      0.0      n_yrows, n_ycols = Y.shape
         4        16     375080.0  23442.5      0.0      Z = Matrix.zeros((n_xrows, n_ycols))
         5      1040     542518.0    521.7      0.0      for i in range(n_xrows):
         6      1024     633692.0    618.8      0.0          Xi = X[i]
         7     66560   35416069.0    532.1      0.7          for k in range(n_ycols):
         8     65536   34007986.0    518.9      0.7              acc = 0
         9   4259840 2168283486.0    509.0     43.4              for j in range(n_xcols):
        10   4194304 2710807108.0    646.3     54.3                  acc += Xi[j] * Y[j][k]
        11     65536   40405377.0    616.5      0.8              Z[i][k] = acc
        12        16      47729.0   2983.1      0.0      return Z


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


    Timer unit: 1e-09 s
    
    Total time: 2.80204 s
    File: /tmp/ipykernel_7428/1886970874.py
    Function: matrix_product at line 1
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         1                                           def matrix_product(X, Y):
         2        16      58653.0   3665.8      0.0      n_xrows, n_xcols = X.shape
         3        16      28991.0   1811.9      0.0      n_yrows, n_ycols = Y.shape
         4        16     284438.0  17777.4      0.0      Z = Matrix.zeros((n_xrows, n_ycols))
         5      1040     566052.0    544.3      0.0      for i in range(n_xrows):
         6      1024     776516.0    758.3      0.0          Xi, Zi = X[i], Z[i]
         7     66560   38053714.0    571.7      1.4          for k in range(n_ycols):
         8     65536 2762244176.0  42148.5     98.6              Zi[k] = sum(Xi[j] * Y[j][k] for j in range(n_xcols))
         9        16      24767.0   1547.9      0.0      return Z


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


```python
%lprun -f matrix_product bench()
```


    Timer unit: 1e-09 s
    
    Total time: 2.57215 s
    File: /tmp/ipykernel_7428/3637370621.py
    Function: matrix_product at line 1
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         1                                           def matrix_product(X, Y):
         2        16      60198.0   3762.4      0.0      n_xrows, n_xcols = X.shape
         3        16      29067.0   1816.7      0.0      n_yrows, n_ycols = Y.shape
         4        16     244289.0  15268.1      0.0      Z = Matrix.zeros((n_xrows, n_ycols))
         5        16    1107858.0  69241.1      0.0      Yt = Y.transpose()  # <--
         6      1040     838559.0    806.3      0.0      for i, (Xi, Zi) in enumerate(zip(X, Z)):
         7     66560   39077755.0    587.1      1.5          for k, Ytk in enumerate(Yt):
         8     65536 2530764392.0  38616.4     98.4              Zi[k] = sum(Xi[j] * Ytk[j] for j in range(n_xcols))
         9        16      25251.0   1578.2      0.0      return Z


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

    The slowest run took 43.72 times longer than the fastest. This could mean that an intermediate result is being cached.
    1.52 ms ± 3.18 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


# Cython


```python
!pip install Cython
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: Cython in /home/fedorov/.local/lib/python3.12/site-packages (3.2.0)



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

    22.7 ms ± 533 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)


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

    132 ms ± 2.8 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


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

    131 ms ± 1.03 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)


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

    556 μs ± 8.42 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)


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

    232 μs ± 1.94 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# Numpy


```python
import numpy as np

X = np.random.randint(-255, 255, shape)
Y = np.random.randint(-255, 255, shape)
```


```python
%timeit -n100 X.dot(Y)
```

    225 μs ± 3.3 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%timeit -n100 X*Y
```

    5.84 μs ± 3.84 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Profit.


```python

```


```python

```


```python

```
