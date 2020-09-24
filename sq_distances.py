import numpy as np

import sys
import os
from time import time

sys.path.append(os.getcwd())
tics = []


# functions to compute square distances
def app0(a, A):  # Original approach
    d = []
    for i in range(0, A.shape[1]):
        d.append(np.linalg.norm(a - A[:, i]))
    return np.argmin(d)


def app1(a, A):
    subs = (a[:, None] - A)
    sq_dist = np.einsum('ij,ij->j', subs, subs)
    return sq_dist.argmin()


def app2(a, A):
    sq_dist = (A ** 2).sum(0) + a.dot(a) - 2 * a.dot(A)
    return sq_dist.argmin()


def app3(a, A):
    sq_dist = np.einsum('ij,ij->j', A, A) + a.dot(a) - 2 * a.dot(A)
    return sq_dist.argmin()


def matrix_sq_dist(x, y):
    [nx, dx] = x.shape
    [ny, dy] = y.shape

    # if dx != dy:
    #     print('Dimension mismatch...', file=sys.stderr)

    # sq_dist = np.asmatrix((x ** 2)).sum(axis=1).dot(np.ones((1, ny))) \
    #             + np.ones((nx, 1)).dot(np.asmatrix((y ** 2)).sum(axis=1).transpose()) \
    #             - 2 * x.dot(y.transpose())

    sq_dist = np.matrix(np.einsum('ij,ij->i', x, x)).transpose().dot(np.ones((1, ny))) \
                + np.ones((nx, 1)).dot(np.matrix(np.einsum('ij,ij->i', y, y))) \
                - 2 * x.dot(y.transpose())

    return sq_dist


# functions for tic & toc
def tic():
    tics.append(time())


def toc():
    if len(tics) == 0:
        return None
    else:
        tics.append(time())


if __name__ == '__main__':
    loops = 1000

    # Vector distance
    n = 128
    m = 2000
    a = np.random.rand(n)  # (128, 1)
    A = np.random.rand(n, m)  # (128, 2000)

    tic()  # profile the matrix operations
    i = loops
    while i > 0:
        d = app3(a, A)  # app0, app1, app2, etc.
        # print('i =', loops - i + 1, '- d =',d)
        i = i - 1
    toc()

    delay = tics[1] - tics[0]
    print('Total delay    = ', delay * 1000, ' ms')
    print('Delay per loop = ', delay / loops * 1000, ' ms')

    # dim = 12
    # nx = 250
    # ny = 1526

    X = np.arange(1, 9).reshape((2, 4)).transpose()
    Y = np.arange(10, 20).reshape((2, 5)).transpose()

    # X = np.random.rand(nx, dim)  # ( 128, 10)
    # Y = np.random.rand(ny, dim)  # (2000, 10)

    tic()  # profile the matrix operations
    i = loops
    while i > 0:
        D2 = matrix_sq_dist(X, Y)
        i = i - 1
    toc()

    delay = tics[3] - tics[2]
    print('\nTotal delay    = ', delay * 1000, ' ms')
    print('Delay per loop = ', delay / loops * 1000, ' ms')
