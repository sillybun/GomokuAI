import numba as nb
import numpy as np

from numba import jit, jitclass, int64, f8

@jitclass([('b', nb.int64[:]), ('c', f8[:])])
class A(object):

    def __init__(self):
        self.b = np.zeros(10, dtype=np.int64)
        self.c = np.zeros(3)

    def read(self, C, index):
        return C[self.b[index]]

a = A()
print(a.b)
C = np.zeros(10)
print(a.read(C, 2))
