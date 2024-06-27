import numpy as np
from simplex import standardize, twoPhase, bBound

num = int(input())
C = []
for _ in range(num):
  C.append(input().split())
C = np.asarray(C, dtype=np.float64)

bBound(num, C)