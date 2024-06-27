import numpy as np
from simplex import standardize, twoPhase

n, u, v = map(int, input().split())
c = np.asarray(input().split(), dtype=np.float64)
A = []
for _ in range(u + v):
  A.append(input().split())
A = np.asarray(A, dtype=np.float64)
b = np.asarray(input().split(), dtype=np.float64)


A, b, c, b_idx, n_idx = standardize(n, u, v, A, b, c)
out = twoPhase(n, u, v, A, b, c, b_idx, n_idx)
if out[1] is not None:
  print("{:0.7f}".format(out[1]))
  for i in range(len(out[5])): print("{:0.7f}".format(out[5][i]), end=' ')
