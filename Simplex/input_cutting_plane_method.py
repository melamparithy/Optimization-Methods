import numpy as np
from simplex import gomory

n, u, v = map(int, input().split())
c = np.asarray(input().split(), dtype=np.float64)
A = []
for _ in range(u + v):
  A.append(input().split())
A = np.asarray(A, dtype=np.float64)
b = np.asarray(input().split(), dtype=np.float64)

'''
n , u , v = 2, 2, 0
c = np.array([-5, -8], dtype=np.float64)
A = np.array([[1, 1], [5, 9]], dtype=np.float64)
b = np.array([6, 45], dtype=np.float64)
'''

out = gomory(n, u, v, A, b, c)
if out[1] is not None:
  print("{:0.7f}".format(out[1]))
  for i in range(len(out[5])): print("{:0.7f}".format(out[5][i]), end=' ')